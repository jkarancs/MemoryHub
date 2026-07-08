"""Bundle packing (pure), token counting, stats round-trip, and Hub.recall_bundle/context_stats.

Fixtures are synthetic and content-agnostic (no personal data). The pure packer is exercised with
an injected ``count=len`` so budgets are exact and predictable — with a character counter,
``total_tokens`` equals ``len(text)`` by construction, which makes "never exceed the budget" a
sharp, checkable property. The real token counters are covered separately.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest

from conftest import block_module
from memoryhub import Bundle, Hub, load_config
from memoryhub import tokens as tokens_mod
from memoryhub.bundle import (
    OVER_FETCH,
    Excluded,
    load_stats,
    log_bundle,
    pack,
    render,
    stats_path,
)
from memoryhub.models import Frontmatter, MemoryDoc

CHARS = "chars"  # counter label used with count=len
HEADER = "# Context for: t"  # header for task "t", handy for budget arithmetic


def make_doc(
    id: str,
    *,
    title: str | None = None,
    description: str = "a one line description",
    body: str = "some body text",
    type: str = "skill",
    tags: list[str] | None = None,
) -> MemoryDoc:
    fm = Frontmatter(
        id=id,
        title=title if title is not None else id,
        type=type,
        description=description,
        tags=tags or [],
        status="active",
        visibility="private",
        created=date(2026, 1, 1),
        updated=date(2026, 1, 2),
    )
    return MemoryDoc(frontmatter=fm, body=body)


# --- packing: budget invariant (property-style) ----------------------------------------


def test_pack_never_exceeds_budget_across_random_fixtures() -> None:
    rng = random.Random(20260720)
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()

    def phrase(lo: int, hi: int) -> str:
        return " ".join(rng.choice(words) for _ in range(rng.randint(lo, hi)))

    for _ in range(400):
        hits = [
            make_doc(
                f"m-{i}",
                title=phrase(1, 4),
                description=phrase(0, 12),
                body=phrase(0, 80),
                tags=rng.sample(words, rng.randint(0, 2)),
            )
            for i in range(rng.randint(0, 15))
        ]
        budget = rng.randint(0, 500)
        want_type = rng.choice([None, "skill", "project"])
        result = pack(hits, "t", budget, type=want_type, count=len, counter=CHARS)

        # With a character counter, the reported total is exactly the rendered length.
        assert result.total_tokens == len(result.text)
        # Never exceed the budget (once the budget can at least hold the task header).
        if budget >= len(HEADER):
            assert result.total_tokens <= budget
        # Every hit is accounted for exactly once, as included or excluded.
        assert len(result.manifest) + len(result.excluded) == len(hits)


# --- rendering -------------------------------------------------------------------------


def test_render_levels_and_heading_format() -> None:
    doc = make_doc("skill-x", title="Async Python", description="one liner", body="the full body")
    # "## id — title" (em-dash) is the pinned heading format P3/P11/P14 consume.
    assert render(doc, "title") == "## skill-x — Async Python"
    assert render(doc, "description") == "## skill-x — Async Python\n\none liner"
    assert render(doc, "full") == "## skill-x — Async Python\n\nthe full body"


def test_render_omits_empty_content() -> None:
    doc = make_doc("m", title="T", description="", body="")
    assert render(doc, "description") == "## m — T"  # no dangling separator
    assert render(doc, "full") == "## m — T"


# --- packing: level degradation --------------------------------------------------------


def test_takes_richest_level_that_fits() -> None:
    doc = make_doc("m", title="T", description="short one-line summary", body="x " * 100)
    full, desc, title = render(doc, "full"), render(doc, "description"), render(doc, "title")
    assert len(full) > len(desc) > len(title)  # the three tiers really do shrink

    def level_at(budget: int) -> str | None:
        result = pack([doc], "t", budget, count=len, counter=CHARS)
        return result.manifest[0].level if result.manifest else None

    assert level_at(len(HEADER) + 2 + len(full)) == "full"
    assert level_at(len(HEADER) + 2 + len(desc)) == "description"  # full no longer fits
    assert level_at(len(HEADER) + 2 + len(title)) == "title"  # only the title fits
    assert level_at(len(HEADER) + 2 + len(title) - 1) is None  # nothing fits


def test_manifest_tokens_equal_the_rendered_block() -> None:
    doc = make_doc("m", description="hello there", body="body body body")
    result = pack([doc], "t", 100_000, count=len, counter=CHARS)
    item = result.manifest[0]
    assert item.level == "full"  # generous budget → richest level
    assert item.tokens == len(render(doc, "full"))
    assert item.rank == 1 and item.score == 1.0


# --- packing: floor rules --------------------------------------------------------------


def test_full_body_floor_demotes_low_ranked_hits() -> None:
    # Bodies are long, budget is effectively unlimited: only the rank floor changes levels.
    hits = [make_doc(f"m-{i}", body="lots of body text here " * 8) for i in range(15)]
    result = pack(hits, "t", 1_000_000, count=len, counter=CHARS, full_body_floor_rank=10)
    level_by_rank = {item.rank: item.level for item in result.manifest}
    assert all(level_by_rank[r] == "full" for r in range(1, 11))  # ranks 1..10 keep full body
    assert all(level_by_rank[r] == "description" for r in range(11, 16))  # 11+ demoted


def test_max_rank_excludes_beyond_the_relevance_horizon() -> None:
    hits = [make_doc(f"m-{i}") for i in range(6)]
    result = pack(hits, "t", 1_000_000, count=len, counter=CHARS, max_rank=3)
    assert [item.rank for item in result.manifest] == [1, 2, 3]
    assert [(e.rank, e.reason) for e in result.excluded] == [
        (4, "floor"),
        (5, "floor"),
        (6, "floor"),
    ]


# --- packing: filtering ----------------------------------------------------------------


def test_filter_drops_non_matching_and_reranks_survivors() -> None:
    hits = [
        make_doc("a", tags=["x"]),
        make_doc("b", tags=["y"]),
        make_doc("c", tags=["x"], type="project"),
        make_doc("d", tags=["x"]),
    ]
    result = pack(hits, "t", 1_000_000, type="skill", tags=["x"], count=len, counter=CHARS)
    assert [item.id for item in result.manifest] == ["a", "d"]
    assert [item.rank for item in result.manifest] == [1, 2]  # reranked among survivors
    # "b" fails the tag filter, "c" fails the type filter; both reported, no rank pre-ranking.
    assert result.excluded == [
        Excluded(id="b", reason="filter", rank=None),
        Excluded(id="c", reason="filter", rank=None),
    ]


# --- packing: edges + determinism ------------------------------------------------------


def test_empty_hits_yield_header_only() -> None:
    result = pack([], "t", 2000, count=len, counter=CHARS)
    assert result.manifest == [] and result.excluded == []
    assert result.text == HEADER
    assert result.total_tokens == len(HEADER)


def test_tiny_budget_below_smallest_title_includes_nothing() -> None:
    doc = make_doc("m", title="Title")
    budget = len(HEADER) + 2 + len(render(doc, "title")) - 1  # one short of a title block
    result = pack([doc], "t", budget, count=len, counter=CHARS)
    assert result.manifest == []
    assert result.excluded == [Excluded(id="m", reason="budget", rank=1)]
    assert result.text == HEADER  # header still fits and frames the (empty) result
    assert result.total_tokens <= budget


def test_packing_is_deterministic() -> None:
    hits = [make_doc(f"m-{i}", body="body " * i) for i in range(8)]
    first = pack(hits, "t", 160, count=len, counter=CHARS)
    second = pack(hits, "t", 160, count=len, counter=CHARS)
    assert first.model_dump() == second.model_dump()


def test_counter_label_comes_from_param_then_default() -> None:
    assert pack([make_doc("m")], "t", 2000, count=len, counter=CHARS).counter == CHARS
    assert pack([make_doc("m")], "t", 2000).counter in {"tiktoken", "heuristic"}


# --- token counting: heuristic vs tiktoken (interface parity) --------------------------


@pytest.fixture
def fresh_token_cache() -> Iterator[None]:
    """Reset the memoized encoder around a test that toggles tiktoken's availability."""
    tokens_mod._encoder.cache_clear()
    yield
    tokens_mod._encoder.cache_clear()


def test_heuristic_fallback_when_tiktoken_missing(
    monkeypatch: pytest.MonkeyPatch, fresh_token_cache: None
) -> None:
    block_module(monkeypatch, "tiktoken")
    tokens_mod._encoder.cache_clear()  # re-evaluate the (now blocked) import
    assert tokens_mod.counter_name() == "heuristic"
    assert tokens_mod.count_tokens("abcd") == 1  # 4 // 4
    assert tokens_mod.count_tokens("a" * 40) == 10
    assert tokens_mod.count_tokens("") == 0


def test_tiktoken_path_when_installed(fresh_token_cache: None) -> None:
    pytest.importorskip("tiktoken")
    assert tokens_mod.counter_name() == "tiktoken"
    # Same interface: a non-negative int, empty string is zero.
    assert isinstance(tokens_mod.count_tokens("hello world"), int)
    assert tokens_mod.count_tokens("") == 0
    assert tokens_mod.count_tokens("hello world") >= 1


def test_pack_with_real_counter_stays_within_budget() -> None:
    # Whatever the process counter is, the additive accounting must not overshoot.
    hits = [make_doc(f"m-{i}", body="a sentence of moderate length here. " * 5) for i in range(12)]
    result = pack(hits, "prep for an interview", 200)
    assert result.total_tokens <= 200
    assert result.counter in {"tiktoken", "heuristic"}


# --- stats log round-trip --------------------------------------------------------------


def test_stats_log_and_load_round_trip(tmp_path: Path) -> None:
    included = pack([make_doc("a"), make_doc("b")], "task one", 1_000_000, count=len, counter=CHARS)
    entry = log_bundle(tmp_path, included)
    assert stats_path(tmp_path).exists()
    assert entry["task"] == "task one"
    assert entry["n_included"] == 2 and entry["n_excluded"] == 0
    assert entry["counter"] == CHARS and "ts" in entry

    starved = pack(
        [make_doc(f"m-{i}", body="x " * 50) for i in range(5)],
        "task two",
        40,
        count=len,
        counter=CHARS,
    )
    log_bundle(tmp_path, starved)

    stats = load_stats(tmp_path, last_n=20)
    assert stats["aggregates"]["calls"] == 2
    assert [e["task"] for e in stats["entries"]] == ["task one", "task two"]
    assert 0.0 <= stats["aggregates"]["inclusion_rate"] <= 1.0
    assert stats["aggregates"]["avg_total_tokens"] >= 0


def test_load_stats_empty_store(tmp_path: Path) -> None:
    stats = load_stats(tmp_path)
    assert stats["entries"] == []
    assert stats["aggregates"] == {
        "calls": 0,
        "avg_total_tokens": 0.0,
        "avg_included": 0.0,
        "inclusion_rate": 0.0,
    }


def test_load_stats_skips_malformed_lines(tmp_path: Path) -> None:
    path = stats_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"task": "ok", "total_tokens": 10, "n_included": 1, "n_excluded": 0}\n'
        "not json at all\n"
        "\n",
        encoding="utf-8",
    )
    stats = load_stats(tmp_path)
    assert stats["aggregates"]["calls"] == 1


def test_load_stats_last_n_window(tmp_path: Path) -> None:
    for i in range(5):
        log_bundle(tmp_path, pack([make_doc(f"m{i}")], f"t{i}", 2000, count=len, counter=CHARS))
    assert [e["task"] for e in load_stats(tmp_path, last_n=2)["entries"]] == ["t3", "t4"]
    assert load_stats(tmp_path, last_n=0)["aggregates"]["calls"] == 5  # <=0 → all


# --- Hub integration (real search → pack → log) ----------------------------------------


def test_recall_bundle_fits_budget_and_logs(seeded_repo: Path) -> None:
    hub = Hub(load_config(seeded_repo))
    result = hub.recall_bundle("async python", 300)
    assert isinstance(result, Bundle)
    assert result.total_tokens <= 300
    assert result.manifest, "expected at least one relevant memory"
    assert result.text.startswith("# Context for: async python")
    assert result.manifest[0].id == "skill-async-python"  # fulltext-fallback ranking

    stats = hub.context_stats()
    assert stats["aggregates"]["calls"] == 1
    assert stats["entries"][0]["task"] == "async python"


def test_recall_bundle_type_filter_reports_exclusions(seeded_repo: Path) -> None:
    hub = Hub(load_config(seeded_repo))
    result = hub.recall_bundle("python", 2000, type="skill")
    assert result.manifest
    assert all(hub.get(item.id).type == "skill" for item in result.manifest)
    # "python" also matches non-skill memories (e.g. project-hub); those are filter-excluded.
    assert any(e.reason == "filter" for e in result.excluded)


def test_recall_bundle_no_matches_is_header_only(seeded_repo: Path) -> None:
    hub = Hub(load_config(seeded_repo))
    result = hub.recall_bundle("zzzznotarealtokenanywhere", 2000)
    assert result.manifest == []
    assert result.text == "# Context for: zzzznotarealtokenanywhere"


def test_over_fetch_caps_retrieval(seeded_repo: Path) -> None:
    # The seeded store is small, but the retrieval limit must be the documented over-fetch.
    hub = Hub(load_config(seeded_repo))
    calls: dict[str, int] = {}
    original = hub.search

    def spy(q: str, **kw: object) -> list[MemoryDoc]:
        calls["limit"] = int(kw.get("limit", -1))
        return original(q, **kw)  # type: ignore[arg-type]

    hub.search = spy  # type: ignore[method-assign]
    hub.recall_bundle("async", 500)
    assert calls["limit"] == OVER_FETCH


def test_stats_log_written_under_content_root(seeded_repo: Path) -> None:
    hub = Hub(load_config(seeded_repo))
    hub.recall_bundle("docker", 400)
    log = stats_path(seeded_repo / "memory")
    assert log.exists()
    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["task"] == "docker" and record["budget"] == 400
