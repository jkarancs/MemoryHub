"""Task-scoped context packing: retrieve, then greedily fill a token budget.

:meth:`Hub.recall_bundle` runs the store's hybrid search for a task and packs the hits into a
markdown *context pack* that never exceeds a token budget. Each memory can be rendered at three
levels — full body → its curated one-line ``description`` → its title alone — and the packer
takes the richest level that still fits, walking hits in relevance order. A fixed floor
(:data:`FULL_BODY_FLOOR_RANK`) denies the full-body level to weakly-ranked hits so one long
memory can't starve the rest.

There is **no LLM summarization here**: the ``description`` field *is* the summary tier. An
injectable summarizer (to add a fourth, model-written level) is left as a seam for P14 — see the
``count`` parameter's sibling ``render`` hook in :func:`pack` (documented, not yet wired).

The packing core (:func:`pack`) is pure — retrieval, config, and instrumentation live in
``Hub`` — so it can be property-tested hard: given any hits and budget, ``total_tokens`` must
never exceed ``budget``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .models import MemoryDoc
from .tokens import count_tokens, counter_name

#: A memory's representation level, richest → leanest.
Level = Literal["full", "description", "title"]
#: Why a retrieved memory didn't make the pack.
ExcludeReason = Literal["budget", "filter", "floor"]

#: Richest → leanest; the packer tries these in order (subject to the full-body floor).
_LEVELS: tuple[Level, ...] = ("full", "description", "title")

#: Default over-fetch: how many hits ``recall_bundle`` retrieves and considers per bundle. This
#: doubles as the relevance horizon (``max_rank``) — hits ranked past it are ``floor``-excluded.
OVER_FETCH = 25
#: Hits ranked below this never get the full-body level (start at ``description``); keeps one
#: long, highly-ranked memory from consuming the whole budget.
FULL_BODY_FLOOR_RANK = 10

_HEADING = "## {id} — {title}"
_HEADER = "# Context for: {task}"
_SEP = "\n\n"


class BundleItem(BaseModel):
    """One packed memory: how it was rendered and what it cost."""

    id: str
    title: str
    rank: int  # 1-based relevance rank among filter-matching hits
    score: float  # rank-derived relevance proxy (1/rank); search returns a ranking, not scores
    level: Level
    tokens: int  # tokens of this memory's rendered block (heading + content at ``level``)


class Excluded(BaseModel):
    """A retrieved memory that didn't make the pack, and why."""

    id: str
    reason: ExcludeReason
    rank: int | None = None  # relevance rank when known (None for ``filter`` drops, pre-ranking)


class Bundle(BaseModel):
    """A budgeted context pack plus the manifest of what's in it and what was dropped."""

    task: str
    text: str
    manifest: list[BundleItem] = Field(default_factory=list)
    excluded: list[Excluded] = Field(default_factory=list)
    total_tokens: int
    budget: int
    counter: str  # "tiktoken" | "heuristic" — how the token numbers were produced


def render(doc: MemoryDoc, level: Level) -> str:
    """Render one memory as a markdown block at ``level`` (``## id — title`` + content).

    ``title`` is heading-only; ``description``/``full`` append the one-liner / the body. This is
    the seam a P14 summarizer would extend with a model-written level.
    """
    heading = _HEADING.format(id=doc.id, title=doc.frontmatter.title)
    if level == "title":
        return heading
    content = doc.frontmatter.description if level == "description" else doc.body
    content = content.strip()
    return f"{heading}{_SEP}{content}" if content else heading


def _matches(doc: MemoryDoc, type: str | None, tags: Sequence[str] | None) -> bool:
    fm = doc.frontmatter
    if type is not None and fm.type != type:
        return False
    if tags and not all(tag in fm.tags for tag in tags):
        return False
    return True


def pack(
    hits: Sequence[MemoryDoc],
    task: str,
    budget: int,
    *,
    type: str | None = None,
    tags: Sequence[str] | None = None,
    count: Callable[[str], int] = count_tokens,
    counter: str | None = None,
    full_body_floor_rank: int = FULL_BODY_FLOOR_RANK,
    max_rank: int | None = None,
) -> Bundle:
    """Greedily pack ``hits`` (in relevance order) into a ``budget``-bounded context pack.

    Pure and deterministic. ``hits`` are assumed already ranked (best first). Packing:

    * **filter** — hits failing ``type``/``tags`` are dropped (reason ``filter``) *before*
      ranking, so a bundle can report what a filter cost you; survivors are ranked ``1..M``;
    * **floor** — a survivor ranked past ``max_rank`` (if set) is dropped (reason ``floor``);
      one ranked past ``full_body_floor_rank`` may not use the full-body level;
    * **fit** — take the richest still-eligible level whose block fits the remaining budget;
      if none fits, drop it (reason ``budget``).

    Token accounting is additive over the header, per-block separators, and each block, so
    ``total_tokens`` equals the reported cost of ``text`` and **never exceeds ``budget``**
    (assuming ``budget`` covers the small task header). ``count``/``counter`` are injectable for
    deterministic tests; they default to the process token counter.
    """
    counter_used = counter if counter is not None else counter_name()
    header = _HEADER.format(task=task)
    sep_tokens = count(_SEP)

    parts: list[str] = [header]
    total = count(header)
    manifest: list[BundleItem] = []
    excluded: list[Excluded] = []

    rank = 0
    for doc in hits:
        if not _matches(doc, type, tags):
            excluded.append(Excluded(id=doc.id, reason="filter"))
            continue
        rank += 1
        if max_rank is not None and rank > max_rank:
            excluded.append(Excluded(id=doc.id, reason="floor", rank=rank))
            continue
        start = 0 if rank <= full_body_floor_rank else 1  # index into _LEVELS
        for level in _LEVELS[start:]:
            block = render(doc, level)
            block_tokens = count(block)
            if total + sep_tokens + block_tokens <= budget:
                parts.append(block)
                total += sep_tokens + block_tokens
                manifest.append(
                    BundleItem(
                        id=doc.id,
                        title=doc.frontmatter.title,
                        rank=rank,
                        score=round(1.0 / rank, 6),
                        level=level,
                        tokens=block_tokens,
                    )
                )
                break
        else:
            excluded.append(Excluded(id=doc.id, reason="budget", rank=rank))

    return Bundle(
        task=task,
        text=_SEP.join(parts),
        manifest=manifest,
        excluded=excluded,
        total_tokens=total,
        budget=budget,
        counter=counter_used,
    )


# --- instrumentation: one JSONL line per bundle -----------------------------------------

#: Context-cost log, relative to the content root (git-ignored in the content repo).
STATS_DIR = ".stats"
STATS_FILE = "context_log.jsonl"


def stats_path(content_root: Path) -> Path:
    return content_root / STATS_DIR / STATS_FILE


def log_bundle(
    content_root: Path, bundle: Bundle, *, now: datetime | None = None
) -> dict[str, Any]:
    """Append one JSON line describing ``bundle`` to the content root's context log.

    This is the tokens-of-context-per-task metric P14 reads back. Returns the written entry.
    """
    entry = {
        "ts": (now or datetime.now(UTC)).isoformat(),
        "task": bundle.task,
        "budget": bundle.budget,
        "total_tokens": bundle.total_tokens,
        "n_included": len(bundle.manifest),
        "n_excluded": len(bundle.excluded),
        "counter": bundle.counter,
    }
    path = stats_path(content_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def load_stats(content_root: Path, last_n: int = 20) -> dict[str, Any]:
    """Read the last ``last_n`` context-log entries plus aggregates (``last_n <= 0`` = all).

    Aggregates over the returned window: call count, average tokens/task, average memories
    included, and the overall inclusion rate (included / (included + excluded)). Malformed lines
    are skipped so a partially-written log still reads.
    """
    path = stats_path(content_root)
    entries: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    recent = entries[-last_n:] if last_n > 0 else entries

    n = len(recent)
    included = sum(int(e.get("n_included", 0)) for e in recent)
    excluded = sum(int(e.get("n_excluded", 0)) for e in recent)
    considered = included + excluded
    aggregates = {
        "calls": n,
        "avg_total_tokens": (
            round(sum(int(e.get("total_tokens", 0)) for e in recent) / n, 1) if n else 0.0
        ),
        "avg_included": round(included / n, 2) if n else 0.0,
        "inclusion_rate": round(included / considered, 3) if considered else 0.0,
    }
    return {"entries": recent, "aggregates": aggregates}
