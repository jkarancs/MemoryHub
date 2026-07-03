"""Golden-query retrieval eval against the real Phase 3 corpus (marker: ``local``).

CI skips this (``addopts`` deselects the ``local`` marker: no GPU, no corpus, no model). Run it
on a machine with the content repo and the embedding stack installed:

    pytest -m local tests/test_eval.py -rA

The store defaults to a ``personal-memory`` checkout next to this repo; point
``MEMORYHUB_EVAL_STORE`` elsewhere to override. The suite reindexes incrementally first, so the
first run pays the full embedding cost and later runs only re-embed edits.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from memoryhub import Hub

QUERIES_PATH = Path(__file__).with_name("eval_queries.yaml")
DEFAULT_STORE = Path(__file__).resolve().parents[2] / "personal-memory"
REQUIRED_HIT_RATE = 0.8
TOP_K = 5

pytestmark = pytest.mark.local


@pytest.fixture(scope="module")
def corpus_hub() -> Hub:
    pytest.importorskip("lancedb")
    store = Path(os.environ.get("MEMORYHUB_EVAL_STORE", DEFAULT_STORE))
    if not (store / "hub.toml").is_file():
        pytest.skip(f"no content repo at {store} (set MEMORYHUB_EVAL_STORE)")
    hub = Hub(store)
    hub.reindex()
    return hub


def load_queries() -> list[dict]:
    return yaml.safe_load(QUERIES_PATH.read_text(encoding="utf-8"))


def test_eval_file_shape() -> None:
    queries = load_queries()
    assert len(queries) >= 15
    for entry in queries:
        assert entry.keys() == {"query", "expect_any"}
        assert entry["expect_any"]


def test_golden_queries_hit_rate(corpus_hub: Hub) -> None:
    queries = load_queries()
    misses: list[str] = []
    for entry in queries:
        top = [doc.id for doc in corpus_hub.search(entry["query"], limit=TOP_K)]
        if not set(entry["expect_any"]) & set(top):
            misses.append(f"{entry['query']!r}: expected any of {entry['expect_any']}, got {top}")
    hit_rate = 1 - len(misses) / len(queries)
    detail = "\n".join(misses)
    assert (
        hit_rate >= REQUIRED_HIT_RATE
    ), f"hit rate {hit_rate:.0%} < {REQUIRED_HIT_RATE:.0%}; misses:\n{detail}"


def test_semantic_query_beats_fulltext(corpus_hub: Hub) -> None:
    """The Phase 4 acceptance example: a query plain `hub find` can't serve."""
    query = "vector databases"
    fulltext_ids = {doc.id for doc in corpus_hub.fulltext(query)}
    hybrid_ids = [doc.id for doc in corpus_hub.search(query, limit=TOP_K)]
    relevant = {"skill-embeddings-note", "skill-rag"}
    assert relevant & set(hybrid_ids)
    assert set(hybrid_ids) - fulltext_ids, "hybrid should surface docs fulltext missed"
