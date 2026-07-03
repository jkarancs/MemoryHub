"""Hub.search / Hub.reindex: hybrid fusion, modes, filters, fallback + staleness warnings.

The embedder is a deterministic fake injected by monkeypatching ``get_embedder`` (as resolved
by the index module), so vector behaviour is exercised end-to-end through LanceDB without any
model. The fallback tests run even where lancedb *is* installed by faking its absence.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from conftest import TopicEmbedder, block_module, write_memory
from memoryhub import Hub, IndexWarning, load_config


@pytest.fixture
def hub(seeded_repo: Path) -> Hub:
    # Semantic-only target: about vector search without ever saying "vector databases".
    write_memory(
        seeded_repo,
        id="skill-embeddings",
        type="skill",
        title="Embeddings",
        description="LanceDB tables and ANN retrieval",
        tags="[ml]",
        extras={"proficiency": "intermediate"},
        body="Upserting into lancedb and querying nearest neighbours.",
    )
    return Hub(load_config(seeded_repo))


@pytest.fixture
def indexed_hub(hub: Hub, monkeypatch: pytest.MonkeyPatch) -> Hub:
    pytest.importorskip("lancedb")
    monkeypatch.setattr("memoryhub.index.get_embedder", lambda config: TopicEmbedder())
    hub.reindex()
    return hub


# --- fallback paths (run everywhere, even without lancedb) ---------------------------


def test_search_without_vectors_extra_falls_back_to_fulltext(
    hub: Hub, monkeypatch: pytest.MonkeyPatch
) -> None:
    block_module(monkeypatch, "lancedb")
    with pytest.warns(IndexWarning, match=r"memoryhub\[vectors\]"):
        docs = hub.search("asyncio")
    assert [doc.id for doc in docs] == [doc.id for doc in hub.fulltext("asyncio")]


def test_search_without_index_built_falls_back_to_fulltext(hub: Hub) -> None:
    pytest.importorskip("lancedb")
    with pytest.warns(IndexWarning, match="run `hub reindex`"):
        docs = hub.search("asyncio")
    assert docs and docs[0].id == "skill-async-python"


def test_text_mode_never_touches_the_index(hub: Hub, monkeypatch: pytest.MonkeyPatch) -> None:
    block_module(monkeypatch, "lancedb")
    with warnings.catch_warnings():
        warnings.simplefilter("error", IndexWarning)  # any fallback warning would fail
        docs = hub.search("asyncio", mode="text", limit=2)
    assert [doc.id for doc in docs] == [doc.id for doc in hub.fulltext("asyncio")[:2]]


def test_search_unknown_mode_rejected(hub: Hub) -> None:
    with pytest.raises(ValueError, match="unknown search mode"):
        hub.search("x", mode="cosine")


def test_reindex_without_vectors_extra_raises_hint(
    hub: Hub, monkeypatch: pytest.MonkeyPatch
) -> None:
    block_module(monkeypatch, "lancedb")
    with pytest.raises(ModuleNotFoundError, match=r"memoryhub\[vectors\]"):
        hub.reindex()


# --- vector + hybrid behaviour (need lancedb) ------------------------------------------


def test_hybrid_finds_semantic_match_that_fulltext_misses(indexed_hub: Hub) -> None:
    query = "vector databases"
    assert all(doc.id != "skill-embeddings" for doc in indexed_hub.fulltext(query))
    docs = indexed_hub.search(query)
    assert docs[0].id == "skill-embeddings"


def test_hybrid_boosts_the_lexical_side_too(indexed_hub: Hub) -> None:
    # "task groups" carries no topic axis for the fake embedder (the vector ranking is all
    # ties), so the exact-phrase fulltext hit must win the fusion.
    docs = indexed_hub.search("task groups")
    assert docs[0].id == "skill-async-python"


def test_vector_mode_is_pure_ann_ranking(indexed_hub: Hub) -> None:
    docs = indexed_hub.search("lancedb", mode="vector", limit=3)
    assert docs[0].id == "skill-embeddings"


def test_search_applies_frontmatter_filters(indexed_hub: Hub) -> None:
    docs = indexed_hub.search("async", type="skill")
    assert {doc.type for doc in docs} == {"skill"}
    docs = indexed_hub.search("async", tags=["python", "async"])
    assert [doc.id for doc in docs] == ["skill-async-python"]
    docs = indexed_hub.search("async", status="draft")
    assert {doc.frontmatter.status for doc in docs} == {"draft"}
    docs = indexed_hub.search("site", visibility="public")
    assert docs and all(doc.frontmatter.visibility == "public" for doc in docs)


def test_search_respects_limit(indexed_hub: Hub) -> None:
    assert len(indexed_hub.search("async", limit=2)) == 2


def test_fresh_index_serves_without_warnings(indexed_hub: Hub) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", IndexWarning)
        indexed_hub.search("lancedb")


def test_stale_index_warns_but_still_serves(indexed_hub: Hub) -> None:
    indexed_hub.update("skill-sql", description="Query planners and window functions")
    with pytest.warns(IndexWarning, match="stale"):
        docs = indexed_hub.search("lancedb")
    assert docs


def test_hub_reindex_is_incremental(indexed_hub: Hub) -> None:
    indexed_hub.update("skill-sql", description="Query planners and window functions")
    stats = indexed_hub.reindex()
    assert stats.embedded == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", IndexWarning)
        indexed_hub.search("lancedb")  # fresh again


def test_embedding_failure_at_query_time_falls_back(
    hub: Hub, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("lancedb")
    from memoryhub.embeddings import EmbeddingError

    monkeypatch.setattr("memoryhub.index.get_embedder", lambda config: TopicEmbedder())
    hub.reindex()

    class Broken:
        def embed(self, texts: object) -> list[list[float]]:
            raise EmbeddingError("backend unavailable")

    fresh = Hub(hub.config)  # new instance so the index re-resolves its embedder
    monkeypatch.setattr("memoryhub.index.get_embedder", lambda config: Broken())
    with pytest.warns(IndexWarning, match="backend unavailable"):
        docs = fresh.search("asyncio")
    assert docs and docs[0].id == "skill-async-python"
