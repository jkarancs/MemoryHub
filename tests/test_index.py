"""VectorIndex tests over a real (local, tiny) LanceDB table with a deterministic fake embedder.

Needs the ``vectors`` extra (lancedb + pyarrow) — skipped when it isn't installed — but no GPU,
no model download, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("lancedb")

from conftest import TopicEmbedder, write_memory  # noqa: E402
from memoryhub import Hub, VectorIndex, load_config  # noqa: E402
from memoryhub.index import ReindexStats  # noqa: E402


@pytest.fixture
def hub(seeded_repo: Path) -> Hub:
    # One doc that talks about vectors only in LanceDB terms (no literal "vector databases").
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
def embedder() -> TopicEmbedder:
    return TopicEmbedder()


@pytest.fixture
def index(hub: Hub, embedder: TopicEmbedder) -> VectorIndex:
    return VectorIndex(hub.config, embedder=embedder)


def test_reindex_builds_table_from_scratch(index: VectorIndex, hub: Hub) -> None:
    docs = hub.all()
    assert not index.exists()
    stats = index.reindex(docs)
    assert stats == ReindexStats(embedded=len(docs), reused=0, removed=0, total=len(docs))
    assert index.exists()
    assert set(index.content_hashes()) == {doc.id for doc in docs}


def test_reindex_is_incremental_on_content_change(
    index: VectorIndex, hub: Hub, embedder: TopicEmbedder
) -> None:
    index.reindex(hub.all())
    embedder.embedded.clear()

    hub.update("skill-sql", description="Window functions and query plans")
    stats = index.reindex(hub.all())
    assert stats.embedded == 1  # exactly the edited doc
    assert stats.reused == len(hub.all()) - 1
    assert stats.removed == 0
    assert len(embedder.embedded) == 1
    assert "Window functions" in embedder.embedded[0]


def test_reindex_refreshes_metadata_without_reembedding(index: VectorIndex, hub: Hub) -> None:
    index.reindex(hub.all())
    # A status flip changes no embedded content: the vector is reused, the column is fresh.
    hub.update("goal-ai-role", status="aspirational")
    stats = index.reindex(hub.all())
    assert stats.embedded == 0
    assert index.search("career async", k=20, status="aspirational") == ["goal-ai-role"]


def test_reindex_drops_ids_that_left_the_store(index: VectorIndex, hub: Hub) -> None:
    index.reindex(hub.all())
    hub.delete("skill-docker")
    stats = index.reindex(hub.all())
    assert stats.removed == 1
    assert stats.total == len(hub.all())
    assert "skill-docker" not in index.content_hashes()


def test_reindex_full_reembeds_everything(
    index: VectorIndex, hub: Hub, embedder: TopicEmbedder
) -> None:
    index.reindex(hub.all())
    embedder.embedded.clear()
    stats = index.reindex(hub.all(), full=True)
    assert stats.embedded == len(hub.all())
    assert stats.reused == 0
    assert len(embedder.embedded) == len(hub.all())


def test_reindex_empty_store_drops_table(index: VectorIndex, hub: Hub) -> None:
    index.reindex(hub.all())
    stats = index.reindex([])
    assert stats == ReindexStats(embedded=0, reused=0, removed=len(hub.all()), total=0)
    assert not index.exists()


def test_search_ranks_by_similarity(index: VectorIndex, hub: Hub) -> None:
    index.reindex(hub.all())
    ranked = index.search("vector lancedb store", k=3)
    assert ranked[0] == "skill-embeddings"


def test_search_pushes_scalar_filters_down(index: VectorIndex, hub: Hub) -> None:
    index.reindex(hub.all())
    ranked = index.search("async", k=20, type="project")
    assert ranked
    assert set(ranked) <= {"project-hub", "project-site"}
    assert index.search("async", k=20, type="skill", status="active", visibility="public") == []


def test_search_rejects_unknown_filters(index: VectorIndex, hub: Hub) -> None:
    index.reindex(hub.all())
    with pytest.raises(ValueError, match="unsupported index filter"):
        index.search("x", tags="python")  # type: ignore[arg-type]


def test_search_respects_k_and_missing_table(index: VectorIndex, hub: Hub) -> None:
    assert index.search("anything") == []  # no table yet
    index.reindex(hub.all())
    assert len(index.search("async", k=2)) == 2


def test_missing_vectors_extra_raises_actionable_hint(
    hub: Hub, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    real_import = builtins.__import__

    def no_lancedb(name: str, *args: object, **kwargs: object) -> object:
        if name == "lancedb":
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", no_lancedb)
    with pytest.raises(ModuleNotFoundError, match=r"memoryhub\[vectors\]"):
        VectorIndex(hub.config)
