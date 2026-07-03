"""Embedding tests: embed-text/hash contract, backend selection, both backends via fakes.

No model downloads, no network: sentence-transformers and openai are stand-ins injected into
``sys.modules``, so these run everywhere (CI included).
"""

from __future__ import annotations

import math
import sys
import types
from collections.abc import Sequence
from datetime import date
from typing import Any

import pytest

from memoryhub import Config, EmbeddingError, Frontmatter, MemoryDoc, get_embedder
from memoryhub.embeddings import (
    EMBED_CHAR_BUDGET,
    ApiEmbedder,
    LocalEmbedder,
    content_hash,
    embedding_text,
    normalize,
)


def make_doc(
    *,
    title: str = "Async Python",
    description: str = "asyncio in production",
    tags: list[str] | None = None,
    body: str = "Event loops and task groups.",
) -> MemoryDoc:
    fm = Frontmatter(
        id="skill-async-python",
        title=title,
        type="skill",
        description=description,
        tags=tags if tags is not None else ["python", "async"],
        status="active",
        visibility="private",
        created=date(2026, 1, 1),
        updated=date(2026, 1, 2),
    )
    return MemoryDoc(frontmatter=fm, body=body)


def make_config(backend: str = "local", **embed_overrides: Any) -> Config:
    return Config.model_validate(
        {"hub": {"name": "test"}, "embeddings": {"backend": backend, **embed_overrides}}
    )


# --- embedding text + content hash ---------------------------------------------------


def test_embedding_text_combines_title_description_tags_body() -> None:
    text = embedding_text(make_doc())
    assert text.splitlines() == [
        "Async Python",
        "asyncio in production",
        "python async",
        "Event loops and task groups.",
    ]


def test_embedding_text_skips_empty_parts() -> None:
    doc = make_doc(description="", tags=[], body="")
    assert embedding_text(doc) == "Async Python"


def test_embedding_text_truncates_long_bodies() -> None:
    doc = make_doc(body="x" * (EMBED_CHAR_BUDGET * 2))
    assert len(embedding_text(doc)) == EMBED_CHAR_BUDGET


def test_content_hash_tracks_embedded_content_only() -> None:
    doc = make_doc()
    assert content_hash(doc) == content_hash(make_doc())
    assert content_hash(doc) != content_hash(make_doc(body="Different body."))
    assert content_hash(doc) != content_hash(make_doc(tags=["python"]))
    # status/visibility/dates are not embedded, so they don't touch the hash.
    flipped = make_doc()
    flipped.frontmatter.status = "draft"
    assert content_hash(flipped) == content_hash(doc)


def test_normalize_unit_length_and_zero_safe() -> None:
    unit = normalize([3.0, 4.0])
    assert math.isclose(math.hypot(*unit), 1.0)
    assert normalize(unit) == pytest.approx(unit)  # idempotent
    assert normalize([0.0, 0.0]) == [0.0, 0.0]


# --- local backend --------------------------------------------------------------------


class FakeSentenceTransformer:
    calls: list[dict[str, Any]] = []

    def __init__(self, model: str, device: str) -> None:
        self.model = model
        self.device = device

    def encode(self, texts: Sequence[str], **kwargs: Any) -> list[Any]:
        FakeSentenceTransformer.calls.append({"texts": list(texts), **kwargs})

        class Vector(list):
            def tolist(self) -> list[float]:
                return list(self)

        return [Vector([float(len(text)), 1.0]) for text in texts]


@pytest.fixture
def fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> type[FakeSentenceTransformer]:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    FakeSentenceTransformer.calls = []
    return FakeSentenceTransformer


def test_get_embedder_local_uses_configured_model_and_device(
    fake_sentence_transformers: type[FakeSentenceTransformer],
) -> None:
    config = make_config("local", local={"model": "my/model", "device": "cpu"})
    embedder = get_embedder(config)
    assert isinstance(embedder, LocalEmbedder)
    assert embedder._model.model == "my/model"  # type: ignore[attr-defined]
    assert embedder._model.device == "cpu"  # type: ignore[attr-defined]


def test_local_embedder_batches_and_normalizes(
    fake_sentence_transformers: type[FakeSentenceTransformer],
) -> None:
    embedder = LocalEmbedder(model="m", device="cpu")
    vectors = embedder.embed(["ab", "abcd"])
    assert vectors == [[2.0, 1.0], [4.0, 1.0]]
    (call,) = fake_sentence_transformers.calls
    assert call["normalize_embeddings"] is True
    assert call["batch_size"] > 0
    assert embedder.embed([]) == []


def test_local_embedder_missing_dependency_hint() -> None:
    if "sentence_transformers" in sys.modules or _importable("sentence_transformers"):
        pytest.skip("sentence-transformers is installed; missing-dep path not reachable")
    with pytest.raises(EmbeddingError, match=r"memoryhub\[local-embed\]"):
        LocalEmbedder(model="m", device="cpu")


def _importable(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


# --- api backend ------------------------------------------------------------------------


class FakeOpenAI:
    last: FakeOpenAI | None = None

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.requests: list[dict[str, Any]] = []
        self.embeddings = types.SimpleNamespace(create=self._create)
        FakeOpenAI.last = self

    def _create(self, model: str, input: list[str]) -> Any:  # noqa: A002 - OpenAI's name
        self.requests.append({"model": model, "input": list(input)})
        # Return items deliberately out of order to prove we re-sort by .index.
        data = [
            types.SimpleNamespace(index=i, embedding=[float(len(text)), 3.0])
            for i, text in enumerate(input)
        ]
        return types.SimpleNamespace(data=list(reversed(data)))


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("openai")
    module.OpenAI = FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    FakeOpenAI.last = None


def test_get_embedder_api_reads_key_from_env(
    fake_openai: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    embedder = get_embedder(make_config("api"))
    assert isinstance(embedder, ApiEmbedder)
    assert FakeOpenAI.last is not None
    assert FakeOpenAI.last.api_key == "sk-test"
    assert FakeOpenAI.last.base_url == "https://openrouter.ai/api/v1"


def test_get_embedder_api_missing_key_errors(
    fake_openai: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(EmbeddingError, match="OPENROUTER_API_KEY"):
        get_embedder(make_config("api"))


def test_api_embedder_restores_order_and_normalizes(fake_openai: None) -> None:
    embedder = ApiEmbedder(base_url="https://x", model="m", api_key="k")
    vectors = embedder.embed(["abcd", "abc"])
    assert vectors == [normalize([4.0, 3.0]), normalize([3.0, 3.0])]
    assert math.isclose(math.hypot(*vectors[0]), 1.0)


def test_api_embedder_batches_large_inputs(fake_openai: None) -> None:
    embedder = ApiEmbedder(base_url="https://x", model="m", api_key="k")
    texts = [f"text-{i}" for i in range(130)]
    vectors = embedder.embed(texts)
    assert len(vectors) == 130
    assert FakeOpenAI.last is not None
    sizes = [len(request["input"]) for request in FakeOpenAI.last.requests]
    assert sizes == [64, 64, 2]
