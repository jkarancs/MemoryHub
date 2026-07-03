"""Embedding backends behind one tiny protocol: ``embed(texts) -> vectors``.

Two implementations, selected by ``[embeddings]`` in ``hub.toml``:

* :class:`LocalEmbedder` — sentence-transformers (default ``BAAI/bge-m3`` on CUDA), batched.
* :class:`ApiEmbedder` — any OpenAI-compatible embeddings endpoint (default OpenRouter); the
  key comes from the env var named by ``api_key_env``, never from the toml.

Both heavy dependencies are imported lazily so the base install stays light. This module also
owns *what* gets embedded: one vector per doc over ``title + description + tags + body``
(:func:`embedding_text`) and the content hash the index uses to skip unchanged docs
(:func:`content_hash`).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .config import Config
    from .models import MemoryDoc

#: Character budget for one doc's embedding text. Docs are small by design (no chunking);
#: this only guards the odd long ``writing`` entry. ~24k chars ≈ 6k tokens, comfortably inside
#: the 8k-token windows of both default models (bge-m3 and text-embedding-3-large).
EMBED_CHAR_BUDGET = 24_000

#: Batch size for embedding calls (both backends; keeps API request bodies bounded).
_BATCH_SIZE = 64


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend cannot be constructed or a call fails."""


class Embedder(Protocol):
    """The interface the vector index depends on."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def embedding_text(doc: MemoryDoc) -> str:
    """The one string per doc that gets embedded: title + description + tags + truncated body."""
    fm = doc.frontmatter
    parts = [fm.title, fm.description, " ".join(fm.tags), doc.body]
    text = "\n".join(part for part in parts if part)
    return text[:EMBED_CHAR_BUDGET]


def content_hash(doc: MemoryDoc) -> str:
    """Hash of exactly what would be embedded — unchanged hash ⇒ the stored vector is reusable."""
    return hashlib.sha256(embedding_text(doc).encode("utf-8")).hexdigest()


def normalize(vector: Sequence[float]) -> list[float]:
    """Unit-normalize so L2 distance ranks identically to cosine similarity (idempotent)."""
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return list(vector)
    return [component / norm for component in vector]


def _batched(texts: Sequence[str]) -> list[Sequence[str]]:
    return [texts[i : i + _BATCH_SIZE] for i in range(0, len(texts), _BATCH_SIZE)]


class LocalEmbedder:
    """sentence-transformers backend; model/device from ``[embeddings.local]``."""

    def __init__(self, model: str, device: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise EmbeddingError(
                "the local embedding backend needs sentence-transformers; "
                'run: pip install "memoryhub[local-embed]"'
            ) from exc
        self.model_name = model
        self.device = device
        self._model = SentenceTransformer(model, device=device)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]


class ApiEmbedder:
    """OpenAI-compatible embeddings endpoint; url/model from ``[embeddings.api]``, key from env."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise EmbeddingError(
                "the API embedding backend needs the openai client; "
                'run: pip install "memoryhub[api-embed]"'
            ) from exc
        self.model_name = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in _batched(texts):
            response = self._client.embeddings.create(model=self.model_name, input=list(batch))
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(normalize(item.embedding) for item in ordered)
        return vectors


def get_embedder(config: Config) -> Embedder:
    """Construct the embedder selected by ``config.embeddings.backend``."""
    backend = config.embeddings.backend
    if backend == "local":
        local = config.embeddings.local
        return LocalEmbedder(model=local.model, device=local.device)
    if backend == "api":
        api = config.embeddings.api
        key = config.resolve_api_key()
        if not key:
            raise EmbeddingError(
                f"embeddings backend is 'api' but the env var {api.api_key_env!r} is not set"
            )
        return ApiEmbedder(base_url=api.base_url, model=api.model, api_key=key)
    raise EmbeddingError(f"unknown embeddings backend {backend!r}")  # pragma: no cover
