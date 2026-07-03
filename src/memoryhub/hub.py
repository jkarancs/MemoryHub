"""``Hub`` — the single public facade over the engine.

Everything (CLI, MCP server, future vector search) goes through ``Hub``; callers should not
reach into ``loader``/``writer``/``query`` directly.

Reads are served from an in-memory doc cache that invalidates on write and on file changes (a
cheap stat-scan — path + mtime + size — on every access; fine at this scale). ``search`` is the
retrieval entry point: hybrid (vector + fulltext, rank-fused) by default, degrading to plain
fulltext — with an :class:`~memoryhub.index.IndexWarning` — when the vector stack isn't
installed or no index has been built yet.
"""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from . import loader, query, writer
from .config import Config, load_config
from .embeddings import EmbeddingError, content_hash
from .index import IndexWarning, ReindexStats, VectorIndex
from .loader import StoreReport
from .models import MemoryDoc
from .profiles import Profile, load_profile

_SEARCH_MODES = ("hybrid", "vector", "text")

#: One stat-scan entry per file: (mtime_ns, size).
_Snapshot = dict[Path, tuple[int, int]]


class Hub:
    """The engine facade a content repo talks to."""

    def __init__(self, config: Config | str | Path) -> None:
        if isinstance(config, Config):
            self.config = config
        else:
            self.config = load_config(config)
        self.profile: Profile = load_profile(self.config.hub.profile)
        self._cache: list[MemoryDoc] | None = None
        self._cache_snapshot: _Snapshot | None = None
        self._vector_index: VectorIndex | None = None

    # --- cache -------------------------------------------------------------------

    def _snapshot(self) -> _Snapshot:
        snap: _Snapshot = {}
        for path in loader.iter_store_paths(self.config.content_root):
            stat = path.stat()
            snap[path] = (stat.st_mtime_ns, stat.st_size)
        return snap

    def _invalidate(self) -> None:
        self._cache = None
        self._cache_snapshot = None

    # --- reads -------------------------------------------------------------------

    def all(self) -> list[MemoryDoc]:
        snap = self._snapshot()
        if self._cache is None or snap != self._cache_snapshot:
            self._cache = loader.load_all(self.config)
            self._cache_snapshot = snap
        return list(self._cache)

    def get(self, id: str) -> MemoryDoc:
        for doc in self.all():
            if doc.id == id:
                return doc
        raise KeyError(f"no memory with id {id!r}")

    def filter(self, **kw: Any) -> list[MemoryDoc]:
        return query.filter(self.all(), **kw)

    def fulltext(self, term: str, **kw: Any) -> list[MemoryDoc]:
        return query.fulltext(self.all(), term, **kw)

    def list_types(self) -> list[str]:
        """The profile's closed type vocabulary (not just the types present in the store)."""
        return self.profile.type_names

    def list_tags(self) -> dict[str, int]:
        """Tag -> usage count over the whole store, most-used first."""
        counts = Counter(tag for doc in self.all() for tag in doc.frontmatter.tags)
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def validate(self) -> StoreReport:
        """Full-store validation report (issues + unresolved-``related`` warnings)."""
        return loader.validate_store(self.config)

    # --- writes ------------------------------------------------------------------

    def add(self, **fields: Any) -> MemoryDoc:
        """Create a memory; ``body`` is taken from the keyword of the same name."""
        body = fields.pop("body", "")
        try:
            return writer.add(self.config, fields, body)
        finally:
            self._invalidate()

    def update(self, id: str, **patch: Any) -> MemoryDoc:
        """Patch a memory; ``body`` is taken from the keyword of the same name."""
        body = patch.pop("body", None)
        try:
            return writer.update(self.config, id, fields=patch, body=body)
        finally:
            self._invalidate()

    def delete(self, id: str) -> None:
        try:
            writer.delete(self.config, id)
        finally:
            self._invalidate()

    # --- vectors ---------------------------------------------------------------------

    def _index(self) -> VectorIndex:
        if self._vector_index is None:
            self._vector_index = VectorIndex(self.config)
        return self._vector_index

    def _usable_index(self) -> VectorIndex | None:
        """The vector index, or ``None`` (with an :class:`IndexWarning`) when it can't serve."""
        try:
            index = self._index()
        except ModuleNotFoundError as exc:
            warnings.warn(f"{exc} — falling back to fulltext", IndexWarning, stacklevel=3)
            return None
        if not index.exists():
            warnings.warn(
                f"no vector index at {self.config.index_path} (run `hub reindex`) "
                "— falling back to fulltext",
                IndexWarning,
                stacklevel=3,
            )
            return None
        return index

    def _warn_if_stale(self, index: VectorIndex, docs: list[MemoryDoc]) -> None:
        indexed = index.content_hashes()
        current = {doc.id: content_hash(doc) for doc in docs}
        if indexed != current:
            warnings.warn(
                "vector index is stale (docs changed since the last `hub reindex`); "
                "results may miss recent edits",
                IndexWarning,
                stacklevel=4,
            )

    def search(
        self,
        q: str,
        mode: str = "hybrid",
        *,
        limit: int = 10,
        type: str | None = None,
        tags: list[str] | None = None,
        status: str | None = None,
        visibility: str | None = None,
    ) -> list[MemoryDoc]:
        """Search the store: ``hybrid`` (default) fuses vector and fulltext rankings via RRF.

        ``vector`` is pure ANN over the LanceDB index; ``text`` is the ranked fulltext search.
        Frontmatter filters always apply (scalar ones are also pushed down to the index as
        ``where`` clauses; ``tags`` is list-valued and filtered here). A missing/unavailable
        index degrades hybrid/vector to fulltext with an :class:`IndexWarning`; a stale index
        warns but still serves.
        """
        if mode not in _SEARCH_MODES:
            raise ValueError(f"unknown search mode {mode!r} (expected one of {_SEARCH_MODES})")
        allowed = self.filter(type=type, tags=tags, status=status, visibility=visibility)

        index = self._usable_index() if mode != "text" else None
        if index is None:
            return query.fulltext(allowed, q)[:limit]
        self._warn_if_stale(index, self.all())

        try:
            # Over-fetch: ids outside `allowed` (tags filter, deleted files) are dropped below,
            # and hybrid fusion needs depth beyond `limit` from each ranking to be meaningful.
            ranked_ids = index.search(
                q,
                k=max(limit * 5, 50),
                type=type,
                status=status,
                visibility=visibility,
            )
        except EmbeddingError as exc:
            warnings.warn(f"{exc} — falling back to fulltext", IndexWarning, stacklevel=2)
            return query.fulltext(allowed, q)[:limit]

        by_id = {doc.id: doc for doc in allowed}
        vector_ranking = [by_id[i] for i in ranked_ids if i in by_id]
        if mode == "vector":
            return vector_ranking[:limit]
        text_ranking = query.fulltext(allowed, q)
        return query.rrf_fuse([vector_ranking, text_ranking])[:limit]

    def reindex(self, *, full: bool = False) -> ReindexStats:
        """Sync the vector index to the store; ``full=True`` re-embeds even unchanged docs.

        Incremental (the default) re-embeds only docs whose embedded content changed and drops
        rows for deleted ids. Needs the ``vectors`` extra plus a working embedding backend.
        """
        return self._index().reindex(self.all(), full=full)
