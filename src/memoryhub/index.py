"""Vector index over the store — a LanceDB table at ``config.index_path``.

One row per doc: ``id, vector, content_hash, type, tags, status, visibility, updated``.
:meth:`VectorIndex.reindex` re-embeds only docs whose :func:`~memoryhub.embeddings.content_hash`
changed (``full=True`` re-embeds everything) and drops rows whose ids left the store; metadata
columns are always refreshed. :meth:`VectorIndex.search` is a plain ANN query returning ranked
ids — hydration back to :class:`~memoryhub.models.MemoryDoc` and rank fusion live in ``Hub``.

``lancedb``/``pyarrow`` are imported lazily (the ``vectors`` extra); the embedder is only
constructed when something actually needs to embed, so a no-op incremental reindex never loads
a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .embeddings import Embedder, content_hash, embedding_text, get_embedder, normalize

if TYPE_CHECKING:
    from .config import Config
    from .models import MemoryDoc

_TABLE_NAME = "memories"

#: Scalar frontmatter columns that can be pushed down as a LanceDB ``where`` clause.
#: ``tags`` is a list column and is filtered by the caller (Hub) instead — list-membership SQL
#: functions are not stable across LanceDB versions.
_WHERE_COLUMNS = ("type", "status", "visibility")


class IndexWarning(UserWarning):
    """Non-fatal vector-index condition (missing index → fulltext fallback, stale index, …)."""


@dataclass(frozen=True)
class ReindexStats:
    """What a reindex did: docs embedded vs. reused, stale rows removed, final row count."""

    embedded: int
    reused: int
    removed: int
    total: int

    def __str__(self) -> str:
        return (
            f"{self.total} doc(s) indexed: {self.embedded} embedded, "
            f"{self.reused} reused, {self.removed} removed"
        )


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class VectorIndex:
    """Thin wrapper over the LanceDB table backing ``Hub.search``/``Hub.reindex``."""

    def __init__(self, config: Config, embedder: Embedder | None = None) -> None:
        try:
            import lancedb
            import pyarrow
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"vector search needs the {exc.name!r} package; "
                'run: pip install "memoryhub[vectors]"'
            ) from exc
        self._pa = pyarrow
        self.config = config
        self._db = lancedb.connect(str(config.index_path))
        self._embedder = embedder

    @property
    def embedder(self) -> Embedder:
        """The configured embedder, constructed on first use (model loads are expensive)."""
        if self._embedder is None:
            self._embedder = get_embedder(self.config)
        return self._embedder

    # --- table access ---------------------------------------------------------------

    def exists(self) -> bool:
        return _TABLE_NAME in self._db.table_names()

    def _rows(self, columns: Sequence[str]) -> list[dict[str, Any]]:
        table = self._db.open_table(_TABLE_NAME)
        return table.to_arrow().select(list(columns)).to_pylist()

    def content_hashes(self) -> dict[str, str]:
        """id → content_hash for every indexed row (used for staleness checks)."""
        if not self.exists():
            return {}
        return {row["id"]: row["content_hash"] for row in self._rows(["id", "content_hash"])}

    # --- write ------------------------------------------------------------------------

    def reindex(self, docs: Sequence[MemoryDoc], *, full: bool = False) -> ReindexStats:
        """Sync the table to ``docs``: embed changed hashes, reuse the rest, drop gone ids."""
        docs = list(docs)
        previous: dict[str, dict[str, Any]] = {}
        if self.exists():
            previous = {row["id"]: row for row in self._rows(["id", "vector", "content_hash"])}
        removed = len(set(previous) - {doc.id for doc in docs})

        if not docs:
            if self.exists():
                self._db.drop_table(_TABLE_NAME)
            return ReindexStats(embedded=0, reused=0, removed=removed, total=0)

        hashes = {doc.id: content_hash(doc) for doc in docs}
        reusable: dict[str, list[float]] = {}
        if not full:
            reusable = {
                doc.id: previous[doc.id]["vector"]
                for doc in docs
                if doc.id in previous and previous[doc.id]["content_hash"] == hashes[doc.id]
            }
        to_embed = [doc for doc in docs if doc.id not in reusable]

        vectors = dict(reusable)
        if to_embed:
            embedded = self.embedder.embed([embedding_text(doc) for doc in to_embed])
            for doc, vector in zip(to_embed, embedded, strict=True):
                vectors[doc.id] = normalize(vector)

        rows = [
            {
                "id": doc.id,
                "vector": vectors[doc.id],
                "content_hash": hashes[doc.id],
                "type": doc.frontmatter.type,
                "tags": list(doc.frontmatter.tags),
                "status": doc.frontmatter.status,
                "visibility": doc.frontmatter.visibility,
                "updated": doc.frontmatter.updated.isoformat(),
            }
            for doc in docs
        ]
        # Full overwrite: exact "remove ids no longer in the store" semantics and always-fresh
        # metadata columns, at a cost that is trivial at this scale (the vectors are reused).
        dimension = len(rows[0]["vector"])
        pa = self._pa
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dimension)),
                pa.field("content_hash", pa.string()),
                pa.field("type", pa.string()),
                pa.field("tags", pa.list_(pa.string())),
                pa.field("status", pa.string()),
                pa.field("visibility", pa.string()),
                pa.field("updated", pa.string()),
            ]
        )
        data = pa.Table.from_pylist(rows, schema=schema)
        self._db.create_table(_TABLE_NAME, data, mode="overwrite")
        return ReindexStats(
            embedded=len(to_embed), reused=len(reusable), removed=removed, total=len(rows)
        )

    # --- read -------------------------------------------------------------------------

    def search(self, query: str, *, k: int = 10, **filters: str | None) -> list[str]:
        """Ranked ids for ``query`` (nearest first); scalar ``filters`` become ``where`` clauses.

        Accepted filters: ``type``, ``status``, ``visibility``. The store is small enough that
        LanceDB runs an exact (flat) search — no ANN index to build or tune.
        """
        unknown = set(filters) - set(_WHERE_COLUMNS)
        if unknown:
            raise ValueError(f"unsupported index filter(s): {', '.join(sorted(unknown))}")
        if not self.exists():
            return []
        vector = normalize(self.embedder.embed([query])[0])
        request = self._db.open_table(_TABLE_NAME).search(vector).limit(k)
        clauses = [
            f"{column} = {_sql_quote(value)}"
            for column, value in filters.items()
            if value is not None
        ]
        if clauses:
            request = request.where(" AND ".join(clauses), prefilter=True)
        # No .select(): projecting columns makes newer lance log a warning about the implicit
        # _distance column on every query; full rows are cheap at this row count.
        return [row["id"] for row in request.to_list()]
