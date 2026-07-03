"""Pure (no-ML) querying over already-loaded documents: filters, full-text, rank fusion."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date

from .models import MemoryDoc

#: Ordering for the conventional string proficiency scale (ints pass through as-is).
_PROFICIENCY_RANK = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}


def _proficiency_rank(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return _PROFICIENCY_RANK.get(value.strip().lower())
    return None


def filter(
    docs: Sequence[MemoryDoc],
    *,
    type: str | None = None,
    tags: Sequence[str] | None = None,
    status: str | None = None,
    visibility: str | None = None,
    proficiency_gte: int | None = None,
    updated_after: date | None = None,
) -> list[MemoryDoc]:
    """Filter ``docs`` by frontmatter dimensions.

    ``tags`` is an AND-match; ``proficiency_gte`` ranks beginner=1 … expert=4 (docs without a
    comparable ``proficiency`` are excluded); ``updated_after`` is strict (``updated > cutoff``).
    """
    result: list[MemoryDoc] = []
    for doc in docs:
        fm = doc.frontmatter
        if type is not None and fm.type != type:
            continue
        if status is not None and fm.status != status:
            continue
        if visibility is not None and fm.visibility != visibility:
            continue
        if tags and not all(tag in fm.tags for tag in tags):
            continue
        if proficiency_gte is not None:
            rank = _proficiency_rank(fm.extra.get("proficiency"))
            if rank is None or rank < proficiency_gte:
                continue
        if updated_after is not None and fm.updated <= updated_after:
            continue
        result.append(doc)
    return result


def _field_text(doc: MemoryDoc, field: str) -> str:
    """The searchable text of one field ('tags' joins; unknown names fall back to extras)."""
    if field == "body":
        return doc.body
    if field == "tags":
        return " ".join(doc.frontmatter.tags)
    fm = doc.frontmatter
    if field in type(fm).model_fields and field != "extra":
        return str(getattr(fm, field))
    value = fm.extra.get(field)
    return "" if value is None else str(value)


def fulltext(
    docs: Sequence[MemoryDoc],
    term: str,
    *,
    fields: Sequence[str] = ("title", "description", "body", "tags"),
    regex: bool = False,
) -> list[MemoryDoc]:
    """Case-insensitive full-text search over ``fields``, ranked by total match count.

    ``term`` is a literal unless ``regex=True`` (invalid patterns raise ``re.error``). Ties are
    broken by id so the ranking is deterministic.
    """
    pattern = re.compile(term if regex else re.escape(term), re.IGNORECASE)
    scored: list[tuple[int, MemoryDoc]] = []
    for doc in docs:
        count = sum(len(pattern.findall(_field_text(doc, field))) for field in fields)
        if count:
            scored.append((count, doc))
    scored.sort(key=lambda pair: (-pair[0], pair[1].id))
    return [doc for _, doc in scored]


#: Standard RRF dampening constant (Cormack et al.): rank 1 in one list ≈ rank ~30 in two.
_RRF_K = 60


def rrf_fuse(rankings: Sequence[Sequence[MemoryDoc]], *, k: int = _RRF_K) -> list[MemoryDoc]:
    """Reciprocal-rank fusion: score(d) = Σ over rankings of 1/(k + rank(d)), best first.

    A doc absent from one ranking simply contributes nothing for it. Ties break by id so the
    fused ordering is deterministic.
    """
    scores: dict[str, float] = {}
    docs: dict[str, MemoryDoc] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (k + rank)
            docs.setdefault(doc.id, doc)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [docs[doc_id] for doc_id, _ in ordered]
