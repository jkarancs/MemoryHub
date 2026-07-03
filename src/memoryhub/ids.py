"""Slug/id generation and uniqueness."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug_part(text: str) -> str:
    """Reduce arbitrary text to slug characters: ascii-fold, lowercase, hyphen-join."""
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return _NON_SLUG_RE.sub("-", folded.lower()).strip("-")


def slugify(title: str, type: str) -> str:
    """Build a slug id from a type + title.

    e.g. ``("Async Python", "skill") -> "skill-async-python"``.

    Raises:
        ValueError: if either part reduces to nothing slug-safe.
    """
    type_part = _slug_part(type)
    title_part = _slug_part(title)
    if not type_part:
        raise ValueError(f"type {type!r} contains no slug-safe characters")
    if not title_part:
        raise ValueError(f"title {title!r} contains no slug-safe characters")
    return f"{type_part}-{title_part}"


def ensure_unique(id: str, existing: Iterable[str]) -> str:
    """Return ``id``, or a suffixed variant (``-2``, ``-3``, ...) if it collides."""
    taken = set(existing)
    if id not in taken:
        return id
    n = 2
    while f"{id}-{n}" in taken:
        n += 1
    return f"{id}-{n}"
