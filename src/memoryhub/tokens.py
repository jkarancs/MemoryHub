"""Token counting for context budgeting.

:func:`count_tokens` uses tiktoken's ``cl100k_base`` encoding when the optional ``tokens`` extra
is installed, and otherwise falls back to a ``len(text) // 4`` heuristic. The interface is
identical either way; :func:`counter_name` reports which counter is active so a bundle manifest
can record how its numbers were produced (the heuristic drifts from real tokenization, so a
consumer should know which it's looking at).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

#: Names reported by :func:`counter_name` (and stored in bundle manifests).
TIKTOKEN = "tiktoken"
HEURISTIC = "heuristic"

_ENCODING = "cl100k_base"
#: Rough characters-per-token used by the no-dependency fallback.
_CHARS_PER_TOKEN = 4


@lru_cache(maxsize=1)
def _encoder() -> Any | None:
    """The ``cl100k_base`` encoder, or ``None`` when tiktoken isn't installed (memoized).

    Memoized because building the encoder is not free; tests that toggle tiktoken's availability
    must call ``_encoder.cache_clear()`` first.
    """
    try:
        import tiktoken
    except ModuleNotFoundError:
        return None
    return tiktoken.get_encoding(_ENCODING)


def counter_name() -> str:
    """Name of the active counter: ``"tiktoken"`` if importable, else ``"heuristic"``."""
    return TIKTOKEN if _encoder() is not None else HEURISTIC


def count_tokens(text: str) -> int:
    """Tokens in ``text`` — exact (tiktoken ``cl100k_base``) or a ``len // 4`` estimate."""
    encoder = _encoder()
    if encoder is None:
        return len(text) // _CHARS_PER_TOKEN
    return len(encoder.encode(text))
