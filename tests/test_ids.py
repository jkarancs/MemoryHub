"""Tests for slug/id generation and uniqueness."""

from __future__ import annotations

import pytest

from memoryhub import ids


class TestSlugify:
    def test_basic(self) -> None:
        assert ids.slugify("Async Python", "skill") == "skill-async-python"

    def test_punctuation_collapses_to_single_hyphens(self) -> None:
        assert ids.slugify("C++ & CUDA!", "skill") == "skill-c-cuda"

    def test_accents_are_ascii_folded(self) -> None:
        assert ids.slugify("Café Ötlet", "project") == "project-cafe-otlet"

    def test_leading_trailing_junk_stripped(self) -> None:
        assert ids.slugify("  --Hello--  ", "bio") == "bio-hello"

    def test_empty_title_raises(self) -> None:
        with pytest.raises(ValueError, match="title"):
            ids.slugify("", "skill")

    def test_symbol_only_title_raises(self) -> None:
        with pytest.raises(ValueError, match="title"):
            ids.slugify("→ ↓ ©", "skill")

    def test_empty_type_raises(self) -> None:
        with pytest.raises(ValueError, match="type"):
            ids.slugify("Hello", "")


class TestEnsureUnique:
    def test_no_collision_returns_id(self) -> None:
        assert ids.ensure_unique("skill-x", ["skill-y"]) == "skill-x"

    def test_collision_appends_2(self) -> None:
        assert ids.ensure_unique("skill-x", ["skill-x"]) == "skill-x-2"

    def test_suffix_chain(self) -> None:
        existing = {"skill-x", "skill-x-2", "skill-x-3"}
        assert ids.ensure_unique("skill-x", existing) == "skill-x-4"

    def test_empty_existing(self) -> None:
        assert ids.ensure_unique("a", []) == "a"
