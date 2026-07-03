"""Tests for schema-profile loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from memoryhub import list_builtin_profiles, load_profile


def test_load_builtin_personal() -> None:
    profile = load_profile("personal")
    assert profile.name == "personal"
    assert profile.type_names == [
        "bio",
        "skill",
        "experience",
        "education",
        "project",
        "goal",
        "preference",
        "writing",
    ]
    assert profile.fields_for("skill") == ["proficiency", "source"]
    assert profile.fields_for("bio") == []
    assert profile.is_known_type("goal")
    assert not profile.is_known_type("nope")
    assert profile.enums["visibility"] == ["public", "private"]


def test_personal_is_listed_as_builtin() -> None:
    assert "personal" in list_builtin_profiles()


def test_unknown_profile_name_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_profile("does-not-exist")


def test_load_profile_from_path(tmp_path: Path) -> None:
    custom = tmp_path / "work.yaml"
    custom.write_text(
        "name: work\n"
        "types:\n"
        "  meeting: {fields: [attendees]}\n"
        "common_required: [id, title, type]\n"
        "enums:\n"
        "  status: [active, archived, draft, aspirational]\n"
        "  visibility: [public, private]\n",
        encoding="utf-8",
    )
    profile = load_profile(custom)
    assert profile.name == "work"
    assert profile.fields_for("meeting") == ["attendees"]


def test_malformed_profile_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_profile(bad)
