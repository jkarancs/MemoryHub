"""Tests for the data models, profile-aware validation, and JSON-schema generation."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from memoryhub import (
    Frontmatter,
    frontmatter_json_schema,
    load_profile,
    validate_against_profile,
)


def _valid_fields(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "skill-async-python",
        "title": "Async Python",
        "type": "skill",
        "description": "asyncio, tasks, and structured concurrency",
        "tags": ["python", "async"],
        "status": "active",
        "visibility": "public",
        "created": date(2026, 1, 1),
        "updated": date(2026, 1, 2),
    }
    base.update(overrides)
    return base


def test_valid_frontmatter() -> None:
    fm = Frontmatter(**_valid_fields())
    assert fm.id == "skill-async-python"
    assert fm.source == "self"
    assert fm.extra == {}


def test_invalid_slug_id_rejected() -> None:
    with pytest.raises(ValidationError):
        Frontmatter(**_valid_fields(id="Not A Slug"))


def test_created_after_updated_rejected() -> None:
    with pytest.raises(ValidationError):
        Frontmatter(**_valid_fields(created=date(2026, 2, 1), updated=date(2026, 1, 1)))


def test_unknown_toplevel_key_forbidden() -> None:
    with pytest.raises(ValidationError):
        Frontmatter(**_valid_fields(bogus="x"))


def test_validate_against_profile_ok() -> None:
    profile = load_profile("personal")
    fm = Frontmatter(**_valid_fields(extra={"proficiency": "advanced"}))
    assert validate_against_profile(fm, profile) == []


def test_validate_against_profile_unknown_type() -> None:
    profile = load_profile("personal")
    fm = Frontmatter(**_valid_fields(type="unknown_type"))
    problems = validate_against_profile(fm, profile)
    assert problems and "not in profile" in problems[0]


def test_validate_against_profile_disallowed_extra() -> None:
    profile = load_profile("personal")
    fm = Frontmatter(**_valid_fields(extra={"org": "ACME"}))  # org is not a skill field
    problems = validate_against_profile(fm, profile)
    assert problems and "not allowed for type 'skill'" in problems[0]


def test_json_schema_reflects_profile() -> None:
    profile = load_profile("personal")
    schema = frontmatter_json_schema(profile)
    assert schema["$schema"].startswith("http://json-schema.org/draft-07")
    assert schema["properties"]["type"]["enum"] == profile.type_names
    assert schema["properties"]["status"]["enum"] == ["active", "archived", "draft", "aspirational"]
    assert schema["properties"]["visibility"]["enum"] == ["public", "private"]
    assert set(profile.common_required).issubset(set(schema["required"]))
    assert "extra" not in schema["properties"]
