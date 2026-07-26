"""Tests for the data models, profile-aware validation, and JSON-schema generation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from conftest import CUSTOM_PROFILE_YAML
from memoryhub import (
    Frontmatter,
    frontmatter_json_schema,
    load_profile,
    validate_against_profile,
)


def _write_workflow_profile(tmp_path: Path) -> Path:
    """A profile with its own types, status vocabulary, and an enum on a type-specific field."""
    path = tmp_path / "workflow.yaml"
    path.write_text(CUSTOM_PROFILE_YAML, encoding="utf-8")
    return path


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


def test_status_outside_profile_enum_is_a_problem() -> None:
    profile = load_profile("personal")
    fm = Frontmatter(**_valid_fields(status="planned"))  # a workflow status, not a personal one
    problems = validate_against_profile(fm, profile)
    assert problems and "status 'planned' is not in profile 'personal'" in problems[0]


def test_custom_profile_accepts_its_own_status_vocabulary(tmp_path: Path) -> None:
    profile = load_profile(_write_workflow_profile(tmp_path))
    fm = Frontmatter(
        **_valid_fields(
            type="subnode", status="planned", visibility="private", extra={"role": "plan"}
        )
    )
    assert validate_against_profile(fm, profile) == []


def test_custom_profile_rejects_bad_status_and_extra_field_enum(tmp_path: Path) -> None:
    profile = load_profile(_write_workflow_profile(tmp_path))
    fm = Frontmatter(
        **_valid_fields(
            type="subnode", status="draft", visibility="private", extra={"role": "boss"}
        )
    )
    problems = validate_against_profile(fm, profile)
    assert any("status 'draft' is not in profile 'workflow'" in p for p in problems)
    assert any("role 'boss' is not in profile 'workflow'" in p for p in problems)


def test_enum_on_a_field_the_doc_omits_is_not_checked(tmp_path: Path) -> None:
    profile = load_profile(_write_workflow_profile(tmp_path))
    fm = Frontmatter(**_valid_fields(type="node", status="done", visibility="private"))
    assert validate_against_profile(fm, profile) == []  # no `role` key, no `role` complaint


def test_json_schema_reflects_profile() -> None:
    profile = load_profile("personal")
    schema = frontmatter_json_schema(profile)
    assert schema["$schema"].startswith("http://json-schema.org/draft-07")
    assert schema["properties"]["type"]["enum"] == profile.type_names
    assert schema["properties"]["status"]["enum"] == ["active", "archived", "draft", "aspirational"]
    assert schema["properties"]["visibility"]["enum"] == ["public", "private"]
    assert set(profile.common_required).issubset(set(schema["required"]))
    assert "extra" not in schema["properties"]
