"""Pydantic data models for MemoryHub, plus the JSON-Schema builder.

``Frontmatter`` models the YAML frontmatter block of a memory file; ``MemoryDoc`` couples it with
the markdown body and source path. Type-specific ("extra") fields are held in ``extra`` — the
loader (Phase 1) routes flat frontmatter keys into it based on the active profile.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .profiles import Profile

#: IDs are slugs: lowercase alphanumerics and hyphens, not starting with a hyphen.
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

Status = Literal["active", "archived", "draft", "aspirational"]
Visibility = Literal["public", "private"]


class Frontmatter(BaseModel):
    """Validated frontmatter for a single memory file.

    Common fields are declared explicitly; type-specific fields (per the active profile) live in
    ``extra``. Profile-aware checks (is ``type`` in the vocab? are the ``extra`` keys permitted?)
    are applied via :func:`validate_against_profile`, since the base model has no profile context.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    type: str
    description: str
    tags: list[str] = Field(default_factory=list)
    status: Status
    visibility: Visibility
    created: date
    updated: date
    related: list[str] = Field(default_factory=list)
    source: str = "self"
    # Type-specific extras, validated dynamically against the active profile.
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id_is_slug(cls, value: str) -> str:
        if not SLUG_RE.match(value):
            raise ValueError(f"id {value!r} is not a valid slug (must match {SLUG_RE.pattern})")
        return value

    @model_validator(mode="after")
    def _created_before_updated(self) -> Frontmatter:
        if self.created > self.updated:
            raise ValueError(
                f"created ({self.created}) must be on or before updated ({self.updated})"
            )
        return self


class MemoryDoc(BaseModel):
    """A parsed memory file: validated frontmatter + markdown body + source path."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frontmatter: Frontmatter
    body: str  # markdown after the frontmatter
    path: Path | None = None  # source file (populated on load)

    @property
    def id(self) -> str:
        return self.frontmatter.id

    @property
    def type(self) -> str:
        return self.frontmatter.type


def validate_against_profile(fm: Frontmatter, profile: Profile) -> list[str]:
    """Return a list of human-readable problems for ``fm`` under ``profile`` (empty == valid).

    Checks that are profile-dependent and therefore cannot live on the model itself:
      * ``type`` is part of the profile's closed vocabulary.
      * every key in ``extra`` is an allowed type-specific field for that type.
    """
    problems: list[str] = []
    if not profile.is_known_type(fm.type):
        problems.append(
            f"type {fm.type!r} is not in profile {profile.name!r} "
            f"(allowed: {', '.join(profile.type_names)})"
        )
        return problems  # can't check extras without a known type

    allowed = set(profile.fields_for(fm.type))
    for key in fm.extra:
        if key not in allowed:
            allowed_str = ", ".join(sorted(allowed)) or "(none)"
            problems.append(
                f"field {key!r} is not allowed for type {fm.type!r} "
                f"(allowed extras: {allowed_str})"
            )
    return problems


def frontmatter_json_schema(profile: Profile) -> dict[str, Any]:
    """Build a JSON Schema (draft-07) for memory-file frontmatter under ``profile``.

    Generated from :class:`Frontmatter` and specialized with the profile: ``type`` is restricted
    to the profile vocabulary, ``status``/``visibility`` use the profile enums, and type-specific
    fields are described. The schema targets flat frontmatter as written in files (so the internal
    ``extra`` bag is dropped and additional properties are allowed for type-specific fields).
    """
    schema: dict[str, Any] = Frontmatter.model_json_schema()
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    schema["title"] = f"MemoryHub frontmatter ({profile.name} profile)"
    schema["description"] = (
        "Frontmatter for a MemoryHub memory file. Generated from the models + the "
        f"{profile.name!r} schema profile via `hub schema export`."
    )

    props: dict[str, Any] = schema.setdefault("properties", {})

    props["type"] = {
        "type": "string",
        "enum": profile.type_names,
        "description": "Memory type (closed vocabulary defined by the profile).",
    }
    if "status" in profile.enums:
        props.setdefault("status", {})
        props["status"] = {"type": "string", "enum": list(profile.enums["status"])}
    if "visibility" in profile.enums:
        props.setdefault("visibility", {})
        props["visibility"] = {"type": "string", "enum": list(profile.enums["visibility"])}

    # In files, type-specific fields are flat keys; the internal `extra` bag is engine-only.
    props.pop("extra", None)

    # Document type-specific fields for editor hover/help.
    type_fields = {t: profile.fields_for(t) for t in profile.type_names}
    schema["x-memoryhub-type-fields"] = type_fields

    schema["required"] = list(profile.common_required)
    # Allow type-specific extras (and forward-compatible keys) in files.
    schema["additionalProperties"] = True
    return schema
