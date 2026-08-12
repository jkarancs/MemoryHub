"""Schema profiles: the closed ``type`` vocabulary and per-type fields.

A *profile* is the generalization seam for MemoryHub. The engine hardcodes nothing about
"personal"; a content repo selects a profile (a built-in name such as ``personal`` or a path to a
custom ``.yaml``) and the models validate against it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .ids import parent_id

# Directory holding the built-in profile YAML files (shipped inside the package).
_BUILTIN_DIR = Path(__file__).parent / "profiles"

#: A derived subfolder must be one plain path segment — no separators, no traversal, no dotfile.
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
#: A ``YYYYMMDD`` run of digits anywhere in the value (``orch-20260812-2`` -> 2026, 08).
_YYYYMMDD_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")


class TypeSpec(BaseModel):
    """Per-type configuration: which extra (type-specific) fields the type allows."""

    model_config = ConfigDict(extra="forbid")

    fields: list[str] = Field(default_factory=list)


class LayoutRule(BaseModel):
    """How one type's canonical subfolder is derived from a document's own frontmatter.

    Derivation is *pure*: one named field, one named transform, no store lookups — so a write
    stays a single validated atomic write and the same rule can be replayed by ``relayout``.

    Transforms:
      * ``none`` — the field's value is the subfolder (``node`` -> its ``supernode``);
      * ``parent-id`` — the value is an id whose trailing sequence segments are dropped
        (a subnode's ``node`` ``foo-03``/``foo-03-r1`` -> the supernode ``foo``);
      * ``year-month`` — the ``YYYYMMDD`` inside the value becomes ``YYYY-MM``
        (``orch-20260812-2`` -> ``2026-08``).
    """

    model_config = ConfigDict(extra="forbid")

    field: str
    transform: Literal["none", "parent-id", "year-month"] = "none"

    def subfolder(self, fields: Mapping[str, Any]) -> str | None:
        """The subfolder for a document with these flat frontmatter fields, or ``None``.

        ``None`` — the type's directory itself — is the answer whenever the rule cannot produce
        one safe path segment: a missing/non-string field, a transform that matched nothing, or
        a value that is not a plain segment. A document is never placed outside its type folder.
        """
        value = fields.get(self.field)
        if not isinstance(value, str) or not value:
            return None
        if self.transform == "parent-id":
            value = parent_id(value)
        elif self.transform == "year-month":
            match = _YYYYMMDD_RE.search(value)
            value = f"{match.group(1)}-{match.group(2)}" if match else ""
        return value if _SEGMENT_RE.match(value) else None


class Profile(BaseModel):
    """A parsed, validated schema profile."""

    model_config = ConfigDict(extra="forbid")

    name: str
    types: dict[str, TypeSpec]
    common_required: list[str] = Field(default_factory=list)
    #: Closed vocabularies by field name (``status``, ``visibility``, or any type-specific field).
    enums: dict[str, list[str]] = Field(default_factory=dict)
    #: Field values ``hub add``/``hub new`` fall back to when a write omits them (see writer.add).
    defaults: dict[str, str] = Field(default_factory=dict)
    #: Per-type subfolder rules (see :class:`LayoutRule`). Types absent from the mapping — and
    #: every type of a profile that declares no ``layout`` — stay flat in ``<content_root>/<type>``.
    layout: dict[str, LayoutRule] = Field(default_factory=dict)

    # --- convenience accessors -------------------------------------------------

    @property
    def type_names(self) -> list[str]:
        """The closed ``type`` vocabulary, in declaration order."""
        return list(self.types.keys())

    def fields_for(self, type_name: str) -> list[str]:
        """Extra (type-specific) fields allowed for ``type_name`` (empty if none/unknown)."""
        spec = self.types.get(type_name)
        return list(spec.fields) if spec else []

    def is_known_type(self, type_name: str) -> bool:
        return type_name in self.types

    def subfolder_for(self, type_name: str, fields: Mapping[str, Any]) -> str | None:
        """The canonical subfolder under ``<content_root>/<type_name>``, or ``None`` for flat."""
        rule = self.layout.get(type_name)
        return rule.subfolder(fields) if rule is not None else None


def _builtin_path(name: str) -> Path | None:
    candidate = _BUILTIN_DIR / f"{name}.yaml"
    return candidate if candidate.is_file() else None


def load_profile(name_or_path: str | Path) -> Profile:
    """Load a profile by built-in name (e.g. ``"personal"``) or by path to a ``.yaml`` file.

    Resolution order:
      1. If ``name_or_path`` points at an existing file, load that file.
      2. Otherwise treat it as a built-in profile name and look in the packaged ``profiles/`` dir.

    Raises:
        FileNotFoundError: if neither a file nor a built-in profile matches.
        ValueError: if the YAML is malformed or fails validation.
    """
    path = Path(name_or_path)
    if path.is_file():
        source = path
    else:
        builtin = _builtin_path(str(name_or_path))
        if builtin is None:
            available = ", ".join(sorted(p.stem for p in _BUILTIN_DIR.glob("*.yaml")))
            raise FileNotFoundError(
                f"No profile found for {name_or_path!r}. "
                f"Provide a path to a .yaml file or a built-in name ({available})."
            )
        source = builtin

    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Profile {source} is not valid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Profile {source} must be a mapping at the top level.")

    return Profile.model_validate(data)


def list_builtin_profiles() -> list[str]:
    """Names of profiles shipped inside the package."""
    return sorted(p.stem for p in _BUILTIN_DIR.glob("*.yaml"))
