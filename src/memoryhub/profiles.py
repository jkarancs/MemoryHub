"""Schema profiles: the closed ``type`` vocabulary and per-type fields.

A *profile* is the generalization seam for MemoryHub. The engine hardcodes nothing about
"personal"; a content repo selects a profile (a built-in name such as ``personal`` or a path to a
custom ``.yaml``) and the models validate against it.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

# Directory holding the built-in profile YAML files (shipped inside the package).
_BUILTIN_DIR = Path(__file__).parent / "profiles"


class TypeSpec(BaseModel):
    """Per-type configuration: which extra (type-specific) fields the type allows."""

    model_config = ConfigDict(extra="forbid")

    fields: list[str] = Field(default_factory=list)


class Profile(BaseModel):
    """A parsed, validated schema profile."""

    model_config = ConfigDict(extra="forbid")

    name: str
    types: dict[str, TypeSpec]
    common_required: list[str] = Field(default_factory=list)
    enums: dict[str, list[str]] = Field(default_factory=dict)

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
