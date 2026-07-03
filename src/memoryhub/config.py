"""Load and validate a content repo's ``hub.toml`` into a :class:`Config`.

The engine hardcodes nothing about a particular deployment; every content repo carries a
``hub.toml`` that selects its content root, schema profile, write policy, and (for later phases)
embedding/index backends. :func:`load_config` walks up from a starting directory to find the
nearest ``hub.toml``, resolves relative paths against it, and validates strictly (unknown keys
are errors).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

CONFIG_FILENAME = "hub.toml"


class ConfigError(ValueError):
    """Raised when a ``hub.toml`` is missing, malformed, or fails validation."""


class HubSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    content_root: str = "./memory"
    profile: str = "personal"  # built-in profile name OR path to a custom .yaml


class WriteSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow_agent_writes: bool = True
    require_confirmation: bool = False


class LocalEmbedSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "BAAI/bge-m3"
    device: str = "cuda"


class ApiEmbedSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "openai/text-embedding-3-large"
    api_key_env: str = "OPENROUTER_API_KEY"


class EmbeddingsSection(BaseModel):
    """Declared now (Phase 0); consumed in Phase 3."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["local", "api"] = "local"
    local: LocalEmbedSection = Field(default_factory=LocalEmbedSection)
    api: ApiEmbedSection = Field(default_factory=ApiEmbedSection)


class IndexSection(BaseModel):
    """Declared now (Phase 0); consumed in Phase 3."""

    model_config = ConfigDict(extra="forbid")

    backend: str = "lancedb"
    path: str = ".index"


class Config(BaseModel):
    """Validated ``hub.toml``. Path-typed accessors resolve against the file's directory."""

    model_config = ConfigDict(extra="forbid")

    hub: HubSection
    write: WriteSection = Field(default_factory=WriteSection)
    embeddings: EmbeddingsSection = Field(default_factory=EmbeddingsSection)
    index: IndexSection = Field(default_factory=IndexSection)

    # Populated by load_config; not part of hub.toml.
    root: Path | None = Field(default=None, exclude=True, repr=False)
    source_path: Path | None = Field(default=None, exclude=True, repr=False)

    @property
    def _base(self) -> Path:
        return self.root if self.root is not None else Path.cwd()

    @property
    def content_root(self) -> Path:
        """Absolute path to the markdown store, resolved against ``hub.toml``'s directory."""
        return (self._base / self.hub.content_root).resolve()

    @property
    def index_path(self) -> Path:
        """Absolute path to the (Phase 3) vector index, resolved against ``hub.toml``."""
        return (self._base / self.index.path).resolve()

    def resolve_api_key(self) -> str | None:
        """Read the embeddings API key from its env var (secrets come from env, never from toml)."""
        return os.environ.get(self.embeddings.api.api_key_env)


def _find_config_file(start: Path) -> Path | None:
    """Walk up from ``start`` (a file or directory) to find the nearest ``hub.toml``."""
    current = start if start.is_dir() else start.parent
    current = current.resolve()
    for directory in (current, *current.parents):
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_config(start_dir: str | Path = ".") -> Config:
    """Find, read, and validate the nearest ``hub.toml`` at or above ``start_dir``.

    Args:
        start_dir: A directory (or file) to begin the upward search from.

    Returns:
        A validated :class:`Config` with ``root``/``source_path`` populated.

    Raises:
        ConfigError: if no ``hub.toml`` is found, if it is not valid TOML, or if validation fails
            (including unknown keys).
    """
    start = Path(start_dir).resolve()
    toml_path = _find_config_file(start)
    if toml_path is None:
        raise ConfigError(f"No {CONFIG_FILENAME} found at or above {start}")

    try:
        with toml_path.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{toml_path} is not valid TOML: {exc}") from exc

    try:
        config = Config.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {toml_path}:\n{exc}") from exc

    config.root = toml_path.parent
    config.source_path = toml_path
    return config
