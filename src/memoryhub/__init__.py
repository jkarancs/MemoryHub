"""MemoryHub — a reusable, content-agnostic engine over a markdown memory store.

Public API surface. Import from here rather than from submodules.
"""

from __future__ import annotations

from .bundle import Bundle, BundleItem, Excluded
from .config import Config, ConfigError, load_config
from .embeddings import Embedder, EmbeddingError, get_embedder
from .export import ExportError, ExportReport, ScanHit
from .hub import Hub
from .index import IndexWarning, ReindexStats, VectorIndex
from .loader import LoadError, StoreReport, ValidationIssue
from .models import (
    Frontmatter,
    MemoryDoc,
    frontmatter_json_schema,
    validate_against_profile,
)
from .profiles import Profile, list_builtin_profiles, load_profile
from .writer import WriteError, WriteWarning

__version__ = "0.1.0"

__all__ = [
    "Hub",
    "Bundle",
    "BundleItem",
    "Excluded",
    "Config",
    "ConfigError",
    "load_config",
    "Embedder",
    "EmbeddingError",
    "get_embedder",
    "ExportError",
    "ExportReport",
    "ScanHit",
    "VectorIndex",
    "IndexWarning",
    "ReindexStats",
    "Profile",
    "load_profile",
    "list_builtin_profiles",
    "Frontmatter",
    "MemoryDoc",
    "frontmatter_json_schema",
    "validate_against_profile",
    "LoadError",
    "StoreReport",
    "ValidationIssue",
    "WriteError",
    "WriteWarning",
    "__version__",
]
