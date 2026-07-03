"""Lock in the public import surface (Phases 0–4 are all implemented; nothing stubs anymore)."""

from __future__ import annotations

from pathlib import Path

from memoryhub import Hub, load_config


def test_public_import_surface() -> None:
    # Everything a consumer may import from the package root, across Phases 0–4. The vector
    # names must import cleanly even without the vectors/embedding extras installed.
    from memoryhub import Embedder as _Embedder  # noqa: F401
    from memoryhub import EmbeddingError as _EmbeddingError  # noqa: F401
    from memoryhub import Hub as _Hub  # noqa: F401
    from memoryhub import IndexWarning as _IndexWarning  # noqa: F401
    from memoryhub import LoadError as _LoadError  # noqa: F401
    from memoryhub import ReindexStats as _ReindexStats  # noqa: F401
    from memoryhub import VectorIndex as _VectorIndex  # noqa: F401
    from memoryhub import WriteError as _WriteError  # noqa: F401
    from memoryhub import WriteWarning as _WriteWarning  # noqa: F401
    from memoryhub import get_embedder as _get_embedder  # noqa: F401
    from memoryhub import load_config as _load_config  # noqa: F401


def test_hub_constructs_and_loads_profile(content_repo: Path) -> None:
    hub = Hub(load_config(content_repo))
    assert hub.profile.name == "personal"
    assert hub.all() == []  # empty store reads fine


def test_hub_accepts_path(content_repo: Path) -> None:
    hub = Hub(content_repo)
    assert hub.config.hub.name == "test"
