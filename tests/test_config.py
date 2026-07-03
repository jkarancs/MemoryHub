"""Tests for config loading/validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from memoryhub import ConfigError, load_config


def test_load_config_basic(content_repo: Path) -> None:
    config = load_config(content_repo)
    assert config.hub.name == "test"
    assert config.hub.profile == "personal"
    assert config.write.allow_agent_writes is True
    assert config.embeddings.backend == "local"
    assert config.embeddings.local.model == "BAAI/bge-m3"
    assert config.index.backend == "lancedb"


def test_paths_resolved_against_toml(content_repo: Path) -> None:
    config = load_config(content_repo)
    assert config.content_root == (content_repo / "memory").resolve()
    assert config.index_path == (content_repo / ".index").resolve()
    assert config.source_path == (content_repo / "hub.toml").resolve()


def test_walks_up_to_find_config(content_repo: Path) -> None:
    nested = content_repo / "memory"  # a subdirectory
    config = load_config(nested)
    assert config.source_path == (content_repo / "hub.toml").resolve()


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    (tmp_path / "hub.toml").write_text('[hub]\nname = "x"\nbogus_key = 1\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(tmp_path)


def test_invalid_toml_rejected(tmp_path: Path) -> None:
    (tmp_path / "hub.toml").write_text("this is = = not toml", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(tmp_path)


def test_resolve_api_key_from_env(content_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(content_repo)
    assert config.resolve_api_key() is None
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    assert config.resolve_api_key() == "secret"
