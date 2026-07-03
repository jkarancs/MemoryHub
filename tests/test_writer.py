"""Writer tests: atomicity, guards, defaults, referential warnings, soft delete."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest

from conftest import HUB_TOML
from memoryhub import WriteError, WriteWarning, load_config, load_profile, loader, writer

PROFILE = load_profile("personal")


def _tree(root: Path) -> set[Path]:
    return {p for p in root.rglob("*") if p.is_file()}


# --- add --------------------------------------------------------------------------


def test_add_creates_validated_file_with_safe_defaults(content_repo: Path) -> None:
    config = load_config(content_repo)
    doc = writer.add(config, {"type": "skill", "title": "Rust"}, "Learning it.")
    assert doc.id == "skill-rust"
    path = content_repo / "memory" / "skill" / "skill-rust.md"
    assert doc.path == path.resolve() or doc.path == path
    assert path.is_file()

    loaded = loader.load_one(path, PROFILE)
    assert loaded.frontmatter.status == "draft"
    assert loaded.frontmatter.visibility == "private"
    assert loaded.frontmatter.created == loaded.frontmatter.updated == date.today()
    assert loaded.body == "Learning it.\n"


def test_add_writes_lf_only_utf8(content_repo: Path) -> None:
    config = load_config(content_repo)
    writer.add(
        config,
        {"type": "skill", "title": "Windows Lines", "description": "árvíztűrő tükörfúrógép"},
        "CRLF\r\nin the\r\nbody.",
    )
    raw = (content_repo / "memory" / "skill" / "skill-windows-lines.md").read_bytes()
    assert b"\r" not in raw
    assert "árvíztűrő".encode() in raw  # UTF-8, not a platform codepage


def test_add_generated_id_gets_unique_suffix(seeded_repo: Path) -> None:
    config = load_config(seeded_repo)
    doc = writer.add(config, {"type": "skill", "title": "SQL"}, "another take")
    assert doc.id == "skill-sql-2"


def test_add_explicit_duplicate_id_hard_fails(seeded_repo: Path) -> None:
    config = load_config(seeded_repo)
    before = _tree(seeded_repo / "memory")
    with pytest.raises(WriteError, match="duplicate id"):
        writer.add(config, {"type": "skill", "title": "SQL", "id": "skill-sql"}, "x")
    assert _tree(seeded_repo / "memory") == before


def test_add_invalid_never_touches_disk(content_repo: Path) -> None:
    config = load_config(content_repo)
    before = _tree(content_repo)
    with pytest.raises(WriteError, match="type"):
        writer.add(config, {"type": "nonsense", "title": "X"}, "")
    with pytest.raises(WriteError, match="title"):
        writer.add(config, {"type": "skill", "title": "  "}, "")
    with pytest.raises(WriteError, match="proficiency"):
        writer.add(config, {"type": "bio", "title": "Me", "proficiency": "expert"}, "")
    with pytest.raises(WriteError, match="status"):
        writer.add(config, {"type": "skill", "title": "X", "status": "bogus"}, "")
    assert _tree(content_repo) == before


def test_add_unresolved_related_warns_but_writes(seeded_repo: Path) -> None:
    config = load_config(seeded_repo)
    with pytest.warns(WriteWarning, match="skill-ghost"):
        doc = writer.add(
            config, {"type": "goal", "title": "Ghost goal", "related": ["skill-ghost"]}, ""
        )
    assert doc.path is not None and doc.path.is_file()


def test_add_resolved_related_does_not_warn(seeded_repo: Path) -> None:
    import warnings as warnings_module

    config = load_config(seeded_repo)
    with warnings_module.catch_warnings():
        warnings_module.simplefilter("error", WriteWarning)
        writer.add(config, {"type": "goal", "title": "Fine", "related": ["skill-sql"]}, "")


def test_add_atomic_no_partial_files_on_failure(content_repo: Path, monkeypatch) -> None:
    config = load_config(content_repo)

    def boom(src: str, dst: str) -> None:
        raise OSError("simulated crash at replace")

    monkeypatch.setattr("memoryhub.writer.os.replace", boom)
    with pytest.raises(OSError, match="simulated"):
        writer.add(config, {"type": "skill", "title": "Crashy"}, "body")
    skill_dir = content_repo / "memory" / "skill"
    assert not (skill_dir / "skill-crashy.md").exists()
    assert list(skill_dir.glob("*.tmp")) == []  # temp file cleaned up


def test_add_refused_when_agent_writes_disabled(tmp_path: Path) -> None:
    toml = HUB_TOML.replace("allow_agent_writes = true", "allow_agent_writes = false")
    (tmp_path / "hub.toml").write_text(toml, encoding="utf-8")
    (tmp_path / "memory").mkdir()
    config = load_config(tmp_path)
    with pytest.raises(WriteError, match="allow_agent_writes"):
        writer.add(config, {"type": "skill", "title": "X"}, "")
    with pytest.raises(WriteError, match="allow_agent_writes"):
        writer.update(config, "whatever")
    with pytest.raises(WriteError, match="allow_agent_writes"):
        writer.delete(config, "whatever")


def test_require_confirmation_hook(tmp_path: Path) -> None:
    toml = HUB_TOML.replace("require_confirmation = false", "require_confirmation = true")
    (tmp_path / "hub.toml").write_text(toml, encoding="utf-8")
    (tmp_path / "memory").mkdir()
    config = load_config(tmp_path)

    # No confirmer registered -> refuse.
    with pytest.raises(WriteError, match="confirm"):
        writer.add(config, {"type": "skill", "title": "X"}, "")

    # Confirmer says no -> refuse.
    writer.confirm_callback = lambda message: False
    with pytest.raises(WriteError, match="not confirmed"):
        writer.add(config, {"type": "skill", "title": "X"}, "")

    # Confirmer says yes -> write proceeds.
    writer.confirm_callback = lambda message: True
    doc = writer.add(config, {"type": "skill", "title": "X"}, "")
    assert doc.path is not None and doc.path.is_file()


def test_path_confinement_rejected(content_repo: Path) -> None:
    config = load_config(content_repo)
    with pytest.raises(WriteError, match="content_root"):
        writer._confine(config, config.content_root / ".." / "escape.md")


# --- update -----------------------------------------------------------------------


def test_update_patches_and_bumps_updated(seeded_repo: Path) -> None:
    config = load_config(seeded_repo)
    doc = writer.update(
        config,
        "skill-sql",
        fields={"description": "window functions and query plans", "proficiency": "advanced"},
    )
    assert doc.frontmatter.updated == date.today()
    reloaded = loader.load_one(seeded_repo / "memory" / "skill" / "skill-sql.md", PROFILE)
    assert reloaded.frontmatter.description == "window functions and query plans"
    assert reloaded.frontmatter.extra["proficiency"] == "advanced"
    assert reloaded.frontmatter.created == date(2026, 1, 1)  # untouched


def test_update_body_only(seeded_repo: Path) -> None:
    config = load_config(seeded_repo)
    writer.update(config, "skill-sql", body="New body.\r\nWith CRLF input.")
    raw = (seeded_repo / "memory" / "skill" / "skill-sql.md").read_bytes()
    assert b"\r" not in raw
    assert b"New body." in raw


def test_update_none_removes_extra_field(seeded_repo: Path) -> None:
    config = load_config(seeded_repo)
    doc = writer.update(config, "skill-sql", fields={"proficiency": None})
    assert "proficiency" not in doc.frontmatter.extra


def test_update_invalid_leaves_file_unchanged(seeded_repo: Path) -> None:
    path = seeded_repo / "memory" / "skill" / "skill-sql.md"
    before = path.read_bytes()
    config = load_config(seeded_repo)
    with pytest.raises(WriteError, match="status"):
        writer.update(config, "skill-sql", fields={"status": "bogus"})
    assert path.read_bytes() == before


def test_update_refuses_id_and_type_changes(seeded_repo: Path) -> None:
    config = load_config(seeded_repo)
    with pytest.raises(WriteError, match="'id'"):
        writer.update(config, "skill-sql", fields={"id": "skill-renamed"})
    with pytest.raises(WriteError, match="'type'"):
        writer.update(config, "skill-sql", fields={"type": "project"})


def test_update_missing_id_fails(seeded_repo: Path) -> None:
    config = load_config(seeded_repo)
    with pytest.raises(WriteError, match="no memory"):
        writer.update(config, "does-not-exist", fields={"description": "x"})


# --- delete -----------------------------------------------------------------------


def test_delete_moves_to_trash(seeded_repo: Path) -> None:
    config = load_config(seeded_repo)
    writer.delete(config, "skill-docker")
    assert not (seeded_repo / "memory" / "skill" / "skill-docker.md").exists()
    assert (seeded_repo / "memory" / ".trash" / "skill-docker.md").is_file()
    # The store no longer sees it (and stays valid).
    docs = loader.load_all(config)
    assert "skill-docker" not in {d.id for d in docs}


def test_delete_twice_same_name_keeps_both_in_trash(content_repo: Path) -> None:
    config = load_config(content_repo)
    writer.add(config, {"type": "skill", "title": "Zig"}, "v1")
    writer.delete(config, "skill-zig")
    writer.add(config, {"type": "skill", "title": "Zig"}, "v2")
    writer.delete(config, "skill-zig")
    trash_files = list((content_repo / "memory" / ".trash").glob("skill-zig*.md"))
    assert len(trash_files) == 2


def test_delete_missing_id_fails(content_repo: Path) -> None:
    config = load_config(content_repo)
    with pytest.raises(WriteError, match="no memory"):
        writer.delete(config, "nope")


# --- scan_ids ----------------------------------------------------------------------


def test_scan_ids_tolerates_broken_files(seeded_repo: Path) -> None:
    # A file that would fail full validation must not block unrelated writes.
    broken = seeded_repo / "memory" / "skill" / "broken.md"
    broken.write_text("---\nid: broken\n---\nmissing everything\n", encoding="utf-8")
    config = load_config(seeded_repo)
    found = writer.scan_ids(config)
    assert "broken" in found
    doc = writer.add(config, {"type": "skill", "title": "Still Works"}, "")
    assert doc.path is not None and doc.path.is_file()


def test_stem_fallback_for_unparseable_frontmatter(content_repo: Path) -> None:
    path = content_repo / "memory" / "skill" / "weird.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\n: [unparseable\n---\n", encoding="utf-8")
    config = load_config(content_repo)
    assert "weird" in writer.scan_ids(config)


def test_atomic_write_uses_replace_semantics(content_repo: Path) -> None:
    # os.replace must overwrite an existing file in one step (Windows semantics matter here).
    config = load_config(content_repo)
    writer.add(config, {"type": "skill", "title": "Go"}, "first")
    doc = writer.update(config, "skill-go", body="second")
    assert doc.path is not None
    assert "second" in doc.path.read_text(encoding="utf-8")
    assert os.path.basename(doc.path) == "skill-go.md"
