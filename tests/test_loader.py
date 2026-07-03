"""Loader tests: aggregated validation, serialization stability, and Windows-safe output."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import memory_text, write_memory
from memoryhub import LoadError, load_config, load_profile, loader

PROFILE = load_profile("personal")


# --- load_all / load_one ---------------------------------------------------------


def test_load_all_returns_every_doc_with_paths(seeded_repo: Path) -> None:
    docs = loader.load_all(load_config(seeded_repo))
    assert len(docs) == 11
    assert all(doc.path is not None and doc.path.is_file() for doc in docs)
    assert {doc.id for doc in docs} >= {"bio-me", "skill-async-python", "goal-ai-role"}


def test_load_one_parses_frontmatter_extras(seeded_repo: Path) -> None:
    path = seeded_repo / "memory" / "skill" / "skill-async-python.md"
    doc = loader.load_one(path, PROFILE)
    assert doc.frontmatter.extra["proficiency"] == "advanced"
    assert doc.frontmatter.related == ["project-hub"]
    assert "asyncio" in doc.body


def test_load_all_aggregates_all_errors(seeded_repo: Path) -> None:
    write_memory(seeded_repo, id="alien-thing", type="alien")  # type not in profile
    bad = seeded_repo / "memory" / "skill" / "broken.md"
    bad.write_text("---\nid: broken\ntype: skill\n---\nno title etc\n", encoding="utf-8")

    with pytest.raises(LoadError) as excinfo:
        loader.load_all(load_config(seeded_repo))
    issues = excinfo.value.issues
    paths = {str(issue.path) for issue in issues}
    assert any("alien-thing" in p for p in paths)
    assert any("broken" in p for p in paths)
    assert len(issues) >= 2  # both bad files reported in one raise


def test_duplicate_id_is_an_error(seeded_repo: Path) -> None:
    write_memory(seeded_repo, id="bio-me", type="preference", title="dupe")
    with pytest.raises(LoadError) as excinfo:
        loader.load_all(load_config(seeded_repo))
    assert any(
        issue.field == "id" and "duplicate" in issue.reason for issue in excinfo.value.issues
    )


def test_unknown_type_specific_field_is_an_error(seeded_repo: Path) -> None:
    write_memory(seeded_repo, id="bio-two", type="bio", extras={"proficiency": "expert"})
    with pytest.raises(LoadError) as excinfo:
        loader.load_all(load_config(seeded_repo))
    assert any("proficiency" in issue.reason for issue in excinfo.value.issues)


def test_missing_frontmatter_block(tmp_path: Path) -> None:
    path = tmp_path / "plain.md"
    path.write_text("just some markdown\n", encoding="utf-8")
    with pytest.raises(LoadError) as excinfo:
        loader.load_one(path, PROFILE)
    assert excinfo.value.issues[0].field == "frontmatter"


def test_trash_and_dot_dirs_are_skipped(seeded_repo: Path) -> None:
    trash = seeded_repo / "memory" / ".trash"
    trash.mkdir()
    (trash / "garbage.md").write_text("not even frontmatter", encoding="utf-8")
    docs = loader.load_all(load_config(seeded_repo))
    assert len(docs) == 11


def test_missing_content_root_is_reported(tmp_path: Path) -> None:
    (tmp_path / "hub.toml").write_text('[hub]\nname = "x"\n', encoding="utf-8")
    with pytest.raises(LoadError) as excinfo:
        loader.load_all(load_config(tmp_path))
    assert excinfo.value.issues[0].field == "content_root"


# --- serialize ---------------------------------------------------------------------


def test_serialize_from_crlf_round_trips_to_lf(tmp_path: Path) -> None:
    text = memory_text(
        id="skill-windows",
        type="skill",
        title="Windows dev",
        tags="[windows]",
        extras={"proficiency": "advanced"},
        body="Line one.\nLine two.",
    )
    path = tmp_path / "skill-windows.md"
    path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))

    doc = loader.load_one(path, PROFILE)
    serialized = loader.serialize(doc, PROFILE)
    assert "\r" not in serialized
    assert serialized.endswith("\n")

    # Round trip: re-loading the serialized text yields the same doc and the same text.
    path2 = tmp_path / "again.md"
    path2.write_text(serialized, encoding="utf-8", newline="\n")
    doc2 = loader.load_one(path2, PROFILE)
    assert doc2.frontmatter.model_dump() == doc.frontmatter.model_dump()
    assert doc2.body == doc.body
    assert loader.serialize(doc2, PROFILE) == serialized


def test_serialize_key_ordering_is_stable(seeded_repo: Path) -> None:
    doc = loader.load_one(seeded_repo / "memory" / "skill" / "skill-async-python.md", PROFILE)
    serialized = loader.serialize(doc, PROFILE)
    frontmatter_block = serialized.split("---")[1]
    keys = [line.split(":")[0] for line in frontmatter_block.strip().splitlines() if ":" in line]
    assert keys == [
        "id",
        "title",
        "type",
        "description",
        "tags",
        "status",
        "visibility",
        "created",
        "updated",
        "proficiency",
        "related",
        "source",
    ]


def test_serialize_shuffled_input_normalizes_ordering(tmp_path: Path) -> None:
    # Keys deliberately out of order in the source file.
    path = tmp_path / "shuffled.md"
    path.write_text(
        "---\n"
        "source: self\n"
        "updated: 2026-01-02\n"
        "title: Shuffled\n"
        "proficiency: expert\n"
        "type: skill\n"
        "created: 2026-01-01\n"
        "visibility: private\n"
        "tags: [x]\n"
        "status: active\n"
        "description: out of order\n"
        "id: skill-shuffled\n"
        "---\n\nBody.\n",
        encoding="utf-8",
    )
    serialized = loader.serialize(loader.load_one(path, PROFILE), PROFILE)
    assert serialized.index("id:") < serialized.index("title:") < serialized.index("proficiency:")
    assert serialized.rindex("source:") > serialized.rindex("related:")


def test_serialize_empty_body(tmp_path: Path) -> None:
    text = memory_text(id="bio-x", type="bio", body="")
    path = tmp_path / "bio-x.md"
    path.write_text(text, encoding="utf-8")
    serialized = loader.serialize(loader.load_one(path, PROFILE), PROFILE)
    assert serialized.endswith("---\n")


# --- validate_store ----------------------------------------------------------------


def test_validate_store_clean(seeded_repo: Path) -> None:
    report = loader.validate_store(load_config(seeded_repo))
    assert report.ok
    assert report.checked == 11
    assert report.issues == []
    assert report.warnings == []


def test_validate_store_collects_issues_without_raising(seeded_repo: Path) -> None:
    write_memory(seeded_repo, id="alien-thing", type="alien")
    report = loader.validate_store(load_config(seeded_repo))
    assert not report.ok
    assert report.checked == 12
    assert any("alien" in issue.reason for issue in report.issues)
    payload = report.to_dict()
    assert payload["valid"] is False
    assert payload["issues"][0]["path"]


def test_validate_store_unresolved_related_is_a_warning(seeded_repo: Path) -> None:
    write_memory(seeded_repo, id="goal-later", type="goal", related="[skill-not-written-yet]")
    report = loader.validate_store(load_config(seeded_repo))
    assert report.ok  # forward refs warn, not fail
    assert len(report.warnings) == 1
    assert "skill-not-written-yet" in report.warnings[0].reason
