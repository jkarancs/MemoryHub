"""CLI tests: every command end-to-end against a seeded store via Typer's CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import TopicEmbedder, block_module, write_memory
from memoryhub.cli import app

runner = CliRunner()


@pytest.fixture
def in_repo(seeded_repo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(seeded_repo)
    return seeded_repo


# --- schema (Phase 0, still green) -------------------------------------------------


def test_schema_export_stdout() -> None:
    result = runner.invoke(app, ["schema", "export", "--stdout", "--profile", "personal"])
    assert result.exit_code == 0, result.output
    schema = json.loads(result.output)
    assert schema["title"].startswith("MemoryHub frontmatter")
    assert "type" in schema["properties"]


def test_schema_export_to_file(tmp_path: Path) -> None:
    out = tmp_path / "schema" / "frontmatter.schema.json"
    result = runner.invoke(app, ["schema", "export", "--profile", "personal", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.is_file()
    schema = json.loads(out.read_text(encoding="utf-8"))
    assert schema["properties"]["type"]["enum"]


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    commands = ("list", "get", "find", "search", "reindex", "export", "new", "add", "update", "rm")
    for cmd in (*commands, "validate", "schema"):
        assert cmd in result.output


def test_no_config_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 2


# --- list / get / find ---------------------------------------------------------------


def test_list_all(in_repo: Path) -> None:
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0, result.output
    assert "skill-async-python" in result.output
    assert "goal-ai-role" in result.output


def test_list_filters(in_repo: Path) -> None:
    result = runner.invoke(app, ["list", "--type", "skill"])
    assert result.exit_code == 0
    assert "skill-sql" in result.output
    assert "project-hub" not in result.output

    result = runner.invoke(app, ["list", "--status", "draft"])
    assert "goal-ai-role" in result.output
    assert "skill-sql" not in result.output

    result = runner.invoke(app, ["list", "--tags", "python,async"])
    assert "skill-async-python" in result.output
    assert "project-hub" not in result.output


def test_list_json_is_machine_readable(in_repo: Path) -> None:
    result = runner.invoke(app, ["list", "--type", "skill", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {entry["id"] for entry in payload} == {"skill-async-python", "skill-sql", "skill-docker"}
    entry = next(e for e in payload if e["id"] == "skill-async-python")
    assert entry["proficiency"] == "advanced"
    assert entry["path"].endswith("skill-async-python.md")
    assert "body" not in entry  # summaries only


def test_get_raw(in_repo: Path) -> None:
    result = runner.invoke(app, ["get", "skill-async-python"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("---\nid: skill-async-python\n")
    assert "asyncio" in result.output


def test_get_json(in_repo: Path) -> None:
    result = runner.invoke(app, ["get", "skill-async-python", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["id"] == "skill-async-python"
    assert payload["created"] == "2026-01-01"
    assert "asyncio" in payload["body"]


def test_get_missing_exits_1(in_repo: Path) -> None:
    result = runner.invoke(app, ["get", "nope"])
    assert result.exit_code == 1
    assert "nope" in result.stderr


def test_find_ranked(in_repo: Path) -> None:
    result = runner.invoke(app, ["find", "asyncio"])
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert lines[0].startswith("skill-async-python")
    assert any(ln.startswith("project-hub") for ln in lines)


def test_find_json_and_field_restriction(in_repo: Path) -> None:
    result = runner.invoke(app, ["find", "asyncio", "--field", "title", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == []


def test_find_invalid_regex_exits_2(in_repo: Path) -> None:
    result = runner.invoke(app, ["find", "[unclosed", "--regex"])
    assert result.exit_code == 2
    assert "regex" in result.stderr


# --- add / update / rm / new ---------------------------------------------------------


def test_add_and_get_round_trip(in_repo: Path) -> None:
    result = runner.invoke(
        app,
        [
            "add",
            "--type",
            "skill",
            "--title",
            "Kubernetes",
            "--description",
            "container orchestration",
            "--tags",
            "devops,containers",
            "--set",
            "proficiency=intermediate",
        ],
    )
    assert result.exit_code == 0, result.output + result.stderr
    assert "skill-kubernetes" in result.output

    result = runner.invoke(app, ["get", "skill-kubernetes", "--json"])
    payload = json.loads(result.output)
    assert payload["tags"] == ["devops", "containers"]
    assert payload["proficiency"] == "intermediate"
    assert payload["status"] == "draft"  # safe default
    assert payload["visibility"] == "private"


def test_add_body_file_and_status(in_repo: Path, tmp_path: Path) -> None:
    body = tmp_path / "body.md"
    body.write_text("Body from a file.\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "add",
            "--type",
            "bio",
            "--title",
            "Short bio",
            "--status",
            "active",
            "--visibility",
            "public",
            "--body-file",
            str(body),
        ],
    )
    assert result.exit_code == 0, result.stderr
    result = runner.invoke(app, ["get", "bio-short-bio", "--json"])
    payload = json.loads(result.output)
    assert payload["status"] == "active"
    assert payload["visibility"] == "public"
    assert "Body from a file." in payload["body"]


def test_add_duplicate_explicit_id_exits_1(in_repo: Path) -> None:
    result = runner.invoke(
        app, ["add", "--type", "skill", "--title", "SQL", "--set", "id=skill-sql"]
    )
    assert result.exit_code == 1
    assert "duplicate" in result.stderr


def test_add_invalid_type_exits_1(in_repo: Path) -> None:
    result = runner.invoke(app, ["add", "--type", "nonsense", "--title", "X"])
    assert result.exit_code == 1


def test_add_unresolved_related_warns_on_stderr(in_repo: Path) -> None:
    result = runner.invoke(
        app,
        ["add", "--type", "goal", "--title", "Ghosted", "--related", "skill-ghost"],
    )
    assert result.exit_code == 0, result.stderr
    assert "skill-ghost" in result.stderr  # warning, not error


def test_update_set_and_body(in_repo: Path, tmp_path: Path) -> None:
    body = tmp_path / "new-body.md"
    body.write_text("Replaced body.", encoding="utf-8")
    result = runner.invoke(
        app,
        ["update", "skill-sql", "--set", "proficiency=advanced", "--body-file", str(body)],
    )
    assert result.exit_code == 0, result.stderr

    payload = json.loads(runner.invoke(app, ["get", "skill-sql", "--json"]).output)
    assert payload["proficiency"] == "advanced"

    raw = runner.invoke(app, ["get", "skill-sql"]).output
    assert "Replaced body." in raw


def test_update_nothing_exits_1(in_repo: Path) -> None:
    result = runner.invoke(app, ["update", "skill-sql"])
    assert result.exit_code == 1
    assert "nothing to update" in result.stderr


def test_update_missing_id_exits_1(in_repo: Path) -> None:
    result = runner.invoke(app, ["update", "ghost", "--set", "description=x"])
    assert result.exit_code == 1


def test_rm_soft_deletes(in_repo: Path) -> None:
    result = runner.invoke(app, ["rm", "skill-docker"])
    assert result.exit_code == 0, result.stderr
    assert (in_repo / "memory" / ".trash" / "skill-docker.md").is_file()
    assert runner.invoke(app, ["get", "skill-docker"]).exit_code == 1


def test_new_scaffolds_interactively(in_repo: Path) -> None:
    result = runner.invoke(app, ["new", "skill"], input="Rust\nsystems programming\n")
    assert result.exit_code == 0, result.output + result.stderr
    assert "skill-rust" in result.output
    assert "proficiency" in result.output  # hints at the type-specific fields
    payload = json.loads(runner.invoke(app, ["get", "skill-rust", "--json"]).output)
    assert payload["status"] == "draft"


# --- validate ------------------------------------------------------------------------


def test_validate_clean_exits_0(in_repo: Path) -> None:
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 0, result.output + result.stderr
    assert "OK" in result.output
    assert "11" in result.output


def test_validate_bad_file_exits_nonzero(in_repo: Path) -> None:
    write_memory(in_repo, id="alien-thing", type="alien")
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 1
    assert "alien" in result.stderr


def test_validate_json_report_parses(in_repo: Path) -> None:
    write_memory(in_repo, id="alien-thing", type="alien")
    result = runner.invoke(app, ["validate", "--json"])
    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["valid"] is False
    issue = report["issues"][0]
    assert set(issue) == {"path", "field", "reason"}
    assert "alien" in issue["reason"]


def test_validate_unresolved_related_warns_but_passes(in_repo: Path) -> None:
    write_memory(in_repo, id="goal-later", type="goal", related="[skill-not-yet]")
    result = runner.invoke(app, ["validate", "--json"])
    assert result.exit_code == 0, result.stderr
    report = json.loads(result.output)
    assert report["valid"] is True
    assert "skill-not-yet" in report["warnings"][0]["reason"]


# --- search / reindex ------------------------------------------------------------------


def test_search_falls_back_and_warns_without_vectors(
    in_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    block_module(monkeypatch, "lancedb")
    result = runner.invoke(app, ["search", "asyncio"])
    assert result.exit_code == 0, result.output + result.stderr
    assert "falling back to fulltext" in result.stderr
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert lines[0].startswith("skill-async-python")


def test_search_json_output(in_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    block_module(monkeypatch, "lancedb")
    result = runner.invoke(app, ["search", "asyncio", "--json", "--limit", "1"])
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)  # the fallback warning goes to stderr, not the JSON
    assert [entry["id"] for entry in payload] == ["skill-async-python"]


def test_search_unknown_mode_exits_2(in_repo: Path) -> None:
    result = runner.invoke(app, ["search", "x", "--mode", "cosine"])
    assert result.exit_code == 2
    assert "unknown search mode" in result.stderr


def test_reindex_without_vectors_extra_exits_2(
    in_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    block_module(monkeypatch, "lancedb")
    result = runner.invoke(app, ["reindex"])
    assert result.exit_code == 2
    assert "memoryhub[vectors]" in result.stderr


def test_reindex_then_search_end_to_end(in_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("lancedb")
    monkeypatch.setattr("memoryhub.index.get_embedder", lambda config: TopicEmbedder())

    result = runner.invoke(app, ["reindex"])
    assert result.exit_code == 0, result.stderr
    assert "11 embedded" in result.output

    result = runner.invoke(app, ["reindex"])  # incremental no-op
    assert "0 embedded, 11 reused" in result.output

    result = runner.invoke(app, ["search", "docker", "--limit", "3"])
    assert result.exit_code == 0, result.stderr
    assert "falling back" not in result.stderr
    assert "skill-docker" in result.output
