"""Export tests: selection, transforms, deterministic sync, safety gates, and the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import write_memory
from memoryhub import export as export_module
from memoryhub.cli import app
from memoryhub.config import load_config
from memoryhub.export import ExportError, export_store
from memoryhub.hub import Hub
from memoryhub.writer import WriteError, add

runner = CliRunner()


@pytest.fixture
def dest(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An export destination outside the content repo's tmp tree."""
    return tmp_path_factory.mktemp("public")


def config_for(repo: Path):
    return load_config(repo)


def exported_md(dest: Path) -> set[str]:
    root = dest / "memory"
    if not root.is_dir():
        return set()
    return {p.relative_to(root).as_posix() for p in root.rglob("*.md")}


def flip_visibility(repo: Path, type_: str, id_: str, to: str) -> None:
    path = repo / "memory" / type_ / f"{id_}.md"
    text = path.read_text(encoding="utf-8")
    frm = "private" if to == "public" else "public"
    path.write_text(text.replace(f"visibility: {frm}", f"visibility: {to}"), encoding="utf-8")


# --- selection -------------------------------------------------------------------


def test_exports_only_public_and_active(seeded_repo: Path, dest: Path) -> None:
    report = export_store(config_for(seeded_repo), dest)
    # bio-me and project-site are public+active; writing-blog-post is public but archived.
    assert exported_md(dest) == {"bio/bio-me.md", "project/project-site.md"}
    assert (dest / "README.md").is_file()
    assert (dest / "hub.toml").is_file()
    assert not report.deleted
    assert len(report.written) == 3  # 2 memories + README


def test_archived_and_draft_never_export_even_when_public(content_repo: Path, dest: Path) -> None:
    write_memory(content_repo, id="bio-a", type="bio", visibility="public", status="archived")
    write_memory(content_repo, id="bio-b", type="bio", visibility="public", status="draft")
    write_memory(content_repo, id="bio-c", type="bio", visibility="private", status="active")
    export_store(config_for(content_repo), dest)
    assert exported_md(dest) == set()


# --- transforms ------------------------------------------------------------------


def test_related_to_non_exported_ids_is_stripped_with_comment(
    content_repo: Path, dest: Path
) -> None:
    write_memory(content_repo, id="bio-private", type="bio")  # private -> not exported
    write_memory(
        content_repo,
        id="bio-pub",
        type="bio",
        visibility="public",
        related="[bio-private, project-pub]",
    )
    write_memory(
        content_repo,
        id="project-pub",
        type="project",
        visibility="public",
        related="[bio-pub]",
    )
    export_store(config_for(content_repo), dest)

    pub = (dest / "memory" / "bio" / "bio-pub.md").read_text(encoding="utf-8")
    assert "bio-private" not in pub
    assert "related: [project-pub]  # 1 link(s) to non-exported memories removed" in pub
    # Fully-resolving related lists pass through without the comment.
    project = (dest / "memory" / "project" / "project-pub.md").read_text(encoding="utf-8")
    assert "related: [bio-pub]\n" in project


def test_body_passes_through_untouched(content_repo: Path, dest: Path) -> None:
    body = "First line.\n\n## Heading\n\nrelated: not-frontmatter\n"
    write_memory(content_repo, id="bio-x", type="bio", visibility="public", body=body)
    export_store(config_for(content_repo), dest)
    exported = (dest / "memory" / "bio" / "bio-x.md").read_text(encoding="utf-8")
    assert exported.endswith("---\n\n" + body)


# --- deterministic sync ----------------------------------------------------------


def test_second_run_is_a_zero_diff(seeded_repo: Path, dest: Path) -> None:
    export_store(config_for(seeded_repo), dest)
    before = {p: p.read_bytes() for p in dest.rglob("*") if p.is_file()}
    report = export_store(config_for(seeded_repo), dest)
    assert report.written == [] and report.deleted == []
    assert len(report.unchanged) == 3
    after = {p: p.read_bytes() for p in dest.rglob("*") if p.is_file()}
    assert before == after


def test_visibility_flip_adds_then_removes_exactly_that_file(seeded_repo: Path, dest: Path) -> None:
    export_store(config_for(seeded_repo), dest)

    flip_visibility(seeded_repo, "skill", "skill-sql", to="public")
    report = export_store(config_for(seeded_repo), dest)
    assert set(report.written) == {Path("README.md"), Path("memory/skill/skill-sql.md")}
    assert report.deleted == []

    flip_visibility(seeded_repo, "skill", "skill-sql", to="private")
    report = export_store(config_for(seeded_repo), dest)
    assert report.deleted == [Path("memory/skill/skill-sql.md")]
    assert report.written == [Path("README.md")]
    assert not (dest / "memory" / "skill").exists()  # emptied type dir is pruned


def test_lf_newlines_regardless_of_source_line_endings(content_repo: Path, dest: Path) -> None:
    path = write_memory(content_repo, id="bio-crlf", type="bio", visibility="public")
    path.write_bytes(path.read_text(encoding="utf-8").replace("\n", "\r\n").encode("utf-8"))
    export_store(config_for(content_repo), dest)
    raw = (dest / "memory" / "bio" / "bio-crlf.md").read_bytes()
    assert b"\r" not in raw


def test_readme_is_a_type_grouped_index(seeded_repo: Path, dest: Path) -> None:
    export_store(config_for(seeded_repo), dest)
    readme = (dest / "README.md").read_text(encoding="utf-8")
    assert "generated by `hub export`" in readme
    assert "## bio (1)" in readme
    assert "## project (1)" in readme
    assert "- [About me](memory/bio/bio-me.md) — A description." in readme


def test_dest_is_a_valid_readonly_hub_store(seeded_repo: Path, dest: Path) -> None:
    export_store(config_for(seeded_repo), dest)
    public_config = load_config(dest)
    report = Hub(public_config).validate()
    assert report.ok and not report.warnings
    with pytest.raises(WriteError, match="allow_agent_writes"):
        add(public_config, {"type": "bio", "title": "Nope"}, "")


def test_dry_run_touches_nothing(seeded_repo: Path, dest: Path) -> None:
    report = export_store(config_for(seeded_repo), dest, dry_run=True)
    assert len(report.written) == 3
    assert list(dest.iterdir()) == []
    assert "dry run" in str(report)


# --- safety gates ----------------------------------------------------------------


def test_refuses_to_export_an_invalid_store(seeded_repo: Path, dest: Path) -> None:
    (seeded_repo / "memory" / "bio" / "broken.md").write_text("no frontmatter", encoding="utf-8")
    with pytest.raises(ExportError, match="validation problem"):
        export_store(config_for(seeded_repo), dest)
    assert list(dest.iterdir()) == []


def test_refuses_a_dest_overlapping_the_source(seeded_repo: Path) -> None:
    for bad in (seeded_repo, seeded_repo / "memory", seeded_repo / "memory" / "sub"):
        with pytest.raises(ExportError, match="overlaps the source store"):
            export_store(config_for(seeded_repo), bad)


def test_sensitive_hits_refuse_without_a_confirmer(content_repo: Path, dest: Path) -> None:
    write_memory(
        content_repo,
        id="bio-leak",
        type="bio",
        visibility="public",
        body="Reach me at me@example.com or +36 30 123 4567.\nKey: sk-abcdefghij0123456789xy",
    )
    with pytest.raises(ExportError, match="sensitive"):
        export_store(config_for(content_repo), dest)
    assert list(dest.iterdir()) == []


def test_sensitive_hits_abort_when_not_confirmed(content_repo: Path, dest: Path) -> None:
    write_memory(
        content_repo,
        id="bio-leak",
        type="bio",
        visibility="public",
        body="token = sk-abcdefghij0123456789xy",
    )
    export_module.confirm_callback = lambda _prompt: False
    with pytest.raises(ExportError, match="not confirmed"):
        export_store(config_for(content_repo), dest)
    assert list(dest.iterdir()) == []


def test_sensitive_hits_export_when_confirmed(content_repo: Path, dest: Path) -> None:
    write_memory(
        content_repo,
        id="bio-mail",
        type="bio",
        visibility="public",
        description="Contact - me@example.com",  # intentional public email in a bio
    )
    prompts: list[str] = []

    def confirm(prompt: str) -> bool:
        prompts.append(prompt)
        return True

    export_module.confirm_callback = confirm
    report = export_store(config_for(content_repo), dest)
    assert exported_md(dest) == {"bio/bio-mail.md"}
    assert [(h.id, h.kind) for h in report.hits] == [("bio-mail", "email address")]
    assert "me@example.com" in prompts[0]


def test_dry_run_reports_hits_without_prompting(content_repo: Path, dest: Path) -> None:
    write_memory(
        content_repo,
        id="bio-leak",
        type="bio",
        visibility="public",
        body="AKIAABCDEFGHIJKLMNOP is an AWS key shape",
    )
    report = export_store(config_for(content_repo), dest, dry_run=True)  # no confirmer needed
    assert [h.kind for h in report.hits] == ["API key"]
    assert list(dest.iterdir()) == []


# --- CLI -------------------------------------------------------------------------


@pytest.fixture
def in_repo(seeded_repo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(seeded_repo)
    return seeded_repo


def test_cli_export_happy_path(in_repo: Path, dest: Path) -> None:
    result = runner.invoke(app, ["export", "--dest", str(dest)])
    assert result.exit_code == 0, result.output
    assert "3 written, 0 deleted, 0 unchanged" in result.output
    assert exported_md(dest) == {"bio/bio-me.md", "project/project-site.md"}


def test_cli_export_dry_run_writes_nothing(in_repo: Path, dest: Path) -> None:
    result = runner.invoke(app, ["export", "--dest", str(dest), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "dry run" in result.output
    assert list(dest.iterdir()) == []


def test_cli_export_prompts_on_hits_and_respects_the_answer(in_repo: Path, dest: Path) -> None:
    write_memory(in_repo, id="bio-leak", type="bio", visibility="public", body="me@example.com")
    result = runner.invoke(app, ["export", "--dest", str(dest)], input="n\n")
    assert result.exit_code == 1
    assert list(dest.iterdir()) == []

    result = runner.invoke(app, ["export", "--dest", str(dest)], input="y\n")
    assert result.exit_code == 0, result.output
    assert "bio/bio-leak.md" in exported_md(dest)


def test_cli_export_yes_skips_the_prompt_but_warns(in_repo: Path, dest: Path) -> None:
    write_memory(in_repo, id="bio-leak", type="bio", visibility="public", body="me@example.com")
    result = runner.invoke(app, ["export", "--dest", str(dest), "--yes"])
    assert result.exit_code == 0, result.output
    assert "possible email address" in result.stderr
    assert "bio/bio-leak.md" in exported_md(dest)


def test_cli_export_fails_cleanly_on_invalid_store(in_repo: Path, dest: Path) -> None:
    (in_repo / "memory" / "bio" / "broken.md").write_text("no frontmatter", encoding="utf-8")
    result = runner.invoke(app, ["export", "--dest", str(dest)])
    assert result.exit_code == 1
    assert "hub validate" in result.stderr
