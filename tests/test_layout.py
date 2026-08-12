"""Nested store layout: profile-declared subfolder rules, nested writes, and ``relayout``.

Reads were already location-agnostic (the loader recurses), so these tests hold both halves of
that claim: new writes land at the canonical nested path, and every read/write path keeps working
over a store that mixes flat and nested files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import write_graph_doc, write_node, write_subnode
from memoryhub import Hub, LoadError, WriteError, ids, load_config, load_profile, writer
from memoryhub.cli import app
from memoryhub.profiles import LayoutRule

runner = CliRunner()


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*.md")}


# --- id derivation ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("id", "expected"),
    [
        ("demo-hardening-01", "demo-hardening"),
        ("demo-hardening-01-r1", "demo-hardening"),
        ("demo-hardening-01-r12", "demo-hardening"),
        ("demo-hardening", "demo-hardening"),
        ("03", "03"),
    ],
)
def test_parent_id_strips_trailing_sequence_segments(id: str, expected: str) -> None:
    assert ids.parent_id(id) == expected


# --- the rules --------------------------------------------------------------------


def test_rule_transforms() -> None:
    assert LayoutRule(field="supernode").subfolder({"supernode": "demo-site"}) == "demo-site"
    parent = LayoutRule(field="node", transform="parent-id")
    assert parent.subfolder({"node": "demo-site-01"}) == "demo-site"
    month = LayoutRule(field="id", transform="year-month")
    assert month.subfolder({"id": "orch-20260812-2"}) == "2026-08"


@pytest.mark.parametrize(
    "fields",
    [
        {},  # field absent
        {"supernode": ""},  # empty
        {"supernode": 7},  # not a string
        {"supernode": "../escape"},  # traversal
        {"supernode": "a/b"},  # separator
        {"supernode": ".hidden"},  # dot-directory (the loader skips those)
    ],
)
def test_unsafe_or_missing_values_fall_back_to_the_type_folder(fields: dict) -> None:
    assert LayoutRule(field="supernode").subfolder(fields) is None


def test_year_month_without_a_date_falls_back() -> None:
    rule = LayoutRule(field="id", transform="year-month")
    assert rule.subfolder({"id": "orch-nodate-1"}) is None


def test_profile_without_layout_keeps_every_type_flat() -> None:
    profile = load_profile("personal")
    assert profile.layout == {}
    assert profile.subfolder_for("skill", {"id": "skill-rust", "type": "skill"}) is None
    assert writer.canonical_relpath(profile, {"id": "skill-rust", "type": "skill"}) == Path(
        "skill/skill-rust.md"
    )


def test_layout_profile_composes_nested_paths(layout_repo: Path) -> None:
    profile = load_profile(layout_repo / "workflow.yaml")
    cases = [
        (
            "demo-hardening-01",
            "node",
            {"supernode": "demo-hardening"},
            "node/demo-hardening/demo-hardening-01.md",
        ),
        (
            "demo-hardening-01-impl",
            "subnode",
            {"node": "demo-hardening-01"},
            "subnode/demo-hardening/demo-hardening-01-impl.md",
        ),
        ("orch-20260812-2", "orchestration", {}, "orchestration/2026-08/orch-20260812-2.md"),
        ("demo-hardening", "supernode", {}, "supernode/demo-hardening.md"),
        ("demo", "project", {}, "project/demo.md"),
    ]
    for id, type_, extra, expected in cases:
        fields = {"id": id, "type": type_, **extra}
        assert writer.canonical_relpath(profile, fields) == Path(expected)


# --- writes ------------------------------------------------------------------------


def _add_node(hub: Hub, id: str, supernode: str) -> Path:
    doc = hub.add(
        type="node",
        title=id,
        description="A node.",
        id=id,
        supernode=supernode,
        depends_on=[],
        subnodes=[],
        attempt=0,
        body="Body.",
    )
    assert doc.path is not None
    return doc.path


def test_add_writes_to_the_nested_canonical_path(layout_repo: Path) -> None:
    hub = Hub(layout_repo)
    path = _add_node(hub, "demo-site-01", "demo-site")
    assert path == (layout_repo / "graph/node/demo-site/demo-site-01.md").resolve()
    assert hub.get("demo-site-01").frontmatter.extra["supernode"] == "demo-site"


def test_add_is_unaffected_when_the_profile_declares_no_layout(workflow_repo: Path) -> None:
    hub = Hub(workflow_repo)
    path = _add_node(hub, "demo-site-01", "demo-site")
    assert path == (workflow_repo / "graph/node/demo-site-01.md").resolve()


def test_personal_store_bytes_are_unchanged(content_repo: Path) -> None:
    hub = Hub(content_repo)
    doc = hub.add(type="skill", title="Rust", body="Learning it.")
    assert doc.path == (content_repo / "memory" / "skill" / "skill-rust.md").resolve()


def test_update_get_and_delete_work_on_a_nested_file(layout_repo: Path) -> None:
    hub = Hub(layout_repo)
    _add_node(hub, "demo-site-01", "demo-site")
    updated = hub.update("demo-site-01", status="implemented")
    assert updated.frontmatter.status == "implemented"
    assert updated.path == (layout_repo / "graph/node/demo-site/demo-site-01.md").resolve()
    assert hub.get("demo-site-01").frontmatter.status == "implemented"
    hub.delete("demo-site-01")
    assert (layout_repo / "graph/.trash/demo-site-01.md").is_file()
    assert not (layout_repo / "graph/node/demo-site/demo-site-01.md").exists()


def test_a_store_mixing_flat_and_nested_files_validates(flat_graph_layout_repo: Path) -> None:
    """Half the store nested, half not — `validate` and `graph validate` are depth-agnostic."""
    hub = Hub(flat_graph_layout_repo)
    hub.add(
        type="node",
        title="demo-site-02",
        description="A node.",
        id="demo-site-02",
        supernode="demo-site",
        depends_on=[],
        subnodes=["demo-site-02-plan"],
        attempt=0,
        body="Body.",
    )
    hub.add(
        type="subnode",
        title="demo-site-02 plan",
        description="A plan.",
        id="demo-site-02-plan",
        status="done",
        node="demo-site-02",
        role="plan",
        verdict="planned",
        body="Body.",
    )
    nested = flat_graph_layout_repo / "graph/node/demo-site/demo-site-02.md"
    flat = flat_graph_layout_repo / "graph/node/demo-hardening-01.md"
    assert nested.is_file() and flat.is_file()
    assert hub.validate().ok
    graph_report = hub.graph().validate()
    assert graph_report.ok, [str(i) for i in graph_report.issues]
    assert {doc.id for doc in hub.all()} >= {"demo-site-02", "demo-hardening-01"}


def test_export_is_location_agnostic(seeded_repo: Path, tmp_path: Path) -> None:
    """Nesting the source store must not change what an export produces."""
    hub = Hub(seeded_repo)
    flat_report = hub.export(tmp_path / "flat")
    moved = seeded_repo / "memory" / "bio" / "nested" / "bio-me.md"
    moved.parent.mkdir(parents=True)
    (seeded_repo / "memory" / "bio" / "bio-me.md").rename(moved)
    nested_report = Hub(seeded_repo).export(tmp_path / "nested")
    assert sorted(nested_report.written) == sorted(flat_report.written)
    assert (tmp_path / "nested" / "memory" / "bio" / "bio-me.md").is_file()


def test_reindex_is_location_agnostic(seeded_repo: Path) -> None:
    """The index is built from loaded documents, never from their paths — depth cannot reach it."""
    pytest.importorskip("lancedb")
    from conftest import TopicEmbedder
    from memoryhub import VectorIndex

    flat_ids = {doc.id for doc in Hub(seeded_repo).all()}
    moved = seeded_repo / "memory" / "bio" / "nested" / "bio-me.md"
    moved.parent.mkdir(parents=True)
    (seeded_repo / "memory" / "bio" / "bio-me.md").rename(moved)

    hub = Hub(seeded_repo)
    index = VectorIndex(hub.config, embedder=TopicEmbedder())
    stats = index.reindex(hub.all())
    assert stats.total == len(flat_ids)
    assert set(index.content_hashes()) == flat_ids


# --- relayout ----------------------------------------------------------------------


def test_relayout_dry_run_reports_moves_without_touching_the_store(
    flat_graph_layout_repo: Path,
) -> None:
    before = _files(flat_graph_layout_repo / "graph")
    report = Hub(flat_graph_layout_repo).relayout()
    assert not report.applied
    assert _files(flat_graph_layout_repo / "graph") == before
    moved_ids = {move.id for move in report.moves}
    assert "demo-hardening-01" in moved_ids and "demo-hardening-01-impl" in moved_ids
    # supernodes and the project have no rule, so they are already canonical
    assert "demo-hardening" not in moved_ids and "demo" not in moved_ids
    assert report.unchanged == 3


def test_relayout_apply_moves_files_and_is_idempotent(flat_graph_layout_repo: Path) -> None:
    hub = Hub(flat_graph_layout_repo)
    ids_before = {doc.id for doc in hub.all()}
    report = hub.relayout(apply=True)
    assert report.applied and report.moves

    root = flat_graph_layout_repo / "graph"
    assert (root / "node/demo-hardening/demo-hardening-01.md").is_file()
    assert (root / "subnode/demo-site/demo-site-01-test.md").is_file()
    assert (root / "supernode/demo-hardening.md").is_file()
    assert not (root / "node/demo-hardening-01.md").exists()

    assert {doc.id for doc in Hub(flat_graph_layout_repo).all()} == ids_before
    assert Hub(flat_graph_layout_repo).graph().validate().ok

    again = Hub(flat_graph_layout_repo).relayout(apply=True)
    assert again.moves == []
    assert again.unchanged == len(ids_before)


def test_relayout_on_a_profile_without_layout_has_nothing_to_do(graph_repo: Path) -> None:
    report = Hub(graph_repo).relayout(apply=True)
    assert report.moves == []
    assert report.unchanged > 0


def test_relayout_refuses_when_a_target_is_occupied(flat_graph_layout_repo: Path) -> None:
    """A file already sitting at a canonical path stops the whole run — before any move.

    Here it is a legal document under a filename that is not its id, so the store still loads:
    the refusal is relayout's own, not the loader's.
    """
    root = flat_graph_layout_repo / "graph"
    write_node(
        flat_graph_layout_repo,
        "demo-hardening-99",
        filename="demo-hardening/demo-hardening-01",
        extras={"supernode": "demo-hardening"},
    )
    before = _files(root)
    with pytest.raises(WriteError, match="already exists"):
        writer.relayout(load_config(flat_graph_layout_repo), apply=True)
    assert _files(root) == before


def test_relayout_refuses_on_a_write_policy_block(flat_graph_layout_repo: Path) -> None:
    toml = (flat_graph_layout_repo / "hub.toml").read_text(encoding="utf-8")
    (flat_graph_layout_repo / "hub.toml").write_text(
        toml.replace("allow_agent_writes = true", "allow_agent_writes = false"), encoding="utf-8"
    )
    config = load_config(flat_graph_layout_repo)
    assert writer.relayout(config).moves  # a dry run touches nothing, so it is always allowed
    with pytest.raises(WriteError, match="allow_agent_writes"):
        writer.relayout(config, apply=True)


def test_relayout_refuses_when_two_documents_claim_one_path(layout_repo: Path) -> None:
    """Same id, two files — the filename is the id, so that is what a path collision looks like."""
    write_node(layout_repo, "demo-site-01", extras={"supernode": "demo-site"})
    write_node(layout_repo, "demo-site-01", filename="copy", extras={"supernode": "demo-site"})
    with pytest.raises(LoadError, match="duplicate id"):
        writer.relayout(load_config(layout_repo))


# --- CLI ---------------------------------------------------------------------------


def test_cli_relayout_dry_run_then_apply(
    flat_graph_layout_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(flat_graph_layout_repo)
    dry = runner.invoke(app, ["relayout"])
    assert dry.exit_code == 0, dry.output
    assert "would move" in dry.output and "Would move" in dry.output
    assert (flat_graph_layout_repo / "graph/node/demo-hardening-01.md").is_file()

    applied = runner.invoke(app, ["relayout", "--apply", "--json"])
    assert applied.exit_code == 0, applied.output
    payload = json.loads(applied.output)
    assert payload["applied"] is True
    assert {"id": "demo", "src": "project/demo.md", "dest": "project/demo.md"} not in payload[
        "moves"
    ]
    assert (flat_graph_layout_repo / "graph/node/demo-hardening/demo-hardening-01.md").is_file()

    validate = runner.invoke(app, ["graph", "validate"])
    assert validate.exit_code == 0, validate.output


def test_cli_new_scaffolds_to_the_nested_canonical_path(
    layout_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`hub new` must reach every layout rule: a field-derived one, and the id-derived one."""
    monkeypatch.chdir(layout_repo)
    cases = [
        (
            ["new", "node", "--title", "Demo node", "--set", "supernode=demo-site"],
            "graph/node/demo-site",
        ),
        (
            [
                "new",
                "subnode",
                "--title",
                "Demo notes",
                "--set",
                "node=demo-site-01",
                "--set",
                "role=impl",
                "--set",
                "verdict=implemented",
                "--set",
                "status=done",
            ],
            "graph/subnode/demo-site",
        ),
        (
            ["new", "orchestration", "--title", "Window", "--id", "orch-20260812-2"],
            "graph/orchestration/2026-08",
        ),
    ]
    for argv, expected_dir in cases:
        result = runner.invoke(app, [*argv, "--description", "Scaffolded."])
        assert result.exit_code == 0, result.output
        written = Path(result.output.split(" at ")[1].split(" (status")[0])
        assert written.parent == (layout_repo / expected_dir).resolve()
        assert written.is_file()


def test_cli_new_stays_flat_without_layout(
    workflow_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same invocation against a layout-less profile is unchanged: straight in `node/`."""
    monkeypatch.chdir(workflow_repo)
    result = runner.invoke(
        app,
        ["new", "node", "--title", "Demo node", "--description", "d", "--set", "supernode=demo"],
    )
    assert result.exit_code == 0, result.output
    assert (workflow_repo / "graph/node/node-demo-node.md").is_file()


def test_cli_new_rejects_an_unknown_field(
    layout_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--set` widens what `new` can supply, not what the profile accepts."""
    monkeypatch.chdir(layout_repo)
    result = runner.invoke(
        app,
        ["new", "node", "--title", "Demo", "--description", "d", "--set", "nonsense=1"],
    )
    assert result.exit_code != 0


def test_cli_add_reports_the_nested_path(
    layout_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(layout_repo)
    write_graph_doc(
        layout_repo, id="demo-site", type="supernode", extras={"project": "demo", "nodes": "[]"}
    )
    result = runner.invoke(
        app,
        [
            "add",
            "--type",
            "subnode",
            "--title",
            "Notes",
            "--status",
            "done",
            "--set",
            "id=demo-site-01-impl",
            "--set",
            "node=demo-site-01",
            "--set",
            "role=impl",
            "--set",
            "verdict=implemented",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "subnode/demo-site/demo-site-01-impl.md" in result.output.replace("\\", "/")


def test_orchestration_documents_group_by_month(layout_repo: Path) -> None:
    hub = Hub(layout_repo)
    doc = hub.add(
        type="orchestration",
        title="Window",
        description="A window.",
        id="orch-20260812-2",
        status="planned",
        body="Body.",
    )
    assert doc.path == (layout_repo / "graph/orchestration/2026-08/orch-20260812-2.md").resolve()
    write_subnode(layout_repo, "demo-site-01", "impl", "implemented")  # flat, still readable
    assert {d.id for d in Hub(layout_repo).all()} == {"orch-20260812-2", "demo-site-01-impl"}
