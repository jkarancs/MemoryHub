"""Graph tests: scope traversal, the next-node payload, status counts, and every §6 invariant.

Each invariant test starts from the healthy `graph_repo`, adds exactly one broken document, and
asserts the *complete* issue list — so a check that fires too eagerly fails here too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import write_graph_doc, write_node, write_subnode
from memoryhub.cli import app
from memoryhub.graph import Graph, GraphError
from memoryhub.hub import Hub
from memoryhub.loader import StoreReport

runner = CliRunner()


@pytest.fixture
def graph(graph_repo: Path) -> Graph:
    return Hub(graph_repo).graph()


def _graph_of(repo: Path) -> Graph:
    return Hub(repo).graph()


def _problems(report: StoreReport) -> list[tuple[str, str]]:
    return [(issue.field, issue.reason) for issue in report.issues]


# --- traversal ---------------------------------------------------------------------


def test_project_scope_walks_started_supernodes_first(graph: Graph) -> None:
    # id order would put commercial first; `in-progress` outranks `planned`.
    assert [doc.id for doc in graph.supernodes("demo")] == ["demo-hardening", "demo-commercial"]


def test_supernode_scope_is_just_itself(graph: Graph) -> None:
    assert [doc.id for doc in graph.supernodes("demo-hardening")] == ["demo-hardening"]


def test_nodes_follow_the_supernodes_nodes_list(graph: Graph) -> None:
    assert [doc.id for doc in graph.nodes("demo")] == [
        "demo-hardening-01",
        "demo-hardening-02",
        "demo-hardening-03",
        "demo-commercial-01",
    ]


def test_unlisted_nodes_are_appended_id_sorted(graph_repo: Path) -> None:
    for seq in ("05", "04"):
        write_node(graph_repo, f"demo-hardening-{seq}")
    assert [doc.id for doc in _graph_of(graph_repo).nodes("demo-hardening")] == [
        "demo-hardening-01",
        "demo-hardening-02",
        "demo-hardening-03",
        "demo-hardening-04",
        "demo-hardening-05",
    ]


def test_ready_stops_at_unmet_dependencies(graph: Graph) -> None:
    # 01 is done (not actionable), 02 is unblocked by it, 03 waits behind 02.
    assert [doc.id for doc in graph.ready("demo")] == [
        "demo-hardening-02",
        "demo-commercial-01",
    ]


def test_ready_filters_by_status(graph: Graph) -> None:
    assert [doc.id for doc in graph.ready("demo", ["rejected"])] == ["demo-commercial-01"]


def test_ready_skips_done_supernodes(graph_repo: Path) -> None:
    write_graph_doc(
        graph_repo,
        id="demo-commercial",
        type="supernode",
        status="done",
        extras={"project": "demo", "nodes": "[demo-commercial-01]"},
    )
    assert [doc.id for doc in _graph_of(graph_repo).ready("demo")] == ["demo-hardening-02"]


def test_a_superseded_dependency_is_not_done(graph_repo: Path) -> None:
    write_node(
        graph_repo,
        "demo-hardening-01",
        status="superseded",
        extras={
            "subnodes": (
                "[demo-hardening-01-plan, demo-hardening-01-impl, demo-hardening-01-test, "
                "demo-hardening-01-plan2]"
            ),
            "attempt": 1,
        },
    )
    write_subnode(graph_repo, "demo-hardening-01", "plan", "superseded", suffix="plan2")
    assert [doc.id for doc in _graph_of(graph_repo).ready("demo")] == ["demo-commercial-01"]


def test_unknown_scope_raises(graph: Graph) -> None:
    with pytest.raises(GraphError, match="no document with id"):
        graph.next("nope")


def test_a_node_is_not_a_scope(graph: Graph) -> None:
    with pytest.raises(GraphError, match="expected a project or supernode"):
        graph.ready("demo-hardening-01")


# --- next --------------------------------------------------------------------------


def test_next_is_the_first_ready_node(graph: Graph) -> None:
    payload = graph.next("demo")
    assert payload is not None
    assert payload["node"]["id"] == "demo-hardening-02"
    assert payload["node"]["body"]  # acceptance criteria travel with the node
    assert payload["supernode"] == "demo-hardening"
    assert payload["project"] == "demo"
    assert payload["repository"] == "Demo"  # inherited from the project
    assert payload["reads"] == ["demo-hardening-02-plan"]


def test_next_on_a_rejected_node_reads_the_plan_and_the_test(graph: Graph) -> None:
    payload = graph.next("demo-commercial")
    assert payload is not None
    assert payload["repository"] == "DemoSite"  # the node overrides its project's repo
    assert payload["reads"] == ["demo-commercial-01-plan", "demo-commercial-01-test"]


def test_next_reads_the_newest_subnode_of_each_role(graph_repo: Path) -> None:
    write_node(
        graph_repo,
        "demo-hardening-04",
        status="implemented",
        extras={
            "subnodes": (
                "[demo-hardening-04-plan, demo-hardening-04-impl, "
                "demo-hardening-04-test, demo-hardening-04-impl2]"
            ),
            "attempt": 2,
        },
    )
    for suffix, role, verdict in (
        ("plan", "plan", "planned"),
        ("impl", "impl", "implemented"),
        ("test", "test", "rejected"),
        ("impl2", "impl", "implemented"),
    ):
        write_subnode(graph_repo, "demo-hardening-04", role, verdict, suffix=suffix)
    payload = _graph_of(graph_repo).next("demo-hardening", ["implemented"])
    assert payload is not None
    # The retry supersedes the first attempt; `fix` has no subnode yet, so it contributes nothing.
    assert payload["reads"] == ["demo-hardening-04-plan", "demo-hardening-04-impl2"]


def test_next_returns_none_when_nothing_is_ready(graph: Graph) -> None:
    assert graph.next("demo", ["needs-feedback"]) is None


def test_repository_is_unknown_when_the_supernode_does_not_resolve(graph_repo: Path) -> None:
    write_node(graph_repo, "demo-orphan", extras={"supernode": "nope"})
    orphan = _graph_of(graph_repo)
    assert orphan.repository(orphan.by_id["demo-orphan"]) is None


# --- status ------------------------------------------------------------------------


def test_status_counts_per_supernode_and_scope(graph: Graph) -> None:
    report = graph.status("demo")
    assert report["scope"] == "demo"
    assert report["type"] == "project"
    assert report["total"] == 4
    assert report["ready"] == 2
    assert report["counts"] == {"done": 1, "planned": 2, "rejected": 1}
    hardening, commercial = report["supernodes"]
    assert hardening["id"] == "demo-hardening"
    assert hardening["counts"] == {"done": 1, "planned": 2}
    assert hardening["ready"] == 1
    assert commercial["counts"] == {"rejected": 1}


def test_status_of_a_supernode_scope_covers_only_itself(graph: Graph) -> None:
    report = graph.status("demo-commercial")
    assert [row["id"] for row in report["supernodes"]] == ["demo-commercial"]
    assert report["total"] == 1


# --- validate ----------------------------------------------------------------------


def test_validate_is_clean_on_a_healthy_graph(graph: Graph) -> None:
    report = graph.validate()
    assert report.ok, _problems(report)
    assert report.checked == len(graph.docs)


def test_validate_flags_a_dangling_reference(graph_repo: Path) -> None:
    write_node(graph_repo, "demo-hardening-04", extras={"depends_on": "[demo-hardening-99]"})
    assert _problems(_graph_of(graph_repo).validate()) == [
        ("depends_on", "id 'demo-hardening-99' does not resolve")
    ]


def test_validate_flags_a_dependency_cycle(graph_repo: Path) -> None:
    for this, other in (("04", "05"), ("05", "04")):
        write_node(
            graph_repo,
            f"demo-hardening-{this}",
            extras={"depends_on": f"[demo-hardening-{other}]"},
        )
    problems = _problems(_graph_of(graph_repo).validate())
    assert len(problems) == 1, problems  # one cycle, reported once
    field, reason = problems[0]
    assert field == "depends_on"
    assert reason.startswith("dependency cycle: demo-hardening-04 -> demo-hardening-05")


def test_validate_reports_a_cycle_once_despite_a_duplicated_edge(graph_repo: Path) -> None:
    write_node(graph_repo, "demo-hardening-04", extras={"depends_on": "[demo-hardening-05]"})
    write_node(
        graph_repo,
        "demo-hardening-05",
        extras={"depends_on": "[demo-hardening-04, demo-hardening-04]"},
    )
    problems = _problems(_graph_of(graph_repo).validate())
    assert len(problems) == 1, problems


def test_validate_flags_an_edge_into_a_superseded_node(graph_repo: Path) -> None:
    write_node(
        graph_repo,
        "demo-hardening-04",
        status="superseded",
        extras={"subnodes": "[demo-hardening-04-plan]"},
    )
    write_subnode(graph_repo, "demo-hardening-04", "plan", "superseded")
    write_node(graph_repo, "demo-hardening-05", extras={"depends_on": "[demo-hardening-04]"})
    assert _problems(_graph_of(graph_repo).validate()) == [
        (
            "depends_on",
            "depends on superseded node 'demo-hardening-04' — rewire to its replacement",
        )
    ]


def test_validate_flags_status_verdict_drift(graph_repo: Path) -> None:
    write_node(
        graph_repo,
        "demo-hardening-04",
        status="done",
        extras={"subnodes": "[demo-hardening-04-impl]", "attempt": 1},
    )
    write_subnode(graph_repo, "demo-hardening-04", "impl", "implemented")
    assert _problems(_graph_of(graph_repo).validate()) == [
        (
            "status",
            "status 'done' does not match the verdict 'implemented' of its newest subnode "
            "'demo-hardening-04-impl'",
        )
    ]


def test_validate_flags_an_attempt_that_does_not_count_impl_subnodes(graph_repo: Path) -> None:
    write_node(
        graph_repo,
        "demo-hardening-04",
        status="implemented",
        extras={"subnodes": "[demo-hardening-04-impl]"},
    )
    write_subnode(graph_repo, "demo-hardening-04", "impl", "implemented")
    assert _problems(_graph_of(graph_repo).validate()) == [
        ("attempt", "attempt is 0 but the node has 1 impl subnode(s)")
    ]


def test_validate_flags_an_id_filename_mismatch(graph_repo: Path) -> None:
    write_node(graph_repo, "demo-hardening-04", filename="demo-hardening-4")
    assert _problems(_graph_of(graph_repo).validate()) == [
        ("id", "id 'demo-hardening-04' does not match filename demo-hardening-4.md")
    ]


def test_validate_flags_a_status_that_is_illegal_for_the_type(graph_repo: Path) -> None:
    # `planned` is in the store-wide status enum (a node may be planned) but not for a subnode.
    write_node(graph_repo, "demo-hardening-04", extras={"subnodes": "[demo-hardening-04-plan]"})
    write_graph_doc(
        graph_repo,
        id="demo-hardening-04-plan",
        type="subnode",
        status="planned",
        extras={"node": "demo-hardening-04", "role": "plan", "verdict": "planned"},
    )
    assert _problems(_graph_of(graph_repo).validate()) == [
        ("status", "status 'planned' is not valid for type 'subnode' (allowed: done)")
    ]


def test_validate_flags_a_missing_required_field(graph_repo: Path) -> None:
    # No `attempt`: the rejection budget (plan §4.3) would go unchecked on this node.
    write_graph_doc(
        graph_repo,
        id="demo-hardening-04",
        type="node",
        status="planned",
        extras={"supernode": "demo-hardening", "depends_on": "[]", "subnodes": "[]"},
    )
    assert _problems(_graph_of(graph_repo).validate()) == [
        ("attempt", "a node must set 'attempt' (the graph reads it)")
    ]


def test_validate_flags_a_started_node_with_no_work_record(graph_repo: Path) -> None:
    write_node(graph_repo, "demo-hardening-04", status="implemented", extras={"attempt": 1})
    assert _problems(_graph_of(graph_repo).validate()) == [
        (
            "subnodes",
            "a 'implemented' node has no subnode — its work record is missing "
            "(only 'planned' nodes may have none)",
        ),
        ("attempt", "attempt is 1 but the node has 0 impl subnode(s)"),
    ]


def test_validate_flags_a_subnode_its_node_does_not_list(graph_repo: Path) -> None:
    # The crash signature of the write order: the subnode landed, the node never picked it up.
    write_subnode(graph_repo, "demo-hardening-02", "impl", "implemented")
    assert _problems(_graph_of(graph_repo).validate()) == [
        (
            "node",
            "node 'demo-hardening-02' does not list this subnode — an interrupted write; "
            "append it to that node's `subnodes` and set the node's status to 'implemented'",
        )
    ]


# --- CLI ---------------------------------------------------------------------------


@pytest.fixture
def in_graph_repo(graph_repo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(graph_repo)
    return graph_repo


def test_cli_help_lists_graph() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "graph" in result.output


def test_cli_next_json(in_graph_repo: Path) -> None:
    result = runner.invoke(app, ["graph", "next", "demo", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["node"]["id"] == "demo-hardening-02"
    assert payload["reads"] == ["demo-hardening-02-plan"]


def test_cli_next_text(in_graph_repo: Path) -> None:
    result = runner.invoke(app, ["graph", "next", "demo"])
    assert result.exit_code == 0, result.output
    assert "demo-hardening-02" in result.output
    assert "Demo" in result.output


def test_cli_next_honours_status_filter(in_graph_repo: Path) -> None:
    result = runner.invoke(app, ["graph", "next", "demo", "--status", "rejected,needs-fix"])
    assert result.exit_code == 0, result.output
    assert "demo-commercial-01" in result.output


def test_cli_next_is_not_an_error_when_idle(in_graph_repo: Path) -> None:
    result = runner.invoke(app, ["graph", "next", "demo", "--status", "replan", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) is None


def test_cli_next_on_an_unknown_scope_exits_1(in_graph_repo: Path) -> None:
    result = runner.invoke(app, ["graph", "next", "nope"])
    assert result.exit_code == 1


def test_cli_ready(in_graph_repo: Path) -> None:
    result = runner.invoke(app, ["graph", "ready", "demo"])
    assert result.exit_code == 0, result.output
    assert "demo-hardening-02" in result.output
    assert "demo-hardening-03" not in result.output

    as_json = runner.invoke(app, ["graph", "ready", "demo", "--json"])
    assert [row["id"] for row in json.loads(as_json.output)] == [
        "demo-hardening-02",
        "demo-commercial-01",
    ]


def test_cli_ready_on_a_node_scope_exits_1(in_graph_repo: Path) -> None:
    result = runner.invoke(app, ["graph", "ready", "demo-hardening-01"])
    assert result.exit_code == 1


def test_cli_status(in_graph_repo: Path) -> None:
    result = runner.invoke(app, ["graph", "status", "demo"])
    assert result.exit_code == 0, result.output
    assert "demo-hardening" in result.output
    assert "TOTAL" in result.output
    assert "done 1" in result.output

    as_json = runner.invoke(app, ["graph", "status", "demo", "--json"])
    assert json.loads(as_json.output)["ready"] == 2


def test_cli_status_on_an_unknown_scope_exits_1(in_graph_repo: Path) -> None:
    result = runner.invoke(app, ["graph", "status", "nope"])
    assert result.exit_code == 1


def test_cli_validate_ok(in_graph_repo: Path) -> None:
    result = runner.invoke(app, ["graph", "validate"])
    assert result.exit_code == 0, result.output
    assert "graph invariants hold" in result.output


def test_cli_validate_reports_and_exits_1(in_graph_repo: Path) -> None:
    write_node(in_graph_repo, "demo-hardening-04", extras={"depends_on": "[demo-hardening-99]"})
    result = runner.invoke(app, ["graph", "validate", "--json"])
    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["valid"] is False
    assert report["issues"][0]["field"] == "depends_on"
