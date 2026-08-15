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


def _warnings(report: StoreReport) -> list[tuple[str, str]]:
    return [(warning.field, warning.reason) for warning in report.warnings]


# --- traversal ---------------------------------------------------------------------


def test_project_scope_walks_started_supernodes_first(graph_repo: Path) -> None:
    # id order would put the backlog first; `in-progress` outranks `planned`.
    write_graph_doc(
        graph_repo,
        id="demo-backlog",
        type="supernode",
        status="planned",
        extras={"project": "demo", "nodes": "[]"},
    )
    assert [doc.id for doc in _graph_of(graph_repo).supernodes("demo")] == [
        "demo-hardening",
        "demo-site",
        "demo-backlog",
    ]


def test_supernode_scope_is_just_itself(graph: Graph) -> None:
    assert [doc.id for doc in graph.supernodes("demo-hardening")] == ["demo-hardening"]


def test_nodes_follow_the_supernodes_nodes_list(graph: Graph) -> None:
    assert [doc.id for doc in graph.nodes("demo")] == [
        "demo-hardening-01",
        "demo-hardening-02",
        "demo-hardening-03",
        "demo-site-01",
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
        "demo-site-01",
    ]


def test_ready_filters_by_status(graph: Graph) -> None:
    assert [doc.id for doc in graph.ready("demo", ["rejected"])] == ["demo-site-01"]


def test_ready_hides_a_blocked_human_gated_node_unless_asked(graph_repo: Path) -> None:
    # Migration writes human-pending items straight to `needs-feedback` (plan §8.2), so one can
    # be gated on a person *and* blocked on unbuilt code. The person can still decide.
    write_node(
        graph_repo,
        "demo-hardening-04",
        status="needs-feedback",
        extras={"depends_on": "[demo-hardening-03]", "subnodes": "[demo-hardening-04-plan]"},
    )
    write_subnode(graph_repo, "demo-hardening-04", "plan", "needs-feedback")
    graph = _graph_of(graph_repo)
    assert graph.ready("demo", ["needs-feedback"]) == []
    assert [doc.id for doc in graph.ready("demo", ["needs-feedback"], include_blocked=True)] == [
        "demo-hardening-04"
    ]


def test_ready_treats_feedback_ready_as_another_human_gate(graph_repo: Path) -> None:
    # A prepped decision is still work, so it is actionable by default — but it is a *person's*
    # work, so like `needs-feedback` it stays hidden while blocked unless the queue asks.
    write_node(
        graph_repo,
        "demo-hardening-04",
        status="feedback-ready",
        extras={
            "depends_on": "[demo-hardening-03]",
            "subnodes": "[demo-hardening-04-plan, demo-hardening-04-prep]",
        },
    )
    write_subnode(graph_repo, "demo-hardening-04", "plan", "needs-feedback")
    write_subnode(graph_repo, "demo-hardening-04", "prep", "feedback-ready")
    graph = _graph_of(graph_repo)
    assert graph.ready("demo", ["feedback-ready"]) == []
    assert [doc.id for doc in graph.ready("demo", ["feedback-ready"], include_blocked=True)] == [
        "demo-hardening-04"
    ]
    assert "demo-hardening-04" in [doc.id for doc in graph.ready("demo", include_blocked=True)]


def test_ready_skips_done_supernodes(graph_repo: Path) -> None:
    write_graph_doc(
        graph_repo,
        id="demo-site",
        type="supernode",
        status="done",
        extras={"project": "demo", "nodes": "[demo-site-01]"},
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
    assert [doc.id for doc in _graph_of(graph_repo).ready("demo")] == ["demo-site-01"]


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
    payload = graph.next("demo-site")
    assert payload is not None
    assert payload["repository"] == "DemoSite"  # the node overrides its project's repo
    assert payload["reads"] == ["demo-site-01-plan", "demo-site-01-test"]


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


def test_next_on_a_human_gated_node_reads_every_role(graph_repo: Path) -> None:
    # Here the raiser is a blocked `/implement` (plan §11.12), not the tester — the status alone
    # never says who raised it, so the human gets the newest record of each role.
    write_node(
        graph_repo,
        "demo-hardening-04",
        status="needs-feedback",
        extras={
            "subnodes": (
                "[demo-hardening-04-plan, demo-hardening-04-test, demo-hardening-04-fdbk, "
                "demo-hardening-04-impl]"
            ),
            "attempt": 1,
        },
    )
    for suffix, role, verdict in (
        ("plan", "plan", "planned"),
        ("test", "test", "rejected"),
        ("fdbk", "fdbk", "needs-fix"),
        ("impl", "impl", "needs-feedback"),
    ):
        write_subnode(graph_repo, "demo-hardening-04", role, verdict, suffix=suffix)
    payload = _graph_of(graph_repo).next("demo-hardening", ["needs-feedback"])
    assert payload is not None
    assert payload["reads"] == [
        "demo-hardening-04-plan",
        "demo-hardening-04-test",
        "demo-hardening-04-fdbk",
        "demo-hardening-04-impl",
    ]


def test_next_on_a_needs_fix_node_reads_the_findings_behind_the_decision(graph_repo: Path) -> None:
    # The `-fdbk` records the human's ruling; the findings it ruled on are in the `-test`.
    write_node(
        graph_repo,
        "demo-hardening-04",
        status="needs-fix",
        extras={
            "subnodes": (
                "[demo-hardening-04-plan, demo-hardening-04-impl, demo-hardening-04-test, "
                "demo-hardening-04-fdbk]"
            ),
            "attempt": 1,
        },
    )
    for suffix, role, verdict in (
        ("plan", "plan", "planned"),
        ("impl", "impl", "implemented"),
        ("test", "test", "needs-feedback"),
        ("fdbk", "fdbk", "needs-fix"),
    ):
        write_subnode(graph_repo, "demo-hardening-04", role, verdict, suffix=suffix)
    payload = _graph_of(graph_repo).next("demo-hardening", ["needs-fix"])
    assert payload is not None
    assert payload["reads"] == [
        "demo-hardening-04-plan",
        "demo-hardening-04-test",
        "demo-hardening-04-fdbk",
    ]


def test_next_on_a_feedback_ready_node_hands_the_human_the_prepared_decision(
    graph_repo: Path,
) -> None:
    # The whole point of the two-step loop: the `-prep` record carries the choices, the
    # recommendation and the playbook, so it is the record the human came to read.
    write_node(
        graph_repo,
        "demo-hardening-04",
        status="feedback-ready",
        extras={
            "subnodes": (
                "[demo-hardening-04-plan, demo-hardening-04-impl, demo-hardening-04-test, "
                "demo-hardening-04-prep]"
            ),
            "attempt": 1,
        },
    )
    for role, verdict in (
        ("plan", "planned"),
        ("impl", "implemented"),
        ("test", "needs-feedback"),
        ("prep", "feedback-ready"),
    ):
        write_subnode(graph_repo, "demo-hardening-04", role, verdict)
    payload = _graph_of(graph_repo).next("demo-hardening", ["feedback-ready"])
    assert payload is not None
    assert payload["reads"] == [
        "demo-hardening-04-plan",
        "demo-hardening-04-impl",
        "demo-hardening-04-test",
        "demo-hardening-04-prep",
    ]


def test_next_on_a_node_prep_sent_to_needs_fix_reads_the_prep_spec(graph_repo: Path) -> None:
    # Bounded prep autonomy: no human ruled here, so there is no `-fdbk` — the fix spec the
    # developer must work from is in the `-prep`.
    write_node(
        graph_repo,
        "demo-hardening-04",
        status="needs-fix",
        extras={
            "subnodes": (
                "[demo-hardening-04-plan, demo-hardening-04-impl, demo-hardening-04-test, "
                "demo-hardening-04-prep]"
            ),
            "attempt": 1,
        },
    )
    for role, verdict in (
        ("plan", "planned"),
        ("impl", "implemented"),
        ("test", "needs-feedback"),
        ("prep", "needs-fix"),
    ):
        write_subnode(graph_repo, "demo-hardening-04", role, verdict)
    payload = _graph_of(graph_repo).next("demo-hardening", ["needs-fix"])
    assert payload is not None
    assert payload["reads"] == [
        "demo-hardening-04-plan",
        "demo-hardening-04-test",
        "demo-hardening-04-prep",
    ]


def test_next_returns_none_when_nothing_is_ready(graph: Graph) -> None:
    assert graph.next("demo", ["needs-feedback"]) is None


# --- claim-check -------------------------------------------------------------------


def test_claim_of_a_claimable_node_is_the_walks_own_payload(graph: Graph) -> None:
    # The whole point: one computation, so the claim and the walk can never disagree.
    assert graph.claim("demo", "demo-hardening-02") == graph.next("demo")


def test_claim_reaches_past_the_walks_first_node(graph: Graph) -> None:
    # `next` would answer 02; the orchestrator is asking about the node it assigned.
    payload = graph.claim("demo", "demo-site-01", ["rejected"])
    assert payload["node"]["id"] == "demo-site-01"
    assert payload["repository"] == "DemoSite"
    assert payload["reads"] == ["demo-site-01-plan", "demo-site-01-test"]


def test_claim_refuses_a_status_the_caller_may_not_act_on(graph: Graph) -> None:
    assert graph.claim("demo", "demo-site-01", ["planned", "needs-fix"]) == {
        "node": None,
        "reason": "status rejected not in planned,needs-fix",
    }


def test_claim_refuses_a_node_with_an_unmet_dependency(graph: Graph) -> None:
    assert graph.claim("demo", "demo-hardening-03") == {
        "node": None,
        "reason": "blocked by demo-hardening-02",
    }


def test_claim_refuses_a_node_outside_the_scope(graph: Graph) -> None:
    assert graph.claim("demo-hardening", "demo-site-01", ["rejected"]) == {
        "node": None,
        "reason": "out-of-scope",
    }


def test_claim_refuses_an_unknown_id(graph: Graph) -> None:
    # An id that resolves to nothing is not in the scope's walk either — same gate, same reason.
    assert graph.claim("demo", "demo-hardening-99") == {"node": None, "reason": "out-of-scope"}


def test_claim_on_an_unknown_scope_still_raises(graph: Graph) -> None:
    # A bad scope is the caller's error, unlike a node that merely isn't claimable.
    with pytest.raises(GraphError):
        graph.claim("nope", "demo-hardening-02")


def _write_done_track(repo: Path, supernode_id: str = "demo-audit") -> str:
    """A finished supernode plus its one done node — the audit-pin fixture."""
    write_graph_doc(
        repo,
        id=supernode_id,
        type="supernode",
        status="done",
        body="The recorded intent for this finished track.",
        extras={"project": "demo", "nodes": f"[{supernode_id}-01]"},
    )
    write_node(
        repo,
        f"{supernode_id}-01",
        status="done",
        extras={
            "subnodes": (
                f"[{supernode_id}-01-plan, {supernode_id}-01-impl, {supernode_id}-01-test]"
            ),
            "attempt": 1,
        },
    )
    write_subnode(repo, f"{supernode_id}-01", "plan", "planned")
    write_subnode(repo, f"{supernode_id}-01", "impl", "implemented")
    write_subnode(repo, f"{supernode_id}-01", "test", "done")
    return supernode_id


def test_claim_of_a_done_supernode_briefs_the_body_and_nodes(graph_repo: Path) -> None:
    _write_done_track(graph_repo)
    payload = _graph_of(graph_repo).claim("demo", "demo-audit")
    assert payload["node"]["id"] == "demo-audit"
    assert payload["node"]["type"] == "supernode"
    assert payload["node"]["status"] == "done"
    assert payload["node"]["body"].strip() == "The recorded intent for this finished track."
    assert payload["supernode"] == "demo-audit"
    assert payload["project"] == "demo"
    assert payload["repository"] == "Demo"
    assert payload["reads"] == ["demo-audit-01"]
    assert "reason" not in payload


def test_claim_of_a_supernode_honours_an_explicit_status_set(graph: Graph) -> None:
    payload = graph.claim("demo", "demo-hardening", ["in-progress"])
    assert payload["node"]["id"] == "demo-hardening"
    assert payload["reads"] == ["demo-hardening-01", "demo-hardening-02", "demo-hardening-03"]


def test_claim_refuses_a_supernode_whose_status_is_not_in_the_set(graph: Graph) -> None:
    # Default for a supernode is `done`; the fixture tracks are still in-progress.
    assert graph.claim("demo", "demo-hardening") == {
        "node": None,
        "reason": "status in-progress not in done",
    }


def test_claim_refuses_a_supernode_outside_the_named_scope(graph_repo: Path) -> None:
    _write_done_track(graph_repo)
    assert _graph_of(graph_repo).claim("demo-hardening", "demo-audit") == {
        "node": None,
        "reason": "out-of-scope",
    }


def test_claim_of_a_supernode_is_in_scope_when_the_scope_is_itself(graph_repo: Path) -> None:
    _write_done_track(graph_repo)
    payload = _graph_of(graph_repo).claim("demo-audit", "demo-audit")
    assert payload["node"]["id"] == "demo-audit"
    assert payload["reads"] == ["demo-audit-01"]


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
    hardening, site = report["supernodes"]
    assert hardening["id"] == "demo-hardening"
    assert hardening["counts"] == {"done": 1, "planned": 2}
    assert hardening["ready"] == 1
    assert site["counts"] == {"rejected": 1}


def test_status_of_a_supernode_scope_covers_only_itself(graph: Graph) -> None:
    report = graph.status("demo-site")
    assert [row["id"] for row in report["supernodes"]] == ["demo-site"]
    assert report["total"] == 1


# --- validate ----------------------------------------------------------------------


def test_validate_is_clean_on_a_healthy_graph(graph: Graph) -> None:
    report = graph.validate()
    assert report.ok, _problems(report)
    assert _warnings(report) == []
    assert report.checked == len(graph.docs)


def test_orchestration_documents_validate_but_do_not_enter_the_graph(graph_repo: Path) -> None:
    baseline = _graph_of(graph_repo)
    baseline_next = baseline.next("demo")
    baseline_ready = [doc.id for doc in baseline.ready("demo")]
    baseline_status = baseline.status("demo")

    # A run is a profile document, not a graph node. Its metadata is recorded on a subnode and
    # tokens intentionally stays a flat, shell/YAML-safe string for later readers.
    write_graph_doc(
        graph_repo,
        id="run-workspace-orchestration-20260731",
        type="orchestration",
        status="running",
    )
    write_graph_doc(
        graph_repo,
        id="run-workspace-orchestration-cancelled",
        type="orchestration",
        status="cancelled",
    )
    write_node(
        graph_repo,
        "demo-hardening-01",
        status="done",
        extras={
            "subnodes": (
                "[demo-hardening-01-plan, demo-hardening-01-impl, demo-hardening-01-test, "
                "demo-hardening-01-run]"
            ),
            "attempt": 2,
        },
    )
    write_graph_doc(
        graph_repo,
        id="demo-hardening-01-run",
        type="subnode",
        status="done",
        extras={
            "node": "demo-hardening-01",
            "role": "impl",
            "verdict": "done",
            "orchestration": "workspace-orchestration",
            "session": "session-01",
            "model": "gpt-5",
            "effort": "high",
            "tokens": "total=128 input=64 cached=8 output=48 reasoning=8",
        },
    )

    hub = Hub(graph_repo)
    store_report = hub.validate()
    assert store_report.ok, [str(issue) for issue in store_report.issues]
    assert "orchestration" in hub.list_types()
    assert {"running", "cancelled"} <= set(hub.profile.enums["status"])
    run_metadata = hub.get("demo-hardening-01-run")
    assert run_metadata.frontmatter.extra["tokens"] == (
        "total=128 input=64 cached=8 output=48 reasoning=8"
    )

    graph = hub.graph()
    graph_report = graph.validate()
    assert graph_report.ok, _problems(graph_report)
    assert baseline_next is not None
    next_node = graph.next("demo")
    assert next_node is not None
    assert next_node["node"]["id"] == baseline_next["node"]["id"]
    assert [doc.id for doc in graph.ready("demo")] == baseline_ready
    assert graph.status("demo") == baseline_status


#: Every arrow out of `feedback-ready`, keyed by the node that exercises it. The human rules
#: through the *existing* machinery — approve, accept the proposed fix, replan/drop/custom text,
#: "I did the playbook" (a tester validates it), cancel — plus §4e's deferral, which is the one
#: record that repeats the status instead of changing it.
_FEEDBACK_EXITS = {
    "04": "done",
    "05": "needs-fix",
    "06": "replan",
    "07": "implemented",
    "08": "superseded",
    "09": "feedback-ready",
}


def test_validate_accepts_every_arrow_of_the_feedback_loop(graph_repo: Path) -> None:
    for seq, exit_status in _FEEDBACK_EXITS.items():
        node = f"demo-hardening-{seq}"
        write_node(
            graph_repo,
            node,
            status=exit_status,
            extras={
                "subnodes": (f"[{node}-plan, {node}-impl, {node}-test, {node}-prep, {node}-fdbk]"),
                "attempt": 1,
            },
        )
        for role, verdict in (
            ("plan", "planned"),
            ("impl", "implemented"),
            ("test", "needs-feedback"),
            ("prep", "feedback-ready"),
            ("fdbk", exit_status),
        ):
            write_subnode(graph_repo, node, role, verdict)
    # Bounded prep autonomy: an unambiguous defect it found itself, with the fix spec written.
    write_node(
        graph_repo,
        "demo-hardening-10",
        status="needs-fix",
        extras={
            "subnodes": (
                "[demo-hardening-10-plan, demo-hardening-10-impl, demo-hardening-10-test, "
                "demo-hardening-10-prep]"
            ),
            "attempt": 1,
        },
    )
    for role, verdict in (
        ("plan", "planned"),
        ("impl", "implemented"),
        ("test", "needs-feedback"),
        ("prep", "needs-fix"),
    ):
        write_subnode(graph_repo, "demo-hardening-10", role, verdict)
    write_graph_doc(
        graph_repo,
        id="demo-hardening",
        type="supernode",
        status="in-progress",
        extras={
            "project": "demo",
            "nodes": "[" + ", ".join(f"demo-hardening-{n:02d}" for n in range(1, 11)) + "]",
        },
    )
    report = _graph_of(graph_repo).validate()
    assert _problems(report) == []
    assert _warnings(report) == []


def test_a_store_on_the_pre_feedback_loop_profile_still_validates(graph_repo: Path) -> None:
    # No breaking change: the new vocabulary is values the engine reads, never an enum it needs.
    profile = (graph_repo / "workflow.yaml").read_text(encoding="utf-8")
    (graph_repo / "workflow.yaml").write_text(
        profile.replace(", feedback-ready", "").replace(", prep", ""), encoding="utf-8"
    )
    hub = Hub(graph_repo)
    assert "feedback-ready" not in hub.profile.enums["status"]
    assert hub.validate().ok
    assert hub.graph().validate().ok


def test_validate_flags_a_feedback_ready_node_no_record_prepped(graph_repo: Path) -> None:
    write_node(
        graph_repo,
        "demo-hardening-04",
        status="feedback-ready",
        extras={"subnodes": "[demo-hardening-04-plan]"},
    )
    write_subnode(graph_repo, "demo-hardening-04", "plan", "planned")
    assert _problems(_graph_of(graph_repo).validate()) == [
        (
            "status",
            "status 'feedback-ready' does not match the verdict 'planned' of its newest subnode "
            "'demo-hardening-04-plan'",
        )
    ]


def test_validate_flags_a_prep_record_that_opened_the_human_gate_itself(graph_repo: Path) -> None:
    # Prep answers a question a human was asked; it may not invent one and prep its own answer.
    write_node(
        graph_repo,
        "demo-hardening-04",
        status="feedback-ready",
        extras={"subnodes": "[demo-hardening-04-plan, demo-hardening-04-prep]"},
    )
    write_subnode(graph_repo, "demo-hardening-04", "plan", "planned")
    write_subnode(graph_repo, "demo-hardening-04", "prep", "feedback-ready")
    assert _problems(_graph_of(graph_repo).validate()) == [
        (
            "subnodes",
            "subnode 'demo-hardening-04-prep' is a 'prep' record on a node that was 'planned' — "
            "prep answers a decision a human was already asked for, so it may only follow a "
            "'needs-feedback' record",
        )
    ]


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


def test_validate_flags_an_unfinished_supersession(graph_repo: Path) -> None:
    # `/replan` crashed after writing the replacement and before retiring the original.
    write_node(
        graph_repo,
        "demo-hardening-02-r1",
        extras={"supernode": "demo-hardening", "supersedes": "demo-hardening-02"},
    )
    assert _problems(_graph_of(graph_repo).validate()) == [
        (
            "supersedes",
            "supersedes 'demo-hardening-02', whose status is 'planned' — finish the "
            "supersession: append the record to that node and set it 'superseded'",
        )
    ]


def test_validate_accepts_a_superseded_node_with_no_replacement(graph_repo: Path) -> None:
    # A cancellation (plan §11.16): retired by `/feedback`, with nothing taking it over.
    write_node(
        graph_repo,
        "demo-hardening-03",
        status="superseded",
        extras={
            "depends_on": "[demo-hardening-02]",
            "subnodes": "[demo-hardening-03-plan, demo-hardening-03-fdbk]",
        },
    )
    write_subnode(graph_repo, "demo-hardening-03", "fdbk", "superseded")
    assert _problems(_graph_of(graph_repo).validate()) == []


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


def test_validate_warns_about_a_node_its_supernode_does_not_list(graph_repo: Path) -> None:
    # The bulk-migration signature: the node landed, the supernode's build order never got it.
    write_node(graph_repo, "demo-hardening-04")
    report = _graph_of(graph_repo).validate()
    assert _warnings(report) == [
        (
            "supernode",
            "supernode 'demo-hardening' does not list this node — insert it into that "
            "supernode's `nodes` at its build-order position (unlisted nodes are walked last, "
            "so ordering silently stops meaning anything)",
        )
    ]
    # It is a warning, so the gate every skill preflights on stays green.
    assert _problems(report) == []
    assert report.ok


def test_validate_warns_when_a_supernode_lags_its_finished_nodes(graph_repo: Path) -> None:
    # Plan §11.8's own example: every node is `done`, the supernode still says `in-progress`.
    write_node(
        graph_repo,
        "demo-site-01",
        status="done",
        extras={
            "subnodes": "[demo-site-01-plan, demo-site-01-impl, demo-site-01-test]",
            "attempt": 1,
            "repository": "DemoSite",
        },
    )
    write_subnode(graph_repo, "demo-site-01", "test", "done")
    report = _graph_of(graph_repo).validate()
    assert _warnings(report) == [
        (
            "status",
            "status 'in-progress' does not follow from its 1 node(s) — expected 'done'; set it "
            "(mid-loop it is legitimately one write behind, which is why this is a warning)",
        )
    ]
    # The lag is legitimate for exactly as long as the loop takes, so the gate stays green.
    assert _problems(report) == []
    assert report.ok


def test_validate_warns_when_a_planned_supernode_has_started_work(graph_repo: Path) -> None:
    write_graph_doc(
        graph_repo,
        id="demo-site",
        type="supernode",
        status="planned",
        extras={"project": "demo", "nodes": "[demo-site-01]"},
    )
    assert _warnings(_graph_of(graph_repo).validate()) == [
        (
            "status",
            "status 'planned' does not follow from its 1 node(s) — expected 'in-progress'; set "
            "it (mid-loop it is legitimately one write behind, which is why this is a warning)",
        )
    ]


def test_validate_warns_when_an_unstarted_supernode_claims_progress(graph_repo: Path) -> None:
    write_graph_doc(
        graph_repo,
        id="demo-backlog",
        type="supernode",
        status="in-progress",
        extras={"project": "demo", "nodes": "[demo-backlog-01]"},
    )
    write_node(graph_repo, "demo-backlog-01")
    assert _warnings(_graph_of(graph_repo).validate()) == [
        (
            "status",
            "status 'in-progress' does not follow from its 1 node(s) — expected 'planned'; set "
            "it (mid-loop it is legitimately one write behind, which is why this is a warning)",
        )
    ]


def test_validate_calls_a_supernode_with_no_nodes_planned_not_done(graph_repo: Path) -> None:
    # "Every node is done" is vacuously true of none of them; §3.3's order is what decides.
    write_graph_doc(
        graph_repo,
        id="demo-backlog",
        type="supernode",
        status="done",
        extras={"project": "demo", "nodes": "[]"},
    )
    assert _warnings(_graph_of(graph_repo).validate()) == [
        (
            "status",
            "status 'done' does not follow from its 0 node(s) — expected 'planned'; set it "
            "(mid-loop it is legitimately one write behind, which is why this is a warning)",
        )
    ]


def test_validate_stays_silent_when_a_listed_node_does_not_resolve(graph_repo: Path) -> None:
    # One fault, one message: an unwritten node hides whatever status it would have contributed,
    # so the expectation would be a guess. The dangling reference owns this.
    write_graph_doc(
        graph_repo,
        id="demo-backlog",
        type="supernode",
        status="done",
        extras={"project": "demo", "nodes": "[demo-backlog-01]"},
    )
    report = _graph_of(graph_repo).validate()
    assert _problems(report) == [("nodes", "id 'demo-backlog-01' does not resolve")]
    assert _warnings(report) == []


def test_validate_stays_silent_when_the_supernode_reference_is_broken(graph_repo: Path) -> None:
    # One fault, one message: the dangling reference owns this, not the membership warning.
    write_node(graph_repo, "demo-hardening-04", extras={"supernode": "demo-nope"})
    report = _graph_of(graph_repo).validate()
    assert _problems(report) == [("supernode", "id 'demo-nope' does not resolve")]
    assert _warnings(report) == []


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
    assert "demo-site-01" in result.output


def test_cli_next_is_not_an_error_when_idle(in_graph_repo: Path) -> None:
    result = runner.invoke(app, ["graph", "next", "demo", "--status", "replan", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) is None


def test_cli_next_on_an_unknown_scope_exits_1(in_graph_repo: Path) -> None:
    result = runner.invoke(app, ["graph", "next", "nope"])
    assert result.exit_code == 1


def test_cli_next_node_briefs_a_claimable_node(in_graph_repo: Path) -> None:
    result = runner.invoke(
        app, ["graph", "next", "demo", "--node", "demo-site-01", "--status", "rejected", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["node"]["id"] == "demo-site-01"
    assert "reason" not in payload


def test_cli_next_node_matches_the_walk_when_it_is_the_first_ready(in_graph_repo: Path) -> None:
    walked = runner.invoke(app, ["graph", "next", "demo", "--json"])
    checked = runner.invoke(app, ["graph", "next", "demo", "--node", "demo-hardening-02", "--json"])
    assert json.loads(checked.output) == json.loads(walked.output)


@pytest.mark.parametrize(
    ("node_id", "statuses", "reason"),
    [
        ("demo-site-01", "planned", "status rejected not in planned"),
        ("demo-hardening-03", "planned", "blocked by demo-hardening-02"),
        ("demo-hardening-99", "planned", "out-of-scope"),
    ],
)
def test_cli_next_node_reports_which_gate_failed(
    in_graph_repo: Path, node_id: str, statuses: str, reason: str
) -> None:
    result = runner.invoke(
        app, ["graph", "next", "demo", "--node", node_id, "--status", statuses, "--json"]
    )
    assert result.exit_code == 0, result.output  # not claimable is an answer, not an error
    assert json.loads(result.output) == {"node": None, "reason": reason}


def test_cli_next_node_text_output_names_the_reason(in_graph_repo: Path) -> None:
    result = runner.invoke(app, ["graph", "next", "demo", "--node", "demo-hardening-03"])
    assert result.exit_code == 0, result.output
    assert "not claimable (blocked by demo-hardening-02)" in result.output


def test_cli_next_node_still_refuses_a_supernode_id(in_graph_repo: Path) -> None:
    # 05's contract: next --node is node targets only. The claim verb is the other pin.
    result = runner.invoke(app, ["graph", "next", "demo", "--node", "demo-hardening", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"node": None, "reason": "out-of-scope"}


def test_cli_claim_of_a_claimable_node_matches_next_node(in_graph_repo: Path) -> None:
    walked = runner.invoke(app, ["graph", "next", "demo", "--node", "demo-hardening-02", "--json"])
    claimed = runner.invoke(app, ["graph", "claim", "demo", "demo-hardening-02", "--json"])
    assert claimed.exit_code == 0, claimed.output
    assert json.loads(claimed.output) == json.loads(walked.output)


@pytest.mark.parametrize(
    ("target_id", "statuses", "reason"),
    [
        ("demo-site-01", "planned", "status rejected not in planned"),
        ("demo-hardening-03", "planned", "blocked by demo-hardening-02"),
        ("demo-hardening-99", "planned", "out-of-scope"),
    ],
)
def test_cli_claim_reports_the_same_node_gates_as_next_node(
    in_graph_repo: Path, target_id: str, statuses: str, reason: str
) -> None:
    result = runner.invoke(
        app, ["graph", "claim", "demo", target_id, "--status", statuses, "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"node": None, "reason": reason}


def test_cli_claim_briefs_a_done_supernode(in_graph_repo: Path) -> None:
    _write_done_track(in_graph_repo)
    result = runner.invoke(app, ["graph", "claim", "demo", "demo-audit", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["node"]["id"] == "demo-audit"
    assert payload["node"]["body"].strip() == "The recorded intent for this finished track."
    assert payload["reads"] == ["demo-audit-01"]


def test_cli_claim_refuses_a_supernode_whose_status_is_not_in_the_set(in_graph_repo: Path) -> None:
    result = runner.invoke(app, ["graph", "claim", "demo", "demo-hardening", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "node": None,
        "reason": "status in-progress not in done",
    }


def test_cli_claim_refuses_a_supernode_outside_the_named_scope(in_graph_repo: Path) -> None:
    _write_done_track(in_graph_repo)
    result = runner.invoke(app, ["graph", "claim", "demo-hardening", "demo-audit", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"node": None, "reason": "out-of-scope"}


def test_cli_ready(in_graph_repo: Path) -> None:
    result = runner.invoke(app, ["graph", "ready", "demo"])
    assert result.exit_code == 0, result.output
    assert "demo-hardening-02" in result.output
    assert "demo-hardening-03" not in result.output

    as_json = runner.invoke(app, ["graph", "ready", "demo", "--json"])
    assert [row["id"] for row in json.loads(as_json.output)] == [
        "demo-hardening-02",
        "demo-site-01",
    ]


def test_cli_ready_include_blocked(in_graph_repo: Path) -> None:
    result = runner.invoke(app, ["graph", "ready", "demo", "--include-blocked", "--json"])
    assert result.exit_code == 0, result.output
    assert [row["id"] for row in json.loads(result.output)] == [
        "demo-hardening-02",
        "demo-hardening-03",  # blocked behind 02, and only listed because we asked
        "demo-site-01",
    ]


def test_cli_ready_full_briefs_every_node(in_graph_repo: Path) -> None:
    # What the human-loop skills call: one query, and the whole queue is briefed.
    result = runner.invoke(app, ["graph", "ready", "demo", "--full"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert [row["node"]["id"] for row in rows] == ["demo-hardening-02", "demo-site-01"]
    assert rows[0]["node"]["body"]
    assert rows[1]["repository"] == "DemoSite"
    assert rows[1]["reads"] == ["demo-site-01-plan", "demo-site-01-test"]
    # `next` is the first of these.
    first = runner.invoke(app, ["graph", "next", "demo", "--json"])
    assert json.loads(first.output) == rows[0]


def test_cli_ready_briefs_the_feedback_ready_queue(in_graph_repo: Path) -> None:
    # Exactly the call the Bridge/`/feedback` makes: every prepped decision, blocked or not.
    write_node(
        in_graph_repo,
        "demo-hardening-04",
        status="feedback-ready",
        extras={
            "depends_on": "[demo-hardening-03]",
            "subnodes": "[demo-hardening-04-plan, demo-hardening-04-prep]",
        },
    )
    write_subnode(in_graph_repo, "demo-hardening-04", "plan", "needs-feedback")
    write_subnode(in_graph_repo, "demo-hardening-04", "prep", "feedback-ready")
    result = runner.invoke(
        app,
        ["graph", "ready", "demo", "--status", "feedback-ready", "--include-blocked", "--full"],
    )
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert [row["node"]["id"] for row in rows] == ["demo-hardening-04"]
    assert rows[0]["reads"] == ["demo-hardening-04-plan", "demo-hardening-04-prep"]


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


def test_cli_validate_surfaces_a_warning_without_failing(in_graph_repo: Path) -> None:
    write_node(in_graph_repo, "demo-hardening-04")
    result = runner.invoke(app, ["graph", "validate"])
    assert result.exit_code == 0, result.output
    assert "warning: " in result.output
    assert "does not list this node" in result.output
    assert "graph invariants hold" in result.output

    as_json = runner.invoke(app, ["graph", "validate", "--json"])
    assert as_json.exit_code == 0
    report = json.loads(as_json.output)
    assert report["valid"] is True
    assert [warning["field"] for warning in report["warnings"]] == ["supernode"]


def test_cli_validate_reports_and_exits_1(in_graph_repo: Path) -> None:
    write_node(in_graph_repo, "demo-hardening-04", extras={"depends_on": "[demo-hardening-99]"})
    result = runner.invoke(app, ["graph", "validate", "--json"])
    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["valid"] is False
    assert report["issues"][0]["field"] == "depends_on"
