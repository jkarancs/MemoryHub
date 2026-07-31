"""Dependency-graph traversal over a ``workflow`` store — what ``hub graph`` serves.

The engine is content-agnostic, but *this* module is not: it encodes the development-graph
data model — ``project > supernode > node > subnode``, ``depends_on`` edges, and the node
state machine — specified in ``specs/graph-workflow-plan.md`` §3–§4. It is pure read-side
traversal over already-loaded documents (no writer, no index, no MCP), so a store on any
other profile simply yields an empty graph.

Two kinds of edge, both frontmatter: ``depends_on`` is the **hard** one this module walks (a
node is ready only when every id in it is ``done``); the engine's own ``related`` holds loose
cross-links and is already warned on by ``hub validate``.

The queries:
  * :meth:`Graph.next` — the single next actionable node in a scope (the skill entry point).
  * :meth:`Graph.ready` — every ready node, in walk order.
  * :meth:`Graph.status` — counts by status per supernode (the ``PROGRESS.md`` replacement).
  * :meth:`Graph.validate` — the graph invariants a per-file schema check can't see, split into
    fatal issues and non-fatal warnings.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from .loader import StoreReport, ValidationIssue, flat_frontmatter
from .models import MemoryDoc
from .profiles import Profile

PROJECT = "project"
SUPERNODE = "supernode"
NODE = "node"
SUBNODE = "subnode"

DONE = "done"
SUPERSEDED = "superseded"

#: Node statuses that still carry work — the default scope of :meth:`Graph.ready`/:meth:`next`.
ACTIONABLE: tuple[str, ...] = (
    "planned",
    "rejected",
    "needs-fix",
    "implemented",
    "needs-feedback",
    "replan",
)

#: Which statuses are legal for which graph type. A profile declares one store-wide ``status``
#: enum (the union of the graph vocabularies plus any content-specific statuses), so "is this
#: status legal for *this* type?" lands here. Profile types outside this mapping, such as
#: ``orchestration``, are content documents and deliberately stay out of graph validation and
#: traversal.
TYPE_STATUSES: dict[str, tuple[str, ...]] = {
    PROJECT: ("active", "paused", DONE),
    SUPERNODE: ("planned", "in-progress", DONE),
    NODE: (*ACTIONABLE, DONE, SUPERSEDED),
    SUBNODE: (DONE,),  # an immutable record; its `verdict` is the payload
}

#: Subnode roles the acting skill needs to read for a node in each status (plan §4.2/§7).
#: :meth:`Graph.next` resolves each to that role's newest subnode, so one call tells a skill
#: exactly which documents to ``hub get``.
#:
#: The rule: the plan, plus the newest record of every role whose findings the reader must act
#: on — and nothing else, so an independent check stays independent (a tester never reads a
#: prior ``test``; a retrying developer reads the findings, not its own old notes). The two
#: human-gated statuses are the exception: any role can raise them (a ``test`` verdict, a
#: blocked ``impl``/``fix`` — plan §11.12 — or a previous deferral's ``fdbk``), so the status
#: does not say who did, and the human reads the node cold. They get the newest of each.
_HUMAN_GATED_READS = ("plan", "impl", "fix", "test", "fdbk")

STATUS_READS: dict[str, tuple[str, ...]] = {
    "planned": ("plan",),
    "rejected": ("plan", "test"),
    "needs-fix": ("plan", "test", "fdbk"),
    "implemented": ("plan", "impl", "fix"),
    "needs-feedback": _HUMAN_GATED_READS,
    "replan": _HUMAN_GATED_READS,
}

#: Frontmatter fields whose values are ids that must resolve to a document in the store.
_REFERENCES: dict[str, tuple[str, ...]] = {
    SUPERNODE: ("project", "nodes"),
    NODE: ("supernode", "depends_on", "subnodes", "supersedes"),
    SUBNODE: ("node",),
}

#: Fields a type must carry: exactly the ones this traversal reads. An absent field is not a
#: harmless default — it silently skips the check that would have used it (``attempt`` disables
#: the rejection budget, ``role`` hides a subnode from ``reads``, ``supernode`` makes a node
#: unreachable from any scope). Optional by design: ``supersedes`` and ``repository`` (set only
#: when they apply), ``created_by`` (a store convention, load-bearing for nothing here), and
#: ``project.repository`` (empty for cross-repo projects).
_REQUIRED: dict[str, tuple[str, ...]] = {
    SUPERNODE: ("project", "nodes"),
    NODE: ("supernode", "depends_on", "subnodes", "attempt"),
    SUBNODE: ("node", "role", "verdict"),
}

#: The one node status that legitimately has no work record yet — everything else is mid-loop.
_UNSTARTED = "planned"

#: Supernode walk order inside a project scope: started work before unstarted, finished last.
#: (Build order across supernodes isn't otherwise encoded; between nodes it is — ``depends_on``.)
_SUPERNODE_ORDER = {"in-progress": 0, "planned": 1, DONE: 2}


class GraphError(ValueError):
    """The requested scope id doesn't exist, or isn't a project/supernode."""


def _refs(doc: MemoryDoc, field: str) -> list[str]:
    """Ids held by ``field`` — scalar or list, empty values dropped."""
    value = doc.frontmatter.extra.get(field)
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in items if str(item).strip()]


def _ref(doc: MemoryDoc, field: str) -> str | None:
    """The single value held by ``field`` (``None`` when unset/empty)."""
    ids = _refs(doc, field)
    return ids[0] if ids else None


def _expected_supernode_status(nodes: Iterable[MemoryDoc]) -> str:
    """The supernode status plan §3.3 derives from its nodes, in the rule's stated order.

    Node vocabulary in, supernode vocabulary out — they merely share the spelling of ``planned``.
    First match wins, which is what gives a supernode whose nodes are not written yet ``planned``
    (nothing has started) instead of a vacuous ``done``.
    """
    statuses = {node.frontmatter.status for node in nodes}
    if statuses <= {_UNSTARTED}:
        return "planned"
    if statuses <= {DONE, SUPERSEDED}:
        return DONE
    return "in-progress"


class Graph:
    """A read-only, indexed view of a workflow store.

    Built from a snapshot of the store's documents (see :meth:`memoryhub.hub.Hub.graph`);
    it does not observe later writes.
    """

    def __init__(self, docs: Iterable[MemoryDoc], profile: Profile | None = None) -> None:
        self.docs = list(docs)
        self.profile = profile
        self.by_id: dict[str, MemoryDoc] = {doc.id: doc for doc in self.docs}

    # --- lookups -------------------------------------------------------------------

    def _of_type(self, type_name: str) -> list[MemoryDoc]:
        return [doc for doc in self.docs if doc.type == type_name]

    def _typed(self, id: str, type_name: str) -> MemoryDoc | None:
        doc = self.by_id.get(id)
        return doc if doc is not None and doc.type == type_name else None

    def _scope_doc(self, scope_id: str) -> MemoryDoc:
        scope = self.by_id.get(scope_id)
        if scope is None:
            raise GraphError(f"no document with id {scope_id!r}")
        if scope.type not in (PROJECT, SUPERNODE):
            raise GraphError(
                f"scope {scope_id!r} is a {scope.type}; expected a project or supernode"
            )
        return scope

    # --- traversal -----------------------------------------------------------------

    def supernodes(self, scope_id: str, *, include_done: bool = True) -> list[MemoryDoc]:
        """The supernodes a scope covers, in walk order (a supernode scope is just itself)."""
        scope = self._scope_doc(scope_id)
        if scope.type == SUPERNODE:
            found = [scope]
        else:
            found = [doc for doc in self._of_type(SUPERNODE) if _ref(doc, PROJECT) == scope_id]
            found.sort(key=lambda d: (_SUPERNODE_ORDER.get(d.frontmatter.status, 9), d.id))
        if not include_done:
            found = [doc for doc in found if doc.frontmatter.status != DONE]
        return found

    def _supernode_nodes(self, supernode: MemoryDoc) -> list[MemoryDoc]:
        """Its nodes: the ``nodes`` list order first, then any unlisted claimant, id-sorted."""
        listed = [
            node
            for node in (self._typed(ref, NODE) for ref in _refs(supernode, "nodes"))
            if node is not None
        ]
        seen = {node.id for node in listed}
        unlisted = sorted(
            (
                node
                for node in self._of_type(NODE)
                if _ref(node, SUPERNODE) == supernode.id and node.id not in seen
            ),
            key=lambda node: node.id,
        )
        return [*listed, *unlisted]

    def nodes(self, scope_id: str, *, include_done_supernodes: bool = True) -> list[MemoryDoc]:
        """Every node in the scope, in walk order (supernode order, then node order)."""
        return [
            node
            for supernode in self.supernodes(scope_id, include_done=include_done_supernodes)
            for node in self._supernode_nodes(supernode)
        ]

    def subnodes(self, node: MemoryDoc) -> list[MemoryDoc]:
        """Its subnodes in history order — the node's ``subnodes`` list is the record."""
        return [
            sub
            for sub in (self._typed(ref, SUBNODE) for ref in _refs(node, "subnodes"))
            if sub is not None
        ]

    def deps_met(self, node: MemoryDoc) -> bool:
        """True when every ``depends_on`` id resolves to a node that is ``done``.

        ``superseded`` is deliberately not ``done``: a dependent must be rewired to the
        replacement node (:meth:`validate` flags such an edge).
        """
        for dep_id in _refs(node, "depends_on"):
            dep = self._typed(dep_id, NODE)
            if dep is None or dep.frontmatter.status != DONE:
                return False
        return True

    # --- queries -------------------------------------------------------------------

    def ready(
        self,
        scope_id: str,
        statuses: Sequence[str] = ACTIONABLE,
        *,
        include_blocked: bool = False,
    ) -> list[MemoryDoc]:
        """Nodes in the scope whose status is in ``statuses`` and whose dependencies are done.

        ``include_blocked`` drops the dependency test, which is what a *human* queue wants:
        ``needs-feedback`` and ``replan`` nodes are waiting on a person, and a person's decision
        is not blocked by unbuilt code. Without it such a node is invisible until its
        dependencies land — and a queue that silently hides items is worse than no queue.
        :meth:`next` deliberately has no such option: it is a claim, and a blocked node is not
        claimable by any machine role.
        """
        allowed = set(statuses)
        return [
            node
            for node in self.nodes(scope_id, include_done_supernodes=False)
            if node.frontmatter.status in allowed and (include_blocked or self.deps_met(node))
        ]

    def reads_for(self, node: MemoryDoc) -> list[str]:
        """Subnode ids the skill acting on ``node`` needs, newest per role (see STATUS_READS)."""
        roles = STATUS_READS.get(node.frontmatter.status, ())
        subnodes = self.subnodes(node)
        wanted: set[str] = set()
        for role in roles:
            newest = [sub for sub in subnodes if sub.frontmatter.extra.get("role") == role]
            if newest:
                wanted.add(newest[-1].id)
        return [sub.id for sub in subnodes if sub.id in wanted]

    def repository(self, node: MemoryDoc) -> str | None:
        """The repo folder to work in: the node's own, else its project's (cross-repo scopes)."""
        own = _ref(node, "repository")
        if own:
            return own
        supernode = self._typed(_ref(node, SUPERNODE) or "", SUPERNODE)
        if supernode is None:
            return None
        project = self._typed(_ref(supernode, PROJECT) or "", PROJECT)
        return _ref(project, "repository") if project is not None else None

    def brief(self, node: MemoryDoc) -> dict[str, Any]:
        """Everything a skill needs to act on ``node``, in one payload.

        The whole node (frontmatter + body, so acceptance criteria need no second call), where to
        run (``repository``), and ``reads``: the subnode ids to fetch.
        """
        supernode_id = _ref(node, SUPERNODE)
        supernode = self._typed(supernode_id or "", SUPERNODE)
        payload = flat_frontmatter(node.frontmatter, self.profile)
        payload["path"] = str(node.path) if node.path else None
        payload["body"] = node.body
        return {
            "node": payload,
            "supernode": supernode_id,
            "project": _ref(supernode, PROJECT) if supernode is not None else None,
            "repository": self.repository(node),
            "reads": self.reads_for(node),
        }

    def next(self, scope_id: str, statuses: Sequence[str] = ACTIONABLE) -> dict[str, Any] | None:
        """The first ready node in walk order, briefed for a skill — or ``None`` if idle."""
        ready = self.ready(scope_id, statuses)
        return self.brief(ready[0]) if ready else None

    def status(self, scope_id: str) -> dict[str, Any]:
        """Node counts by status, per supernode and for the scope as a whole."""
        scope = self._scope_doc(scope_id)
        ready_ids = {node.id for node in self.ready(scope_id)}
        supernodes = []
        totals: Counter[str] = Counter()
        for supernode in self.supernodes(scope_id):
            nodes = self._supernode_nodes(supernode)
            counts = Counter(node.frontmatter.status for node in nodes)
            totals.update(counts)
            supernodes.append(
                {
                    "id": supernode.id,
                    "title": supernode.frontmatter.title,
                    "status": supernode.frontmatter.status,
                    "total": len(nodes),
                    "ready": sum(1 for node in nodes if node.id in ready_ids),
                    "counts": dict(counts),
                }
            )
        return {
            "scope": scope.id,
            "type": scope.type,
            "title": scope.frontmatter.title,
            "status": scope.frontmatter.status,
            "supernodes": supernodes,
            "counts": dict(totals),
            "total": sum(totals.values()),
            "ready": len(ready_ids),
        }

    # --- validation ----------------------------------------------------------------

    def validate(self) -> StoreReport:
        """Check the graph invariants no per-file schema check can see (plan §6).

        **Issues** (they fail the report, hence the exit code): dangling references · dependency
        cycles · edges into ``superseded`` nodes · unfinished supersessions · node status vs the
        newest subnode's ``verdict`` · ``attempt`` vs the number of ``impl`` subnodes ·
        id/filename agreement · per-type status validity · per-type required fields · a work
        record for every started node · subnodes their node actually lists.

        **Warnings** (reported, never fatal): nodes their supernode does not list · a supernode
        status that does not follow from its nodes.
        """
        issues: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []
        for doc in self.docs:
            issues += self._check_filename(doc)
            issues += self._check_status(doc)
            issues += self._check_required(doc)
            issues += self._check_references(doc)
        for supernode in self._of_type(SUPERNODE):
            warnings += self._check_supernode_status(supernode)
        for node in self._of_type(NODE):
            issues += self._check_superseded_deps(node)
            issues += self._check_supersession(node)
            issues += self._check_history(node)
            warnings += self._check_membership(node)
        for sub in self._of_type(SUBNODE):
            issues += self._check_attachment(sub)
        issues += self._check_cycles()
        return StoreReport(issues=issues, warnings=warnings, checked=len(self.docs))

    def _check_filename(self, doc: MemoryDoc) -> list[ValidationIssue]:
        if doc.path is None or doc.path.stem == doc.id:
            return []
        reason = f"id {doc.id!r} does not match filename {doc.path.name}"
        return [ValidationIssue(doc.path, "id", reason)]

    def _check_status(self, doc: MemoryDoc) -> list[ValidationIssue]:
        allowed = TYPE_STATUSES.get(doc.type)
        status = doc.frontmatter.status
        if allowed is None or status in allowed:
            return []
        return [
            ValidationIssue(
                doc.path,
                "status",
                f"status {status!r} is not valid for type {doc.type!r} "
                f"(allowed: {', '.join(allowed)})",
            )
        ]

    def _check_required(self, doc: MemoryDoc) -> list[ValidationIssue]:
        """Fields the traversal reads must be present — see :data:`_REQUIRED`."""
        return [
            ValidationIssue(
                doc.path, field, f"a {doc.type} must set {field!r} (the graph reads it)"
            )
            for field in _REQUIRED.get(doc.type, ())
            if doc.frontmatter.extra.get(field) is None
        ]

    def _check_references(self, doc: MemoryDoc) -> list[ValidationIssue]:
        issues = []
        for field in _REFERENCES.get(doc.type, ()):
            for ref in _refs(doc, field):
                if ref not in self.by_id:
                    issues.append(ValidationIssue(doc.path, field, f"id {ref!r} does not resolve"))
        return issues

    def _check_superseded_deps(self, node: MemoryDoc) -> list[ValidationIssue]:
        """A superseded dependency can never become ``done`` — rewire it to the replacement."""
        issues = []
        for dep_id in _refs(node, "depends_on"):
            dep = self._typed(dep_id, NODE)
            if dep is not None and dep.frontmatter.status == SUPERSEDED:
                issues.append(
                    ValidationIssue(
                        node.path,
                        "depends_on",
                        f"depends on superseded node {dep_id!r} — rewire to its replacement",
                    )
                )
        return issues

    def _check_supersession(self, node: MemoryDoc) -> list[ValidationIssue]:
        """A replacement's ``supersedes`` target must actually be ``superseded``.

        This is the crash signature ``/replan`` opens: it writes the replacement first (so an
        interruption leaves a dangling id rather than a silently orphaned node) and retires the
        old node last. Between the two the store looks legal — a fresh ``planned`` node and an
        untouched original that the queue will hand out again. The forward pointer is what makes
        it detectable. The converse is *not* an error: a ``superseded`` node with no replacement
        is a cancellation (plan §11.16), and its own record says so.
        """
        target_id = _ref(node, "supersedes")
        target = self._typed(target_id or "", NODE)
        if target is None or target.frontmatter.status == SUPERSEDED:
            return []  # unresolvable/absent `supersedes` is already reported by the other checks
        return [
            ValidationIssue(
                node.path,
                "supersedes",
                f"supersedes {target.id!r}, whose status is "
                f"{target.frontmatter.status!r} — finish the supersession: append the record to "
                f"that node and set it {SUPERSEDED!r}",
            )
        ]

    def _check_attachment(self, sub: MemoryDoc) -> list[ValidationIssue]:
        """A subnode must appear in its node's ``subnodes`` list, or nothing can see it.

        This is the crash signature the write order predicts (plan §10: subnode first, node
        status last). The record exists and names its node, but the node never picked it up —
        so it is invisible to ``next``, to ``reads``, and to the status/verdict check. The
        repair is mechanical, hence the instruction in the message.
        """
        node = self._typed(_ref(sub, NODE) or "", NODE)
        if node is None or sub.id in _refs(node, "subnodes"):
            return []  # unresolvable/absent `node` is already reported by the other checks
        verdict = sub.frontmatter.extra.get("verdict")
        return [
            ValidationIssue(
                sub.path,
                NODE,
                f"node {node.id!r} does not list this subnode — an interrupted write; append it "
                f"to that node's `subnodes` and set the node's status to {verdict!r}",
            )
        ]

    def _check_membership(self, node: MemoryDoc) -> list[ValidationIssue]:
        """A node should appear in its supernode's ``nodes`` — a **warning**, not an issue.

        :meth:`_supernode_nodes` tolerates the omission by walking unlisted claimants last, which
        is what keeps a bulk migration workable while it writes nodes faster than it can order
        them. The cost is that ``nodes`` degrades from a build-order record into a hint, so the
        drift is reported rather than ignored — but as a warning, because a legitimate mid-write
        moment produces it and every skill's preflight gates on the exit code (plan §11.7).
        """
        supernode = self._typed(_ref(node, SUPERNODE) or "", SUPERNODE)
        if supernode is None or node.id in _refs(supernode, "nodes"):
            return []  # unresolvable/absent `supernode` is already reported by the other checks
        return [
            ValidationIssue(
                node.path,
                SUPERNODE,
                f"supernode {supernode.id!r} does not list this node — insert it into that "
                f"supernode's `nodes` at its build-order position (unlisted nodes are walked "
                f"last, so ordering silently stops meaning anything)",
            )
        ]

    def _check_supernode_status(self, supernode: MemoryDoc) -> list[ValidationIssue]:
        """A supernode's status should follow from its nodes — a **warning**, not an issue.

        Node status is cross-checked against its newest subnode's verdict (:meth:`_check_history`);
        supernode status had no such check, so ``hub graph status`` could report an ``in-progress``
        track whose nodes were all ``done``. Plan §3.3 assigns the value to "whichever agent
        completes/creates its nodes", which is precisely why this is a warning: between a node's
        transition and the container update the stored value is legitimately one write stale. It
        stays *stored* rather than derived for the same reason node status does — a stored value is
        greppable and visible in Obsidian.
        """
        if any(self._typed(ref, NODE) is None for ref in _refs(supernode, "nodes")):
            return []  # an entry that resolves to no node hides one, and is already reported
        nodes = self._supernode_nodes(supernode)
        status = supernode.frontmatter.status
        expected = _expected_supernode_status(nodes)
        if status == expected:
            return []
        return [
            ValidationIssue(
                supernode.path,
                "status",
                f"status {status!r} does not follow from its {len(nodes)} node(s) — expected "
                f"{expected!r}; set it (mid-loop it is legitimately one write behind, which is "
                f"why this is a warning)",
            )
        ]

    def _check_history(self, node: MemoryDoc) -> list[ValidationIssue]:
        """Node status must equal its newest subnode's verdict; ``attempt`` counts impl subnodes.

        A node past ``planned`` must also *have* a work record: every other status is one an
        agent reached by writing a subnode.
        """
        issues = []
        subnodes = self.subnodes(node)
        status = node.frontmatter.status
        if not subnodes and status != _UNSTARTED:
            issues.append(
                ValidationIssue(
                    node.path,
                    "subnodes",
                    f"a {status!r} node has no subnode — its work record is missing "
                    f"(only {_UNSTARTED!r} nodes may have none)",
                )
            )
        if subnodes:
            verdict = subnodes[-1].frontmatter.extra.get("verdict")
            if verdict != status:
                issues.append(
                    ValidationIssue(
                        node.path,
                        "status",
                        f"status {status!r} does not match the verdict "
                        f"{verdict!r} of its newest subnode {subnodes[-1].id!r}",
                    )
                )
        attempt = node.frontmatter.extra.get("attempt")
        impls = sum(1 for sub in subnodes if sub.frontmatter.extra.get("role") == "impl")
        if attempt is not None and attempt != impls:
            issues.append(
                ValidationIssue(
                    node.path,
                    "attempt",
                    f"attempt is {attempt!r} but the node has {impls} impl subnode(s)",
                )
            )
        return issues

    def _check_cycles(self) -> list[ValidationIssue]:
        """Report each ``depends_on`` cycle once, as the path that closes it."""
        issues: list[ValidationIssue] = []
        explored: set[str] = set()
        reported: set[frozenset[str]] = set()

        def walk(node_id: str, stack: list[str]) -> None:
            if node_id in stack:
                cycle = stack[stack.index(node_id) :]
                if frozenset(cycle) in reported:
                    return
                reported.add(frozenset(cycle))
                issues.append(
                    ValidationIssue(
                        self.by_id[node_id].path,
                        "depends_on",
                        "dependency cycle: " + " -> ".join([*cycle, node_id]),
                    )
                )
                return
            if node_id in explored:
                return
            stack.append(node_id)
            for dep_id in _refs(self.by_id[node_id], "depends_on"):
                if self._typed(dep_id, NODE) is not None:
                    walk(dep_id, stack)
            stack.pop()
            explored.add(node_id)

        for node in self._of_type(NODE):
            walk(node.id, [])
        return issues
