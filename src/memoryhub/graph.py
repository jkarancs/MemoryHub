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
  * :meth:`Graph.validate` — the graph invariants a per-file schema check can't see.
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

#: Which statuses are legal for which type. A profile declares one store-wide ``status`` enum
#: (the union of all four vocabularies), so "is this status legal for *this* type?" lands here.
TYPE_STATUSES: dict[str, tuple[str, ...]] = {
    PROJECT: ("active", "paused", DONE),
    SUPERNODE: ("planned", "in-progress", DONE),
    NODE: (*ACTIONABLE, DONE, SUPERSEDED),
    SUBNODE: (DONE,),  # an immutable record; its `verdict` is the payload
}

#: Subnode roles the acting skill needs to read for a node in each status (plan §4.2/§7).
#: :meth:`Graph.next` resolves each to that role's newest subnode, so one call tells a skill
#: exactly which documents to ``hub get``.
STATUS_READS: dict[str, tuple[str, ...]] = {
    "planned": ("plan",),
    "rejected": ("plan", "test"),
    "needs-fix": ("plan", "fdbk"),
    "implemented": ("plan", "impl", "fix"),
    "needs-feedback": ("plan", "test"),
    "replan": ("plan", "test", "fdbk"),
}

#: Frontmatter fields whose values are ids that must resolve to a document in the store.
_REFERENCES: dict[str, tuple[str, ...]] = {
    SUPERNODE: ("project", "nodes"),
    NODE: ("supernode", "depends_on", "subnodes", "supersedes"),
    SUBNODE: ("node",),
}

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

    def ready(self, scope_id: str, statuses: Sequence[str] = ACTIONABLE) -> list[MemoryDoc]:
        """Nodes in the scope whose status is in ``statuses`` and whose dependencies are done."""
        allowed = set(statuses)
        return [
            node
            for node in self.nodes(scope_id, include_done_supernodes=False)
            if node.frontmatter.status in allowed and self.deps_met(node)
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

    def next(self, scope_id: str, statuses: Sequence[str] = ACTIONABLE) -> dict[str, Any] | None:
        """The first ready node in walk order, packaged for a skill — or ``None`` if idle.

        The payload carries the whole node (frontmatter + body, so acceptance criteria need no
        second call), where to run (``repository``), and ``reads``: the subnode ids to fetch.
        """
        ready = self.ready(scope_id, statuses)
        if not ready:
            return None
        node = ready[0]
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

        Dangling references · dependency cycles · edges into ``superseded`` nodes · node status
        vs the newest subnode's ``verdict`` · ``attempt`` vs the number of ``impl`` subnodes ·
        id/filename agreement · per-type status validity.
        """
        issues: list[ValidationIssue] = []
        for doc in self.docs:
            issues += self._check_filename(doc)
            issues += self._check_status(doc)
            issues += self._check_references(doc)
        for node in self._of_type(NODE):
            issues += self._check_superseded_deps(node)
            issues += self._check_history(node)
        issues += self._check_cycles()
        return StoreReport(issues=issues, warnings=[], checked=len(self.docs))

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

    def _check_history(self, node: MemoryDoc) -> list[ValidationIssue]:
        """Node status must equal its newest subnode's verdict; ``attempt`` counts impl subnodes."""
        issues = []
        subnodes = self.subnodes(node)
        if subnodes:
            verdict = subnodes[-1].frontmatter.extra.get("verdict")
            if verdict != node.frontmatter.status:
                issues.append(
                    ValidationIssue(
                        node.path,
                        "status",
                        f"status {node.frontmatter.status!r} does not match the verdict "
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
