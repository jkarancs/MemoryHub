"""FastMCP (stdio) server exposing a guarded MemoryHub store to agents.

Read tools are always registered. Write tools exist only when ``allow_agent_writes = true`` in
``hub.toml`` — and even then they are guardrailed, not merely gated:

* ``add_memory`` forces ``status: draft`` / ``source: agent`` / ``visibility: private``; a human
  reviews drafts (``hub list --status draft``) and flips them to ``active``.
* ``update_memory`` refuses ``id``/``type``/``created``/``updated``/``visibility``/``source`` and
  may not set ``status`` to ``active`` — activating (and publishing) stays a human decision.
* the only delete is ``archive_memory``, a soft move into ``content_root/.trash/``.

``search_memory`` runs ``Hub.search`` (hybrid vector + fulltext) and degrades gracefully to
plain fulltext when the vector stack isn't installed or no index exists — same tool schema
either way. Start the server with ``hub mcp`` from anywhere inside a content repo (``hub.toml``
is resolved by walking up from the cwd), or register it in the repo's ``.mcp.json``.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from .config import load_config
from .hub import Hub
from .loader import flat_frontmatter
from .models import MemoryDoc
from .profiles import Profile
from .writer import WriteError, WriteWarning

_INSTRUCTIONS = """\
MemoryHub: a store of markdown memories with typed, validated frontmatter.

- Search before you add — the fact may already exist. Use search_memory, then get_memory for the
  full text; prefer update_memory over creating near-duplicates.
- Don't guess vocabulary: list_types() gives the closed type list plus each type's allowed extra
  fields; list_tags() gives the tags already in use. Reuse existing tags.
- Writes (when enabled) always produce drafts: add_memory forces status=draft and source=agent.
  A human reviews drafts and flips them to active — never try to activate or publish a memory.
"""

#: Keys an agent may never smuggle into ``add_memory``'s ``extra`` (engine- or human-owned, or
#: already covered by a dedicated argument).
_ADD_PROTECTED = frozenset(
    {
        "id",
        "type",
        "title",
        "description",
        "tags",
        "body",
        "status",
        "visibility",
        "source",
        "created",
        "updated",
    }
)
#: Keys ``update_memory`` refuses outright. ``status`` is special-cased: any value but "active".
_UPDATE_PROTECTED = frozenset({"id", "type", "created", "updated", "visibility", "source"})

_PREVIEW_CHARS = 240

_READ_ONLY = ToolAnnotations(readOnlyHint=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _summary(doc: MemoryDoc, profile: Profile) -> dict[str, Any]:
    """Flat, stably-ordered, JSON-safe frontmatter summary (no body)."""
    summary = _jsonable(flat_frontmatter(doc.frontmatter, profile))
    assert isinstance(summary, dict)
    return summary


def _preview(body: str) -> str:
    text = " ".join(body.split())
    return text if len(text) <= _PREVIEW_CHARS else text[: _PREVIEW_CHARS - 1] + "…"


@contextlib.contextmanager
def _collect_write_warnings(into: list[str]) -> Iterator[None]:
    """Capture :class:`WriteWarning` (e.g. unresolved ``related`` ids) for the tool response."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", WriteWarning)
        yield
    into.extend(str(w.message) for w in caught if issubclass(w.category, WriteWarning))


def _register_read_tools(server: FastMCP, hub: Hub) -> None:
    profile = hub.profile

    @server.tool(annotations=_READ_ONLY)
    def search_memory(
        query: Annotated[
            str,
            Field(
                description=(
                    "Search query — natural language works (semantic search over title, "
                    "description, body, and tags), exact keywords work too."
                )
            ),
        ],
        type: Annotated[
            str | None,
            Field(description="Restrict to one memory type (see list_types)."),
        ] = None,
        tags: Annotated[
            list[str] | None,
            Field(description="Restrict to memories carrying ALL of these tags (see list_tags)."),
        ] = None,
        status: Annotated[
            str | None,
            Field(description="Restrict by status: active, draft, archived, or aspirational."),
        ] = None,
        limit: Annotated[int, Field(description="Maximum number of results.", ge=1)] = 10,
    ) -> list[dict[str, Any]]:
        """Search the memory store, ranked by relevance (hybrid semantic + keyword search).

        Returns frontmatter summaries plus a short body_preview; call get_memory(id) for the
        full text. Examples: search_memory("asyncio"); search_memory("rag", type="skill",
        limit=5); search_memory("physics", tags=["career"]).
        """
        # Falls back to fulltext (with an IndexWarning on stderr) if no vector index is usable.
        docs = hub.search(query, mode="hybrid", limit=limit, type=type, tags=tags, status=status)
        results = []
        for doc in docs:
            item = _summary(doc, profile)
            item["body_preview"] = _preview(doc.body)
            results.append(item)
        return results

    @server.tool(annotations=_READ_ONLY)
    def get_memory(
        id: Annotated[str, Field(description="The memory id, e.g. 'skill-async-python'.")],
    ) -> dict[str, Any]:
        """Fetch a single memory by id: full frontmatter plus the complete markdown body.

        Use search_memory or list_memories to discover ids. Example: get_memory("bio-me").
        """
        try:
            doc = hub.get(id)
        except KeyError:
            raise ToolError(
                f"no memory with id {id!r} — find ids via search_memory or list_memories"
            ) from None
        item = _summary(doc, profile)
        item["body"] = doc.body
        item["path"] = str(doc.path) if doc.path else None
        return item

    @server.tool(annotations=_READ_ONLY)
    def list_memories(
        type: Annotated[
            str | None,
            Field(description="Restrict to one memory type (see list_types)."),
        ] = None,
        tags: Annotated[
            list[str] | None,
            Field(description="Restrict to memories carrying ALL of these tags (see list_tags)."),
        ] = None,
        status: Annotated[
            str | None,
            Field(description="Restrict by status: active, draft, archived, or aspirational."),
        ] = None,
        visibility: Annotated[
            str | None,
            Field(description="Restrict by visibility: public or private."),
        ] = None,
    ) -> list[dict[str, Any]]:
        """Browse frontmatter summaries (no body), filtered by frontmatter dimensions.

        Examples: list_memories(type="goal") for all goals; list_memories(status="draft") for
        items awaiting human review; list_memories(tags=["python", "async"]) for an AND-match.
        """
        docs = hub.filter(type=type, tags=tags, status=status, visibility=visibility)
        return [_summary(doc, profile) for doc in docs]

    @server.tool(annotations=_READ_ONLY)
    def list_types() -> dict[str, list[str]]:
        """The closed vocabulary of memory types, mapped to each type's allowed extra fields.

        Consult this before add_memory: 'type' must be one of these keys and every key in
        'extra' must come from that type's field list (e.g. skill allows proficiency, source).
        """
        return {name: profile.fields_for(name) for name in hub.list_types()}

    @server.tool(annotations=_READ_ONLY)
    def list_tags() -> dict[str, int]:
        """Tags already used in the store with usage counts, most-used first.

        Check this before tagging a new memory: reuse an existing spelling instead of inventing
        a near-duplicate (e.g. prefer an existing 'ml' over a new 'machine-learning').
        """
        return hub.list_tags()

    @server.tool(annotations=_READ_ONLY)
    def recall_bundle(
        task: Annotated[
            str,
            Field(
                description=(
                    "What you need context for, in natural language — used as the search query "
                    "and as the pack's header (e.g. 'prep for the NVIDIA interview loop')."
                )
            ),
        ],
        token_budget: Annotated[
            int,
            Field(description="Hard cap on the pack's size in tokens; it is never exceeded.", ge=1),
        ],
        type: Annotated[
            str | None,
            Field(description="Restrict to one memory type (see list_types)."),
        ] = None,
        tags: Annotated[
            list[str] | None,
            Field(description="Restrict to memories carrying ALL of these tags (see list_tags)."),
        ] = None,
    ) -> dict[str, Any]:
        """Build a task-scoped context pack that fits a token budget, richest memories first.

        Returns ready-to-paste markdown ('text') plus an inline manifest so one call is
        self-explaining: 'included' lists each memory with the level it was rendered at (full
        body / one-line description / title only) and its token cost; 'excluded' says what was
        dropped and why (budget / filter / floor). Prefer this over several get_memory calls when
        you need the most relevant context under a size limit. Example:
        recall_bundle("summarize my backend experience", 1500, type="experience").
        """
        result = hub.recall_bundle(task, token_budget, type=type, tags=tags)
        return {
            "text": result.text,
            "total_tokens": result.total_tokens,
            "budget": result.budget,
            "counter": result.counter,
            "included": [item.model_dump() for item in result.manifest],
            "excluded": [item.model_dump() for item in result.excluded],
        }

    @server.tool(annotations=_READ_ONLY)
    def context_stats(
        last_n: Annotated[
            int,
            Field(description="How many recent bundle calls to summarize (<=0 for all)."),
        ] = 20,
    ) -> dict[str, Any]:
        """Recent recall_bundle calls plus aggregates: what context cost per task, over time.

        Returns the last N logged bundles ('entries': task, budget, tokens used, how many
        memories were included vs excluded, which token counter) and 'aggregates' (calls, average
        tokens per task, average memories included, overall inclusion rate). Use it to see whether
        budgets are too tight (low inclusion rate) or generous. Example: context_stats(50).
        """
        return hub.context_stats(last_n)


def _register_write_tools(server: FastMCP, hub: Hub) -> None:
    profile = hub.profile

    @server.tool(annotations=ToolAnnotations(destructiveHint=False, idempotentHint=False))
    def add_memory(
        type: Annotated[
            str,
            Field(description="Memory type from the closed vocabulary — call list_types() first."),
        ],
        title: Annotated[
            str,
            Field(description="Short human-readable title; the id slug is generated from it."),
        ],
        description: Annotated[
            str,
            Field(description="One-sentence summary shown in listings and search results."),
        ],
        body: Annotated[str, Field(description="Markdown body with the full detail.")],
        tags: Annotated[
            list[str] | None,
            Field(description="Tags for retrieval; reuse spellings from list_tags()."),
        ] = None,
        extra: Annotated[
            dict[str, Any] | None,
            Field(
                description=(
                    "Type-specific fields allowed for this type per list_types(), e.g. "
                    "{'proficiency': 'advanced'} for a skill; may also carry 'related' "
                    "(a list of memory ids)."
                )
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Create a new memory as a draft for human review; returns it with the generated id.

        Guardrails: status is forced to 'draft', source to 'agent', and visibility to 'private'
        — you cannot set those; a human reviews and activates the draft. Search first
        (search_memory) to avoid duplicates. Example: add_memory(type="skill",
        title="RAG pipelines", description="Retrieval-augmented generation systems",
        body="...", tags=["ml", "rag"], extra={"proficiency": "intermediate"}).
        """
        supplied = dict(extra or {})
        forbidden = sorted(_ADD_PROTECTED & supplied.keys())
        if forbidden:
            raise ToolError(
                f"extra may not set {', '.join(forbidden)} — those fields are engine- or "
                "human-owned (agent memories start as private drafts; a human activates them)"
            )
        fields: dict[str, Any] = dict(supplied)
        fields.update(
            type=type,
            title=title,
            description=description,
            tags=list(tags or []),
            status="draft",
            visibility="private",
            source="agent",
        )
        caught: list[str] = []
        try:
            with _collect_write_warnings(caught):
                doc = hub.add(**fields, body=body)
        except WriteError as exc:
            raise ToolError(str(exc)) from exc
        result = _summary(doc, profile)
        if caught:
            result["warnings"] = caught
        return result

    @server.tool(annotations=ToolAnnotations(destructiveHint=True))
    def update_memory(
        id: Annotated[str, Field(description="Id of the memory to update.")],
        fields: Annotated[
            dict[str, Any] | None,
            Field(
                description=(
                    "Frontmatter patch: common fields (title, description, tags, related) and "
                    "type-specific fields; set a type-specific field to null to remove it."
                )
            ),
        ] = None,
        body: Annotated[
            str | None,
            Field(description="Replacement markdown body (omit to keep the current one)."),
        ] = None,
    ) -> dict[str, Any]:
        """Patch an existing memory's frontmatter and/or replace its body.

        Guardrails: id, type, created, updated, visibility, and source cannot change, and
        status may not be set to 'active' — activating (and publishing) a memory is the human
        reviewer's job. 'updated' is bumped automatically. Example:
        update_memory("skill-rag", fields={"proficiency": "advanced", "tags": ["ml", "rag"]}).
        """
        patch = dict(fields or {})
        if not patch and body is None:
            raise ToolError("nothing to update: pass fields and/or body")
        if "body" in patch:
            raise ToolError("pass the body via the dedicated 'body' argument, not fields")
        refused = sorted(_UPDATE_PROTECTED & patch.keys())
        if refused:
            raise ToolError(
                f"update_memory may not change {', '.join(refused)} — those fields are engine- "
                "or human-owned"
            )
        if patch.get("status") == "active":
            raise ToolError(
                "agents may not set status to 'active' — a human reviews and activates drafts"
            )
        if body is not None:
            patch["body"] = body
        caught: list[str] = []
        try:
            with _collect_write_warnings(caught):
                doc = hub.update(id, **patch)
        except WriteError as exc:
            raise ToolError(str(exc)) from exc
        result = _summary(doc, profile)
        if caught:
            result["warnings"] = caught
        return result

    @server.tool(annotations=ToolAnnotations(destructiveHint=True, idempotentHint=False))
    def archive_memory(
        id: Annotated[str, Field(description="Id of the memory to archive.")],
    ) -> dict[str, Any]:
        """Soft-delete a memory: its file moves to the store's .trash/ folder.

        A human can restore it from there; there is no hard-delete tool. To keep a memory in
        the store but mark it inactive, use update_memory(id, fields={"status": "archived"})
        instead. Example: archive_memory("skill-outdated").
        """
        try:
            hub.delete(id)
        except WriteError as exc:
            raise ToolError(str(exc)) from exc
        return {
            "id": id,
            "archived": True,
            "note": "file moved to .trash/ inside the store; a human can restore it",
        }


def build_server(hub: Hub) -> FastMCP:
    """Build the FastMCP server over ``hub``; write tools only exist if the config allows them."""
    server = FastMCP(name="memoryhub", instructions=_INSTRUCTIONS)
    _register_read_tools(server, hub)
    if hub.config.write.allow_agent_writes:
        _register_write_tools(server, hub)
    return server


def main() -> None:
    """Start a stdio MCP server over the store whose ``hub.toml`` is at or above the cwd."""
    build_server(Hub(load_config(Path.cwd()))).run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
