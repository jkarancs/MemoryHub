"""MCP server tests via the SDK's in-memory client (no subprocess, no network, no GPU).

Covers: tool registration + write gating, each tool's happy path, the agent-write guardrails
(draft/source/visibility forcing, protected fields, no activation), and soft archive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

from mcp.shared.memory import (  # noqa: E402
    create_connected_server_and_client_session as client_session,
)
from mcp.types import CallToolResult  # noqa: E402

from memoryhub import Hub, load_config  # noqa: E402
from memoryhub.mcp_server import build_server  # noqa: E402

pytestmark = pytest.mark.anyio

READ_TOOLS = {"search_memory", "get_memory", "list_memories", "list_types", "list_tags"}
WRITE_TOOLS = {"add_memory", "update_memory", "archive_memory"}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def server(seeded_repo: Path) -> Any:
    return build_server(Hub(load_config(seeded_repo)))


async def call(server: Any, tool: str, args: dict[str, Any] | None = None) -> CallToolResult:
    async with client_session(server._mcp_server) as client:
        return await client.call_tool(tool, args or {})


def unwrap(result: CallToolResult) -> Any:
    """The tool's return value from structured content (lists arrive wrapped in 'result')."""
    assert not result.isError, result.content
    payload = result.structuredContent
    assert payload is not None
    if isinstance(payload, dict) and set(payload) == {"result"}:
        return payload["result"]
    return payload


def error_text(result: CallToolResult) -> str:
    assert result.isError
    text = result.content[0].text  # type: ignore[union-attr]
    assert isinstance(text, str)
    return text


# --- registration / gating ---------------------------------------------------------


async def test_all_tools_registered_when_writes_allowed(server: Any) -> None:
    async with client_session(server._mcp_server) as client:
        tools = {tool.name for tool in (await client.list_tools()).tools}
    assert tools == READ_TOOLS | WRITE_TOOLS


async def test_write_tools_absent_when_agent_writes_disabled(seeded_repo: Path) -> None:
    toml_path = seeded_repo / "hub.toml"
    toml_path.write_text(
        toml_path.read_text(encoding="utf-8").replace(
            "allow_agent_writes = true", "allow_agent_writes = false"
        ),
        encoding="utf-8",
    )
    server = build_server(Hub(load_config(seeded_repo)))
    async with client_session(server._mcp_server) as client:
        tools = {tool.name for tool in (await client.list_tools()).tools}
    assert tools == READ_TOOLS


# --- read tools ----------------------------------------------------------------------


async def test_search_memory_finds_and_ranks(server: Any) -> None:
    items = unwrap(await call(server, "search_memory", {"query": "async"}))
    ids = [item["id"] for item in items]
    assert ids[0] == "skill-async-python"
    assert "project-hub" in ids
    assert all("body" not in item for item in items)
    assert "asyncio" in items[0]["body_preview"]


async def test_search_memory_filters_and_limit(server: Any) -> None:
    items = unwrap(await call(server, "search_memory", {"query": "async", "type": "project"}))
    assert [item["id"] for item in items] == ["project-hub"]
    items = unwrap(await call(server, "search_memory", {"query": "async", "limit": 1}))
    assert len(items) == 1
    items = unwrap(await call(server, "search_memory", {"query": "no-such-term-anywhere"}))
    assert items == []


async def test_get_memory_full_document(server: Any) -> None:
    item = unwrap(await call(server, "get_memory", {"id": "skill-async-python"}))
    assert item["title"] == "Async Python"
    assert item["proficiency"] == "advanced"
    assert item["created"] == "2026-01-01"
    assert "asyncio task groups" in item["body"]
    assert item["path"] and item["path"].endswith("skill-async-python.md")


async def test_get_memory_unknown_id_errors(server: Any) -> None:
    result = await call(server, "get_memory", {"id": "nope"})
    assert "no memory with id 'nope'" in error_text(result)


async def test_list_memories_summaries_only(server: Any) -> None:
    items = unwrap(await call(server, "list_memories", {"type": "skill"}))
    assert {item["id"] for item in items} == {"skill-async-python", "skill-sql", "skill-docker"}
    assert all("body" not in item for item in items)
    drafts = unwrap(await call(server, "list_memories", {"status": "draft"}))
    assert [item["id"] for item in drafts] == ["goal-ai-role"]


async def test_list_types_maps_to_extra_fields(server: Any) -> None:
    types = unwrap(await call(server, "list_types"))
    assert set(types) == {
        "bio",
        "skill",
        "experience",
        "education",
        "project",
        "goal",
        "preference",
        "writing",
    }
    assert types["skill"] == ["proficiency", "source"]
    assert types["bio"] == []


async def test_list_tags_counts(server: Any) -> None:
    tags = unwrap(await call(server, "list_tags"))
    assert tags["python"] == 4
    assert list(tags)[0] == "python"


# --- add_memory guardrails ------------------------------------------------------------


async def test_add_memory_forces_draft_agent_private(server: Any, seeded_repo: Path) -> None:
    result = unwrap(
        await call(
            server,
            "add_memory",
            {
                "type": "skill",
                "title": "Vector Databases",
                "description": "LanceDB and vector index tooling",
                "body": "Hands-on with LanceDB upserts and ANN queries.",
                "tags": ["ml"],
                "extra": {"proficiency": "beginner"},
            },
        )
    )
    assert result["id"] == "skill-vector-databases"
    assert result["status"] == "draft"
    assert result["source"] == "agent"
    assert result["visibility"] == "private"

    path = seeded_repo / "memory" / "skill" / "skill-vector-databases.md"
    assert path.exists()
    report = Hub(load_config(seeded_repo)).validate()
    assert report.ok, report.to_dict()


@pytest.mark.parametrize("field", ["status", "visibility", "source", "id", "created"])
async def test_add_memory_rejects_protected_extra(
    server: Any, seeded_repo: Path, field: str
) -> None:
    before = sorted((seeded_repo / "memory").rglob("*.md"))
    result = await call(
        server,
        "add_memory",
        {
            "type": "skill",
            "title": "Sneaky",
            "description": "d",
            "body": "b",
            "extra": {field: "active"},
        },
    )
    assert f"extra may not set {field}" in error_text(result)
    assert sorted((seeded_repo / "memory").rglob("*.md")) == before


async def test_add_memory_reports_unresolved_related_as_warning(server: Any) -> None:
    result = unwrap(
        await call(
            server,
            "add_memory",
            {
                "type": "goal",
                "title": "Learn CUDA",
                "description": "GPU programming goal",
                "body": "Study plan TBD.",
                "extra": {"related": ["skill-does-not-exist"]},
            },
        )
    )
    assert result["status"] == "draft"
    assert any("skill-does-not-exist" in warning for warning in result["warnings"])


async def test_add_memory_invalid_type_errors(server: Any) -> None:
    result = await call(
        server,
        "add_memory",
        {"type": "nonsense", "title": "t", "description": "d", "body": "b"},
    )
    assert "'type'" in error_text(result)


# --- update_memory guardrails -----------------------------------------------------------


async def test_update_memory_patches_fields_and_body(server: Any) -> None:
    result = unwrap(
        await call(
            server,
            "update_memory",
            {
                "id": "skill-sql",
                "fields": {"proficiency": "advanced", "description": "Window functions et al."},
                "body": "New body about SQL.",
            },
        )
    )
    assert result["proficiency"] == "advanced"
    assert result["description"] == "Window functions et al."
    item = unwrap(await call(server, "get_memory", {"id": "skill-sql"}))
    assert item["body"] == "New body about SQL.\n"
    assert item["updated"] >= item["created"]


async def test_update_memory_cannot_activate(server: Any) -> None:
    result = await call(
        server, "update_memory", {"id": "goal-ai-role", "fields": {"status": "active"}}
    )
    assert "may not set status to 'active'" in error_text(result)
    item = unwrap(await call(server, "get_memory", {"id": "goal-ai-role"}))
    assert item["status"] == "draft"


async def test_update_memory_allows_other_status_changes(server: Any) -> None:
    result = unwrap(
        await call(
            server, "update_memory", {"id": "goal-ai-role", "fields": {"status": "aspirational"}}
        )
    )
    assert result["status"] == "aspirational"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "goal-renamed"),
        ("type", "skill"),
        ("created", "2020-01-01"),
        ("updated", "2020-01-01"),
        ("visibility", "public"),
        ("source", "self"),
    ],
)
async def test_update_memory_refuses_protected_fields(server: Any, field: str, value: str) -> None:
    result = await call(server, "update_memory", {"id": "goal-ai-role", "fields": {field: value}})
    assert f"may not change {field}" in error_text(result)


async def test_update_memory_requires_a_change(server: Any) -> None:
    result = await call(server, "update_memory", {"id": "goal-ai-role"})
    assert "nothing to update" in error_text(result)


async def test_update_memory_body_belongs_in_dedicated_argument(server: Any) -> None:
    result = await call(
        server, "update_memory", {"id": "goal-ai-role", "fields": {"body": "smuggled"}}
    )
    assert "dedicated 'body' argument" in error_text(result)


# --- archive_memory ---------------------------------------------------------------------


async def test_archive_memory_is_soft(server: Any, seeded_repo: Path) -> None:
    result = unwrap(await call(server, "archive_memory", {"id": "skill-docker"}))
    assert result["archived"] is True
    assert not (seeded_repo / "memory" / "skill" / "skill-docker.md").exists()
    assert (seeded_repo / "memory" / ".trash" / "skill-docker.md").exists()
    remaining = unwrap(await call(server, "list_memories", {"type": "skill"}))
    assert "skill-docker" not in {item["id"] for item in remaining}


async def test_archive_memory_unknown_id_errors(server: Any) -> None:
    result = await call(server, "archive_memory", {"id": "nope"})
    assert "no memory with id 'nope'" in error_text(result)


# --- `hub mcp` CLI wiring ----------------------------------------------------------------


def test_cli_mcp_command_builds_and_runs_stdio_server(
    seeded_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from memoryhub.cli import app

    seen: dict[str, Any] = {}

    class FakeServer:
        def run(self, transport: str) -> None:
            seen["transport"] = transport

    def fake_build_server(hub: Hub) -> FakeServer:
        seen["content_root"] = hub.config.content_root
        return FakeServer()

    monkeypatch.chdir(seeded_repo)
    monkeypatch.setattr("memoryhub.mcp_server.build_server", fake_build_server)
    result = CliRunner().invoke(app, ["mcp"])
    assert result.exit_code == 0, result.output
    assert seen == {"content_root": seeded_repo / "memory", "transport": "stdio"}
