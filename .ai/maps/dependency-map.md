---
description: MemoryHub dependency map — engine ↔ content-repo relations and optional-extra → capability mapping.
status: reviewed
---
# Dependency Map

> Parent: [../CONTEXT.md](../CONTEXT.md)

## Cross-repo relations
| Edge | Evidence |
|---|---|
| `../personal-memory` (content repo) → depends on this engine; its `hub.toml` configures the store | `../personal-memory/hub.toml` header names the MemoryHub engine |
| Workspace agents → reach the store through the `memoryhub` MCP server (`hub mcp`) served by this engine | `src/memoryhub/mcp_server.py · main`; server registration is workspace-level (`../.mcp.json`) |
| Public export target → any destination repo passed to `hub export`; becomes a read-only hub store | `src/memoryhub/export.py · export_store` |

## Optional extras → capabilities (authority: `pyproject.toml [project.optional-dependencies]`)
| Extra | Enables |
|---|---|
| `vectors` | LanceDB index — serve/build vector search |
| `local-embed` / `api-embed` | embedding backend (sentence-transformers / OpenAI-compatible API) — required to build indexes or run vector queries |
| `mcp` | `hub mcp` stdio server |
| `tokens` | exact token counting for bundles (else `len//4` heuristic, recorded in the bundle manifest) |

Missing extras degrade, never crash: search falls back to fulltext with a warning (`src/memoryhub/hub.py · Hub._usable_index`).
