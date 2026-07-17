---
description: Guarded write flow — add/update/delete a memory via CLI, MCP, or Hub; validation, atomicity, and agent guardrails.
status: reviewed
---
# Guarded Write Flow

> Parent: [README.md](README.md)

**Trigger:** `hub new/add/update/rm`, MCP `add_memory`/`update_memory`/`archive_memory`, or `Hub.add/update/delete` → **Outcome:** a validated, atomically written (or trashed) markdown file in the content store.

| # | Step | Code | Rule enforced |
|---|---|---|---|
| 1 | MCP write tools are registered only when `write.allow_agent_writes = true` in `hub.toml` | `src/memoryhub/mcp_server.py · build_server` | agent writes are opt-in per store |
| 2 | `add_memory` rejects protected keys in `extra` and forces `status=draft`, `source=agent`, `visibility=private` | `src/memoryhub/mcp_server.py · add_memory` | agent memories start as private drafts |
| 3 | `update_memory` refuses `id`/`type`/`created`/`updated`/`visibility`/`source`, and any `status: active` | `src/memoryhub/mcp_server.py · update_memory` | activating/publishing is a human decision |
| 4 | Policy check: refuse when `allow_agent_writes` is false; `require_confirmation` needs a registered confirmer | `src/memoryhub/writer.py · _check_policy` | — |
| 5 | Frontmatter validated against the Pydantic model + active profile **before** any disk I/O | `src/memoryhub/writer.py · _validate_or_raise` | validate-before-disk: on failure, no file changes |
| 6 | Id resolution: explicit duplicate id hard-fails; generated slug gets a `-2`/`-3` suffix to uniqueness | `src/memoryhub/writer.py · add` · `src/memoryhub/ids.py · ensure_unique` | ids are slugs, unique across the store |
| 7 | Path confined under `content_root`; write via same-directory temp file + `os.replace` | `src/memoryhub/writer.py · _confine / _atomic_write` | no traversal; no partial writes |
| 8 | Delete is a soft move into `content_root/.trash/` | `src/memoryhub/writer.py · delete` | no hard delete anywhere in the engine |

## Failure paths
- `WriteError` (policy refusal, invalid frontmatter, duplicate id) → surfaced to agents as MCP `ToolError` (`src/memoryhub/mcp_server.py · add_memory / update_memory`).
- Unresolved `related` ids **warn** (`WriteWarning`), never fail; MCP returns them in the tool response (`src/memoryhub/writer.py · _warn_unresolved_related`, `src/memoryhub/mcp_server.py · _collect_write_warnings`).

## Invariants
- `update` may not change `id` or `type` — a rename/move is not a single-file write (`src/memoryhub/writer.py · update`).
- `add` defaults `status=draft`, `visibility=private` for every caller, not just agents (`src/memoryhub/writer.py · add`).
- Every write invalidates the Hub's in-memory doc cache (`src/memoryhub/hub.py · Hub.add/update/delete`).
