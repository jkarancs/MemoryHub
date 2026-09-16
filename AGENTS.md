# AGENTS.md

> **Project:** MemoryHub — reusable, content-agnostic engine over a markdown memory store; content lives in separate repos (`personal-memory`, future `work-memory`) that depend on this package.
> **Core constraints:** Never commit memory content or personal data here — this repo is the engine only. `Hub` (`src/memoryhub/hub.py`) is the only public entry point: CLI, MCP server, and search all go through it. Writes validate before touching disk, write atomically, and obey the content repo's `hub.toml` write policy.

## Commands
| Intent | Command | Authority |
|---|---|---|
| Setup | `uv pip install --python .venv/bin/python -e ".[dev,mcp,vectors]"` | CI-equivalent install — see `.github/workflows/ci.yml` |
| Test — all | `uv run --no-sync pytest` | pytest — see `pyproject.toml`; deselects `-m local` |
| Test — single | `uv run --no-sync pytest tests/test_writer.py -k <expr>` | |
| Test — golden eval | `uv run --no-sync pytest -m local` | needs GPU/model + `../personal-memory` corpus — never in CI |
| Lint | `uv run --no-sync ruff check .` | ruff — see `pyproject.toml [tool.ruff]` |
| Format | `uv run --no-sync black --check .` | black — see `pyproject.toml [tool.black]` |
| Type check | `uv run --no-sync mypy` | `pyproject.toml [tool.mypy]` |
| Schema export | `uv run --no-sync hub schema export` | writes `schema/frontmatter.schema.json` |
| MCP server | `hub mcp` | run from inside a content repo — `hub.toml` resolves upward from cwd |

## Conventions
- Work tracking: the **development graph** in `/home/jkarancsi/git/agent-memory` — MemoryHub nodes live under `memoryhub-career` (Phase 6 acceptance), `personalsite-build` (store-side work), `workspace-graph-workflow` (the `hub graph` engine), and the `ai-roadmap-*` arcs. `hub graph status <scope>` is the live status; take work with `/implement`, start new work with `/create-plan`. The workspace specs (`../specs/ai-roadmap/`, `../specs/memoryhub/`) stay the acceptance contracts, but their status tables are a frozen archive — never read one to find work and never write a status into it · Branch: `main`, direct commits · Commit: one-line summary, no body — on the user's behalf; amend-on-fix ok
- Done means: `uv run --no-sync pytest` passes; keep core coverage ≥90% (repo convention, not CI-enforced).

## Judgment Boundaries
**NEVER**
- Commit memory content or personal data to this repo — content belongs in the content repos.
- Commit secrets, tokens, or credentials; never copy values from env files or keyrings into any file.

**PROCEED**
- Add an external dependency when the node's plan needs one — record the addition and the reason in the node's `-impl` handoff.
- Changing the public API surface (`src/memoryhub/__init__.py`) or a schema profile's type vocabulary: proceed only if the node's plan authorizes it — the content repos and `hub graph` consume both. Otherwise mark the node `needs-feedback` with the proposed shape and the consumers it breaks.

**ALWAYS**
- Read `.ai/CONTEXT.md` (Knowledge Layer, below) before exploring source or editing files.
- For multi-step work: state a short plan with a verifiable check per step, then run `uv run --no-sync pytest` before declaring done.
- Keep diffs minimal — touch only what the task requires; match existing style.

Workspace layer: [../CLAUDE.md](../CLAUDE.md) — cross-repo conventions (uv-managed venvs, commit rules, personal-memory protocol). Claude Code auto-loads it; the load-bearing rules for other agents are inlined above. · Project standards — indexed in CONTEXT.md.

## Knowledge Layer
**Before exploring source or editing anything, read [.ai/CONTEXT.md](.ai/CONTEXT.md).** It is the complete index of this repo's knowledge — structure map, key flows, domain terms, and the rules for adding knowledge. Nothing in `.ai/` is routed from here; the full index is one file away.
Read the smallest file that answers the task; stop when the next action is clear; capture reusable findings per the governance rules indexed there.

@RTK.md
