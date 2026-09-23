# Showcase audit — MemoryHub

The stranger test (roadmap P26, tasks 1, 2 and 7): clone the repo into an empty directory, give
it a clean environment (`env -i`, a fresh `HOME`, no pre-built venv), and follow the README
**literally**. Anything that only works because a maintainer knows something the README does
not say is a finding.

- **Audited:** 2026-09-23, from commit `93e50e0` (local `main`).
- **Environment:** Ubuntu 24.04 on WSL2; system Python 3.12.3 with PEP 668 enabled; uv-managed
  CPython 3.11.15, 3.12.13 and 3.13.14.
- **Also checked:** the public GitHub repo (`jkarancs/MemoryHub`) and its Actions history.

## Inventory

What the roadmap and specs claim for this repo, against what exists.

| Claimed | Exists |
|---|---|
| Content-agnostic engine; one `Hub` facade behind CLI, MCP, and search | yes — `src/memoryhub/hub.py`; CLI and MCP server both go through it |
| Guarded, atomic, validated writes; agent writes land as drafts | yes — `writer.py`; MCP `add_memory` forces `status: draft` |
| Hybrid search over LanceDB + golden-query eval | yes — `index.py`, `tests/eval_queries.yaml` (`-m local`, not in CI) |
| Token-budgeted context bundles (`recall_bundle`) | yes — `bundle.py`, `hub bundle`, MCP `recall_bundle` / `context_stats` |
| Deterministic public export | yes — `export.py`, `hub export` |
| Workflow dependency graph (`hub graph`) | yes — `graph.py`, six subcommands |
| CI gate (lint, format, types, tests on 3.11–3.13) | configured — but **red on GitHub since 2026-07-08** (finding F1) |
| MIT licence | declared in `pyproject.toml` and README — **no licence file** (F5) |

## Matrix — before

| # | Check | Result | Evidence |
|---|---|---|---|
| C1 | README install works as written | **fail** | no clone step; `pip install -e .` is refused on a PEP 668 system Python (F3) |
| C2 | `uv sync; uv run pytest` (P26 acceptance command) | **fail** | `Failed to spawn: pytest` — test tools live in an extra `uv sync` does not install (F2) |
| C3 | Repo verification green from a clean install | **fail** | `mcp` resolves to 2.2.0: `tests/test_mcp_server.py` fails to collect, mypy errors (F1); `black --check` fails (F4); mypy fails on 3.12+ (F6) and without the `vectors` extra (F7) |
| C4 | Quick start works for a stranger | **fail** | points at `path/to/content-repo/`; no store ships with the repo (F8) |
| C5 | Licence present | **fail** | README says MIT; no `LICENSE` file (F5) |
| C6 | No secrets | pass | tree and full history scanned for key shapes; only the deliberate fakes in `tests/test_export.py` |
| C7 | No personal data | pass | no names, emails or content outside agent-tooling paths (see W2) |
| C8 | No dead code | pass | `vulture --min-confidence 60`: every hit is a framework registration (Typer command, MCP tool, Pydantic validator) or public API |
| C9 | No stale TODOs | pass | two hits, both intentional: the notebook's exercise stub and the body `hub new` scaffolds |
| C10 | Docs match the CLI surface | **fail** | README omits `bundle`, `relayout`, `export`, `graph`; layout omits `bundle.py`, `tokens.py`, `export.py` (F9) |
| C11 | Package metadata | **fail** | project URLs are `https://example.com/MemoryHub` placeholders (F10) |
| C12 | 60-second demo | **fail** | none (F8) |
| C13 | CI face: badge, visible green gate, gate documented | **fail** | no badge; last three `main` runs red; gate undocumented (F1, F11) |

## Fix list

| # | Finding | Fix |
|---|---|---|
| F1 | `mcp` is unpinned. mcp 2.x renamed `FastMCP` to `MCPServer` and moved `mcp.server.fastmcp`, so every fresh install — including GitHub CI since 2026-07-08 — fails. Local runs stayed green only because a warm venv had mcp 1.28. | Pin `mcp>=1.14,<2`. The floor was bisected: 1.12 and 1.13 fail the MCP tests; 1.14 and 1.15 pass them and mypy. Porting to mcp 2 is a backlog item, not audit work. |
| F2 | `uv sync; uv run pytest` cannot work: pytest and friends are an optional extra. | Add a PEP 735 `dev` dependency group, `memoryhub[dev,mcp,vectors]` — the same set CI installs — which `uv sync` installs by default. |
| F3 | Install section had no clone step and a bare `pip install`. | README install now: clone, `uv sync`; or a venv, then `pip`. |
| F4 | `tests/test_writer.py` was committed unformatted, so `black --check` failed. | `black tests/test_writer.py`. |
| F5 | No licence file. | Add `LICENSE` (MIT, "The MemoryHub authors"). |
| F6 | `[tool.mypy] python_version = "3.11"` makes mypy parse newer numpy stubs as 3.11 and fail on their 3.12-only `type` statement. | Drop the pin: mypy checks at the running interpreter's version, and CI's 3.11 leg still type-checks 3.11. |
| F7 | mypy fails when `lancedb` is absent (`pip install -e ".[dev,mcp]"`): the override covered `lancedb` but not `lancedb.background_loop`. | Add `lancedb.*` to the ignore-missing-imports override. |
| F8 | No store to run the quick start or a demo against. | `examples/demo-store/`: a synthetic `personal` store (a fictional engineer, 7 memories). README gains a 60-second demo with verbatim expected output and a runnable Python quick start. |
| F9 | README CLI and layout lists predate `bundle`, `relayout`, `export`, `graph`, and three modules. | CLI list, layout, and the MCP tool surface now match `hub --help` and `mcp_server.py`. |
| F10 | Placeholder project URLs. | Point both at the GitHub repo. |
| F11 | CI invisible and undocumented; actions warn about the Node 20 deprecation. | CI and licence badges; a README table of what each CI step blocks and what is deliberately not in CI; `actions/checkout@v5`, `actions/setup-python@v6`. |
| F12 | Found during the re-run: every `hub bundle` appends to `<content_root>/.stats/context_log.jsonl`, so running the demo dirties the clone. | Ignore `examples/*/memory/.stats/` (and the demo's `.index/`, `.trash/`), and say in the README that the log exists. `uv.lock`, which `uv sync` writes, is ignored too — this is a library, and pins live in `pyproject.toml`. |

## Matrix — after

Re-run from a fresh clone of the fixed commit, same clean environment.

| # | Check | Result | Evidence |
|---|---|---|---|
| C1 | README install works as written | pass | `uv sync` on 3.11, 3.12, 3.13; pip-in-venv path installs `.[dev,mcp,vectors]` |
| C2 | `uv sync; uv run pytest` | pass | 418 passed, 3 deselected (`-m local`) on each of 3.11 / 3.12 / 3.13 |
| C3 | Repo verification from a clean install | pass | `pytest`, `ruff check .`, `black --check .`, `mypy`, import check, `hub schema export` — all green on each interpreter |
| C4 | Quick start | pass | Python snippet runs from the clone root and prints what the README says |
| C5 | Licence present | pass | `LICENSE` |
| C6–C9 | Secrets, personal data, dead code, TODOs | pass | unchanged |
| C10 | Docs match the CLI surface | pass | every `hub --help` command listed; layout lists every module |
| C11 | Package metadata | pass | GitHub URLs |
| C12 | 60-second demo | pass | every command's output in the README matched the fresh-clone run; `git status` clean afterwards |
| C13 | CI face | **pass locally, pending on GitHub** | badge and gate docs in the README; the CI-equivalent install passes locally on all three interpreters. The badge turns green only once this branch reaches `main` on GitHub — see W3. |

## Waivers

| # | Item | Why waived |
|---|---|---|
| W1 | `hub search` / `hub bundle` with a natural-language query return nothing without a vector index; the demo uses a single-word query instead. | Fulltext fallback is literal by design and documented. A token-level fallback would be a new capability — backlog for the retro, not audit work. |
| W2 | `AGENTS.md`, `CLAUDE.md`, `RTK.md` and `.ai/` describe the maintainer's agent workflow, including one absolute local path. | They are the repo's agent-instruction layer, deliberately committed; the path is not a secret and holds no personal content. |
| W3 | The GitHub badge is red until the fixes are pushed; local `main` is 16 commits ahead of `origin/main`. | Pushing to a public repo is the owner's call, not the audit's. |
| W4 | `authors = [{ name = "MemoryHub" }]` and the licence's "The MemoryHub authors" name no person. | This repo carries no personal data by rule; naming the owner is the owner's choice. |
| W5 | The golden-query retrieval eval (`pytest -m local`) was not run. | It needs a GPU embedding model and the private corpus — out of reach for a stranger by design, and documented as not in CI. |

## Backlog (for the roadmap retro)

- Port `mcp_server.py` and its tests to mcp 2.x (`MCPServer`), then lift the `<2` pin.
- Consider a token-level fulltext fallback so `search`/`bundle` answer natural-language queries
  without an index (W1).
