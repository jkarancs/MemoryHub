# MemoryHub

[![CI](https://github.com/jkarancs/MemoryHub/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jkarancs/MemoryHub/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A reusable, **content-agnostic** engine over a markdown memory store.

MemoryHub reads a directory of markdown files with YAML frontmatter, validates them against a
**schema profile**, and exposes a single `Hub` facade for reading, querying, and (guarded) writing.
The engine holds no personal data — content lives in separate repos (e.g. `personal-memory`) that
depend on this package and carry their own `hub.toml` config and schema profile.

```
 MemoryHub (engine, reusable)
        ▲                    ▲
        │ pip/git dep        │ pip/git dep
 personal-memory       (future) work-memory
   (private content)     (private content)
```

## Install

Requires Python 3.11+. With [uv](https://docs.astral.sh/uv/) (what the maintainers use):

```bash
git clone https://github.com/jkarancs/MemoryHub.git
cd MemoryHub
uv sync            # creates .venv with the engine + the dev group (tests, lint, MCP, vectors)
```

Or with plain `pip`, inside a virtual environment (many distros refuse a system-wide install):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                                  # core engine
pip install -e ".[mcp]"                           # + the MCP server (`hub mcp`)
pip install -e ".[vectors,local-embed]"           # + vector search with a local embedder
pip install -e ".[dev,mcp,vectors]"               # what CI installs
```

> The `local-embed` extra pulls in `torch`, whose wheels may lag the newest CPython release; use
> a Python version with published `torch` wheels for that extra. `api-embed` (OpenAI-compatible
> embeddings) is the lightweight alternative.

## 60-second demo

The repo ships a synthetic store, [`examples/demo-store/`](examples/demo-store) — a fictional
engineer, no real data. `hub` resolves `hub.toml` by walking up from the current directory, so
run it from there (with pip, drop the `uv run` prefix):

```console
$ cd examples/demo-store
$ uv run hub validate
OK: 7 file(s) valid

$ uv run hub list --type skill
skill-async-python   skill  active  Async Python
skill-hybrid-search  skill  active  Hybrid search
skill-postgresql     skill  active  PostgreSQL

$ uv run hub find retrieval
project-support-search          project  active        Support search
skill-hybrid-search             skill    active        Hybrid search
bio-robin-vega                  bio      active        Robin Vega
goal-lead-a-retrieval-platform  goal     aspirational  Lead a retrieval platform

$ uv run hub bundle retrieval --budget 120      # a context pack that never exceeds the budget
warning: no vector index at …/examples/demo-store/.index (run `hub reindex`) — falling back to fulltext
# Context for: retrieval

## project-support-search — Support search
…
119/120 tokens (3 included, 1 excluded, counter: tiktoken)
  #1  project-support-search  full     32 tok
  #2  skill-hybrid-search     full     36 tok
  #3  bio-robin-vega          full     43 tok
  excluded #4 goal-lead-a-retrieval-platform (budget)

$ uv run hub add --type recipe --title Pancakes  # writes are validated before touching disk
add requires a 'type' from the profile vocabulary (bio, skill, experience, education, project, goal, preference, writing)
```

The pack goes to stdout and the manifest to stderr, so `hub bundle … > pack.md` captures just
the pack; each call also appends a line to `<content_root>/.stats/context_log.jsonl`, which
`hub mcp`'s `context_stats` tool summarizes. Without a vector index, `search` and `bundle` fall
back to literal fulltext (hence the single-word query); `hub reindex` with an embedding backend
enables natural-language queries.

## Quick start (Python)

```python
from memoryhub import Hub

hub = Hub("examples/demo-store")          # a directory; walks up to find hub.toml
print(hub.validate().ok)                  # True
for doc in hub.filter(type="skill"):
    print(doc.frontmatter.id, "-", doc.frontmatter.description)
pack = hub.recall_bundle("retrieval", 300)   # same packer as `hub bundle`; warns without an index
print(pack.total_tokens, [item.id for item in pack.manifest])
```

A store of your own is a directory with a `hub.toml` (copy
[`examples/demo-store/hub.toml`](examples/demo-store/hub.toml)) and markdown files under its
`content_root`. `hub new <type> --title …` scaffolds a file in the right folder; the
[`templates/`](templates) directory has one annotated starter per type of the built-in
`personal` profile. A different deployment is a new profile YAML (`profile = "./my.yaml"`),
not an engine change.

## CLI

```bash
hub list --type skill        # frontmatter summaries, filterable (--tags/--status/--json …)
hub get <id>                 # one memory (raw markdown, or --json)
hub find <term>              # full-text search, ranked by match count
hub search <query>           # hybrid semantic search (--mode vector|text|hybrid, --limit, --json)
hub bundle <task>            # token-budgeted context pack (--budget, --type, --tags, --json)
hub reindex                  # build/update the vector index (--full to re-embed everything)
hub validate                 # CI-grade store validation (non-zero exit on failure, --json)
hub new / add / update / rm  # guarded writes (validated before disk, atomic, soft delete)
hub relayout                 # move files to the paths the profile's layout rules imply (dry-run)
hub export --dest <dir>      # deterministic sync of public+active memories into a public store
hub graph <cmd>              # next/claim/ready/bulk/status/validate over a workflow-profile store
hub mcp                      # serve the store to agents over MCP (stdio; needs the [mcp] extra)
hub schema export            # write frontmatter.schema.json from models + active profile
```

Every command resolves `hub.toml` by walking up from the current directory; `hub <cmd> --help`
documents each one.

`hub search` needs the `vectors` extra plus an embedding backend (`local-embed` or `api-embed`,
selected in `hub.toml`); without them — or before the first `hub reindex` — it degrades to
fulltext with a warning. Retrieval quality is tracked by a golden-query eval
(`tests/eval_queries.yaml`) that runs only locally: `pytest -m local`.

`hub mcp` exposes `search_memory`, `get_memory`, `list_memories`, `list_types`, `list_tags`,
`recall_bundle`, and `context_stats`, plus — when `hub.toml` allows agent writes —
`add_memory` (always a private draft), `update_memory` (cannot set `status: active`), and
`archive_memory` — activating or publishing a memory stays a human decision.

## CI and development

Every push and pull request to `main` runs [`ci.yml`](.github/workflows/ci.yml) on Python 3.11,
3.12, and 3.13. It is hermetic — no model downloads, no API keys, no content repo — and
each step blocks a specific regression:

| Step | Blocks |
|---|---|
| Import check | a broken public API (`from memoryhub import Hub, load_config, load_profile`) |
| Schema export | models or the built-in profile no longer producing a JSON schema |
| `ruff check .` / `black --check .` | lint errors and unformatted code |
| `mypy` | type errors in `src/memoryhub` |
| `pytest --cov=memoryhub` | behavior regressions — config, validation, guarded writes, search fusion (with a fake embedder over a real LanceDB index), bundling, export, the graph engine, and the MCP server over an in-memory client |

Not in CI: the golden-query retrieval eval (`pytest -m local`), which needs a GPU-backed
embedding model and a real corpus. Run the same gate locally:

```bash
uv run pytest && uv run ruff check . && uv run black --check . && uv run mypy
```

## Layout

```
src/memoryhub/
├── __init__.py       public API surface
├── config.py         load/validate hub.toml → Config
├── profiles.py       schema profiles (type vocab + per-type fields)
├── models.py         Pydantic: MemoryDoc, Frontmatter, JSON-schema builder
├── loader.py         parse/serialize markdown+frontmatter, validate
├── query.py          frontmatter filtering + full-text
├── graph.py          dependency-graph traversal for workflow stores (`hub graph`)
├── writer.py         add/update/delete memory files (atomic, guarded)
├── hub.py            Hub facade — the single engine everything calls
├── ids.py            slug/id generation + uniqueness
├── bundle.py         token-budgeted context packs (`hub bundle`, `recall_bundle`)
├── tokens.py         token counting (tiktoken, or a len//4 fallback)
├── export.py         public-subset export with a sensitive-content scan
├── cli.py            `hub` command (Typer)
├── mcp_server.py     FastMCP stdio server over Hub (guarded agent writes)
├── embeddings.py     embedding backends (sentence-transformers local / OpenAI-compatible API)
├── index.py          LanceDB vector index (incremental reindex, filtered ANN search)
└── profiles/
    └── personal.yaml the personal schema profile
examples/demo-store/  synthetic store for the demo above
templates/            one starter file per `personal` type
notebooks/            context_budgeting.ipynb — the bundle packer, worked as an exercise
```

## License

[MIT](LICENSE)
