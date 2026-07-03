# MemoryHub

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

```bash
# core engine
pip install -e .

# with the MCP server (`hub mcp`)
pip install -e ".[mcp]"

# with vector-store / embedding extras
pip install -e ".[vectors,local-embed]"

# developer tooling
pip install -e ".[dev,mcp]"
```

> **Python:** requires 3.11+. The `local-embed` extra pulls in `torch`, whose wheels may lag the
> newest CPython release; use a Python version with published `torch` wheels for that extra.

## Quick start

```python
from memoryhub import Hub, load_config, load_profile

config = load_config("path/to/content-repo/")   # walks up to find hub.toml
profile = load_profile(config.hub.profile)       # built-in name or path to a .yaml
hub = Hub(config)
```

## CLI

```bash
hub list --type skill      # frontmatter summaries, filterable (--tags/--status/--json …)
hub get <id>               # one memory (raw markdown, or --json)
hub find <term>            # full-text search, ranked by match count
hub search <query>         # hybrid semantic search (--mode vector|text|hybrid, --limit, --json)
hub reindex                # build/update the vector index (--full to re-embed everything)
hub validate               # CI-grade store validation (non-zero exit on failure, --json)
hub new / add / update / rm  # guarded writes (validated before disk, atomic, soft delete)
hub mcp                    # serve the store to agents over MCP (stdio; needs the [mcp] extra)
hub schema export          # write frontmatter.schema.json from models + active profile
```

Every command resolves `hub.toml` by walking up from the current directory.

`hub search` needs the `vectors` extra plus an embedding backend (`local-embed` or `api-embed`,
selected in `hub.toml`); without them — or before the first `hub reindex` — it degrades to
fulltext with a warning. Retrieval quality is tracked by a golden-query eval
(`tests/eval_queries.yaml`) that runs only locally: `pytest -m local`.

## Layout

```
src/memoryhub/
├── __init__.py       public API surface
├── config.py         load/validate hub.toml → Config
├── profiles.py       schema profiles (type vocab + per-type fields)
├── models.py         Pydantic: MemoryDoc, Frontmatter, JSON-schema builder
├── loader.py         parse/serialize markdown+frontmatter, validate
├── query.py          frontmatter filtering + full-text
├── writer.py         add/update/delete memory files (atomic, guarded)
├── hub.py            Hub facade — the single engine everything calls
├── ids.py            slug/id generation + uniqueness
├── cli.py            `hub` command (Typer)
├── mcp_server.py     FastMCP stdio server over Hub (guarded agent writes)
├── embeddings.py     embedding backends (sentence-transformers local / OpenAI-compatible API)
├── index.py          LanceDB vector index (incremental reindex, filtered ANN search)
└── profiles/
    └── personal.yaml the personal schema profile
```

## License

MIT
