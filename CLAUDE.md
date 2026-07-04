# CLAUDE.md — MemoryHub (engine)

Reusable, content-agnostic engine over a markdown memory store. **Contains no personal data** —
content lives in separate repos (`personal-memory`, future `work-memory`) that depend on this
package. Never commit content here.

## Architecture
- `Hub` (`hub.py`) is the **only** public entry point — CLI, MCP server, and search all go through it.
- Schema **profiles** (`profiles/*.yaml`) define the closed `type` vocabulary + per-type fields —
  the generalization seam. New deployment = new `*.yaml`, no engine changes.
- Config comes from a content repo's `hub.toml`, loaded by `config.load_config` (walks up to find it).



## Commands
```bash
pip install -e ".[dev,mcp,vectors]"        # engine + tooling (CI-equivalent install)
pytest                                     # tests (deselects `-m local`; ≥90% cov on core)
pytest -m local                            # golden-query eval vs ../personal-memory (GPU/model)
ruff check . && black --check . && mypy    # lint / format / types
hub schema export                          # write schema/frontmatter.schema.json
hub reindex && hub search "<query>"        # vector index + hybrid search (from a content repo)
```
Local venvs here are **uv-managed** (no pip): `uv pip install --python .venv/Scripts/python.exe …`

## Conventions
- Python 3.11+, `src/` layout, Pydantic v2 (`extra="forbid"` so typos surface as errors).
- IDs are slugs `^[a-z0-9][a-z0-9-]*$`, unique across the store.
- Writes are atomic (temp file + `os.replace`) and validated **before** touching disk.

## Committing

- Commit on the user's behalf: one-line summary only (conventional prefix), never a multi-line body.
- When fixing something just committed, amend the previous commit rather than stacking fixups.
- Default branch is `main`.
