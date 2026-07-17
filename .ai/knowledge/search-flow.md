---
description: Search flow — hybrid vector + fulltext retrieval, degradation to fulltext, staleness warnings, reindex.
status: reviewed
---
# Search Flow

> Parent: [README.md](README.md)

**Trigger:** `Hub.search`, `hub search`, or MCP `search_memory` → **Outcome:** ranked `MemoryDoc` list (hybrid RRF-fused by default).

| # | Step | Code | Rule enforced |
|---|---|---|---|
| 1 | Frontmatter filters applied in memory; scalar ones (`type`/`status`/`visibility`) also pushed down to the index as `where` clauses; list-valued `tags` filtered engine-side | `src/memoryhub/hub.py · Hub.search` | filters always apply, whatever the mode |
| 2 | `hybrid`/`vector` need a usable index: missing `vectors` extra or unbuilt index → degrade to fulltext with an `IndexWarning` | `src/memoryhub/hub.py · Hub._usable_index` | search never hard-fails for a missing vector stack |
| 3 | Stale index (stored content hashes ≠ current docs) warns but still serves | `src/memoryhub/hub.py · Hub._warn_if_stale` | — |
| 4 | Vector search over-fetches `k = max(limit*5, 50)`; ids outside the filtered set are dropped after ranking | `src/memoryhub/hub.py · Hub.search` | fusion needs depth beyond `limit` |
| 5 | `hybrid` fuses vector + fulltext rankings via RRF | `src/memoryhub/hub.py · Hub.search` · `src/memoryhub/query.py · rrf_fuse` | — |
| 6 | `reindex` is incremental by content hash; `--full` re-embeds everything | `src/memoryhub/hub.py · Hub.reindex` · `src/memoryhub/index.py · VectorIndex.reindex` | — |

## Failure paths
- `EmbeddingError` at query time → fulltext fallback with `IndexWarning` (`src/memoryhub/hub.py · Hub.search`).
- MCP startup warms the search stack on the **main thread before** FastMCP's event loop — first-import of native deps (torch/lancedb) inside the loop can deadlock the OS library loader, observed reliably on Windows; warmup failure only means fulltext fallback, never a dead server (`src/memoryhub/mcp_server.py · warm_search_stack`).

## Invariants
- Reads are served from an in-memory doc cache invalidated by a stat-scan (path + mtime + size) on every access (`src/memoryhub/hub.py · Hub.all`).
- Retrieval quality is tracked by a golden-query eval (`tests/eval_queries.yaml`), local-only: `uv run --no-sync pytest -m local` (`tests/test_eval.py`).
