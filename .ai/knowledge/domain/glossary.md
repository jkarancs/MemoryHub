---
description: Domain glossary for MemoryHub — memory, store, profile, type vocabulary, status/visibility enums, bundles.
status: reviewed
---
# Glossary

> Parent: [../README.md](../README.md)

| Term | Meaning |
|---|---|
| Memory | One markdown file with typed YAML frontmatter + body, parsed as `MemoryDoc` (`src/memoryhub/models.py · MemoryDoc`) |
| Store | A directory tree of memories inside a content repo, rooted at `content_root` from `hub.toml` |
| Content repo | A repo (e.g. `../personal-memory`) that holds a store + `hub.toml` and depends on this engine; the engine itself holds no content |
| Schema profile | Closed `type` vocabulary + per-type extra fields, from a YAML file — the generalization seam (`src/memoryhub/profiles.py`, `src/memoryhub/profiles/personal.yaml`) |
| Type | One of the profile's closed vocabulary; `personal`: bio, skill, experience, education, project, goal, preference, writing (`src/memoryhub/profiles/personal.yaml`) |
| `status` | `active` (established fact) · `draft` (awaiting human review — all agent writes start here) · `archived` · `aspirational` (`src/memoryhub/profiles/personal.yaml · enums`) |
| `visibility` | `public` (exportable once active) · `private` (`src/memoryhub/profiles/personal.yaml · enums`) |
| `source` | Who authored the memory; defaults `self`, forced to `agent` for MCP writes (`src/memoryhub/models.py · Frontmatter`, `src/memoryhub/mcp_server.py · add_memory`) |
| `related` | List of memory ids linking memories; unresolved ids warn, never fail (`src/memoryhub/writer.py · _warn_unresolved_related`) |
| Id | Slug `^[a-z0-9][a-z0-9-]*$` (`src/memoryhub/models.py · SLUG_RE`), unique across the store, generated from type + title (`src/memoryhub/ids.py · slugify`) |
| Bundle / context pack | Token-budget-bounded markdown pack built by `recall_bundle`: greedy fill, richest render level that fits, with an included/excluded manifest (`src/memoryhub/bundle.py · pack`) |
| Context log | Per-store log of bundle calls — the per-task context-cost metric read by `context_stats` (`src/memoryhub/bundle.py · log_bundle / load_stats`) |
| Golden-query eval | Local-only retrieval-quality suite over the real corpus: `tests/eval_queries.yaml`, `pytest -m local` (`tests/test_eval.py`) |
