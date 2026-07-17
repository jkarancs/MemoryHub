---
description: Project coding standards no linter enforces — for AI-assisted coding in MemoryHub.
status: reviewed
---
# Coding Standards — Project Layer

> Parent: [README.md](README.md)

## Project rules (not tool-enforced)
| Rule | Why |
|---|---|
| Pydantic v2 models with `extra="forbid"` for frontmatter/config DTOs | typos surface as validation errors instead of silently passing (`src/memoryhub/models.py · Frontmatter`) |
| Callers go through the `Hub` facade — never reach into `loader`/`writer`/`query` directly | single seam for cache invalidation and policy (`src/memoryhub/hub.py`) |
| Writes are atomic (same-dir temp file + `os.replace`) and validated **before** touching disk | a failed write must leave the store untouched (`src/memoryhub/writer.py`) |

## Baseline
1. **Think before coding** — state assumptions; ask on ambiguity; present options when several interpretations are valid.
2. **Simplicity first** — minimum code that solves the stated problem; no speculative features, abstractions, or configurability.
3. **Surgical changes** — touch only what the task requires; match existing style; remove only what your change orphaned.
4. **Goal-driven execution** — turn vague asks into verifiable goals ("fix the bug" → "write a failing test, make it pass"); plan multi-step work with a check per step; run the project verification before declaring done.
