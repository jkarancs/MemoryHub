---
description: Public export flow — hub export selection rules, deterministic sync, and safety gates before publishing.
status: reviewed
---
# Public Export Flow

> Parent: [README.md](README.md)

**Trigger:** `hub export` / `Hub.export(dest)` (run from a content repo) → **Outcome:** the public subset of the store deterministically synced into a destination repo that is itself a valid read-only hub store.

| # | Step | Code | Rule enforced |
|---|---|---|---|
| 1 | Refuse destinations that overlap the source store (either direction, or the repo root) | `src/memoryhub/export.py · _check_dest` | export can never clobber the source |
| 2 | Refuse to run while store validation fails | `src/memoryhub/export.py · export_store` | only a valid store is published |
| 3 | Select `visibility: public` **and** `status: active` docs only, sorted by id | `src/memoryhub/export.py · _select` | drafts/archived/aspirational never export |
| 4 | Strip `related` ids pointing at non-exported docs; note the count as an inline YAML comment; bodies untouched | `src/memoryhub/export.py · _render / _annotate_related` | no dangling links in the public store |
| 5 | Scan outgoing text for emails / phone numbers / API-key shapes / secret assignments; hits require confirmation, no confirmer → refuse | `src/memoryhub/export.py · _scan / _confirm_or_refuse` | warn-and-confirm, never silent publish |
| 6 | Plan all managed files, then write only changed ones; delete files that left the export set; prune empty type dirs | `src/memoryhub/export.py · export_store / _prune_empty_dirs` | deterministic full sync — running twice yields zero diff |
| 7 | Regenerate the destination `README.md` index (grouped by type); create a minimal read-only `hub.toml` once (`allow_agent_writes = false`) | `src/memoryhub/export.py · _readme / _minimal_hub_toml` | destination stays read-only |

## Failure paths
- Any refusal raises `ExportError`; `dry_run=True` reports what would change without touching the destination (scan still runs, but does not prompt) (`src/memoryhub/export.py · export_store`).
