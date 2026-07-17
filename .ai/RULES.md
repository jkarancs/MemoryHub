---
description: Governance for the .ai/ knowledge layer — loading order, canonical locations, extension protocol, capture triggers.
status: reviewed
---
# Knowledge Rules

> Parent: [CONTEXT.md](CONTEXT.md)

## Loading order
1. `AGENTS.md` (always in context) → 2. [CONTEXT.md](CONTEXT.md) — **mandatory before any source exploration**; it is the complete index → 3. the one topic or flow file the task needs. Stop when the next action is clear. Prefer summaries over source; expand into source only when the knowledge layer does not answer.

## Canonical locations
| Knowledge type | Home |
|---|---|
| Commands, conventions, judgment boundaries | `AGENTS.md` |
| System snapshot, mental model, navigation | `CONTEXT.md` |
| Cross-cutting topics, key flows, domain language | `knowledge/` |
| Repo structure & regenerable inventories | `maps/` (`repo-map.md`, `dependency-map.md` — refresh overwrites; never hand-edit) |
| Transient notes | `scratchpad/` (gitignored; safe to delete) |
| Secrets | **nowhere** — reference names + store (keyring service, env var), never values |

## Extension protocol (adding knowledge)
1. Update the canonical location first; create a new file only for a new, long, or independently reusable topic.
2. New file = frontmatter (`description`, `status: draft`) + `> Parent:` line + a link from its parent index.
3. Distilled facts only — no raw logs, command output, chat history, or dates.
4. One fact, one home: link, never duplicate.
5. Respect the caps (see the generator prompt, §8); split before exceeding.

## Capture triggers — write a note when you discover…
- A non-obvious root cause or fix path.
- A command/query/test sequence reliably used more than once (consider a skill in `.agents/skills/` if it is a multi-step procedure).
- A stale fact corrected against source or config.
- A tradeoff decision that shaped the design (consider an ADR).
Do NOT capture: anything the repo or its configs already express, speculation, one-off local state, or personal facts — those belong in the memoryhub store, not the repo KB.

## Lifecycle
- Generated files start as `status: draft`; an independent review pass (generator prompt §7.2 — fresh context, never the writer's own) verifies and promotes them to `reviewed`. The whole lifecycle is agent-run; the commit diff is the audit trail.
- The KB is code: the invariants (KB01–KB18) are the review bar, and every run ends in a one-line commit.
- Refresh runs regenerate `maps/` + the manifest; proven drift is corrected in place, the touched file reverts to `status: draft`, and the run's review pass re-verifies it. Ambiguous drift is only proposed in the run report.
- Verification bar for "done" work: `uv run --no-sync pytest` (ledger: workspace roadmap specs `../../specs/ai-roadmap/`).
