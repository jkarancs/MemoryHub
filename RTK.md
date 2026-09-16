# RTK

Prefix **filterable programs** with `rtk`: `rtk git status`, `rtk cargo test`,
`rtk npm run build`, `rtk ls src/`. Keep the prefix on real programs inside
chains: `rtk git add . && rtk git commit -m "msg"`.

Do **not** prefix shell builtins or keywords. `rtk` execs `argv[0]`; it is not
a shell. `rtk cd`, `rtk source`, `rtk export`, `rtk eval`, `rtk set`, `rtk
unset`, `rtk alias`, `rtk pushd`, `rtk type`, and other builtins fail with
127. Keep the builtin bare, or wrap a whole script in `rtk run`:

```bash
cd /path && rtk git status
rtk run 'cd /path && git status && ls'
```

`rtk run '…'` is `sh -c`. `rtk proxy <cmd>` still execs, so it is not a
builtin workaround. A program RTK has no filter for can still be prefixed
when it is a real executable (`rtk ./tool`); that passthrough is exec, not
`sh -c`.

# Command output

Command output here is condensed to save tokens, keeping every signal and
dropping costly noise. Treat it as the complete result: run commands
normally, and batch related commands into one call to avoid extra turns.
Truncated results state their recovery path in their own output. Re-run a
command as `rtk proxy <cmd>` only when its result is unusable: empty when
output was clearly expected, contradicting its exit code, or garbled.

## About RTK

RTK (Rust Token Killer) is a CLI proxy that filters command output to save
tokens; behavior and exit code are unchanged.

- `rtk gain` / `rtk gain --history` — token savings, overall and per command.
- `rtk proxy <cmd>` — run a command unfiltered, still tracked (must be an executable).
- `rtk run '…'` — `sh -c`, the way to include builtins.
- `RTK_DISABLED=1 <cmd>` — skip RTK for one command.
- `rtk discover` — find past commands RTK could have condensed.
