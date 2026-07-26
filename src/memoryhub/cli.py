"""The ``hub`` command-line interface (Typer).

Every command resolves ``hub.toml`` by walking up from the current directory (via
:func:`memoryhub.config.load_config`). Read commands accept ``--json`` so pipelines can script
against the CLI without parsing tables; ``hub validate`` is CI-grade (non-zero exit on any
invalid file, ``--json`` for the aggregated report).

Exit codes: 0 success · 1 validation/write/not-found errors · 2 configuration errors.
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn

import typer
import yaml

from . import export as export_module
from . import writer
from .bundle import Bundle
from .config import ConfigError, load_config
from .embeddings import EmbeddingError
from .export import ExportError
from .graph import ACTIONABLE, NODE, TYPE_STATUSES, Graph, GraphError
from .hub import Hub
from .index import IndexWarning
from .loader import LoadError, StoreReport, flat_frontmatter, serialize
from .models import MemoryDoc, frontmatter_json_schema
from .profiles import Profile, load_profile
from .writer import WriteError, WriteWarning

app = typer.Typer(
    name="hub",
    help="MemoryHub CLI — read, query, and (guarded) write a markdown memory store.",
    no_args_is_help=True,
    add_completion=False,
)
schema_app = typer.Typer(help="Schema tooling.", no_args_is_help=True)
app.add_typer(schema_app, name="schema")
graph_app = typer.Typer(
    help="Dependency-graph queries over a workflow store (see memoryhub.graph).",
    no_args_is_help=True,
)
app.add_typer(graph_app, name="graph")


def _force_utf8_streams() -> None:
    """Emit UTF-8 regardless of the platform's locale encoding.

    On Windows a *redirected* stdout/stderr defaults to cp1252, which mojibakes any non-ASCII in a
    pack — the heading em-dash, accented names, Hungarian text — so `hub bundle ... > pack.md`
    would corrupt. Reconfiguring to UTF-8 matches how the engine writes files. A no-op where the
    stream can't be reconfigured (e.g. test capture buffers already on UTF-8).
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8")


@app.callback()
def _root() -> None:
    """MemoryHub CLI — read, query, and (guarded) write a markdown memory store."""
    _force_utf8_streams()


DEFAULT_SCHEMA_OUT = Path("schema") / "frontmatter.schema.json"

# --- shared helpers ------------------------------------------------------------


def _fail(message: str, code: int = 1) -> NoReturn:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=code)


def _open_hub() -> Hub:
    try:
        return Hub(load_config(Path.cwd()))
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        _fail(str(exc), code=2)


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _doc_summary(doc: MemoryDoc, profile: Profile | None) -> dict[str, Any]:
    summary = flat_frontmatter(doc.frontmatter, profile)
    summary["path"] = str(doc.path) if doc.path else None
    return summary


def _print_table(docs: list[MemoryDoc]) -> None:
    if not docs:
        typer.secho("(no memories)", dim=True)
        return
    id_w = max(len(d.id) for d in docs)
    type_w = max(len(d.type) for d in docs)
    status_w = max(len(d.frontmatter.status) for d in docs)
    for doc in docs:
        fm = doc.frontmatter
        typer.echo(f"{doc.id:<{id_w}}  {doc.type:<{type_w}}  {fm.status:<{status_w}}  {fm.title}")


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def _parse_sets(pairs: list[str] | None) -> dict[str, Any]:
    """Parse repeated ``--set key=value`` pairs; values go through YAML for typing."""
    fields: dict[str, Any] = {}
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        if not sep or not key.strip():
            _fail(f"--set expects key=value, got {pair!r}")
        try:
            fields[key.strip()] = yaml.safe_load(raw)
        except yaml.YAMLError:
            fields[key.strip()] = raw
    return fields


def _read_body_file(body_file: Path | None) -> str | None:
    if body_file is None:
        return None
    try:
        return body_file.read_text(encoding="utf-8")
    except OSError as exc:
        _fail(f"cannot read body file: {exc}")


@contextlib.contextmanager
def _guarded_write() -> Iterator[None]:
    """Wire the interactive confirmer and surface WriteWarnings/WriteErrors nicely."""
    writer.confirm_callback = typer.confirm
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", WriteWarning)
        try:
            yield
        except WriteError as exc:
            _fail(str(exc))
        finally:
            for warning in caught:
                if issubclass(warning.category, WriteWarning):
                    typer.secho(f"warning: {warning.message}", fg=typer.colors.YELLOW, err=True)


def _resolve_profile(profile_opt: str | None) -> Profile:
    """Resolve the active profile.

    Priority: explicit ``--profile`` > the profile named in a nearby ``hub.toml`` > built-in
    ``personal`` (so the command works even in the engine repo, which has no ``hub.toml``).
    """
    if profile_opt:
        return load_profile(profile_opt)
    try:
        config = load_config(Path.cwd())
    except ConfigError:
        return load_profile("personal")
    return load_profile(config.profile_ref)


# --- schema --------------------------------------------------------------------


@schema_app.command("export")
def schema_export(
    profile: str | None = typer.Option(
        None,
        "--profile",
        "-p",
        help="Profile name or path to a .yaml. Defaults to the nearby hub.toml, else 'personal'.",
    ),
    out: Path = typer.Option(
        DEFAULT_SCHEMA_OUT,
        "--out",
        "-o",
        help="Output path for the generated JSON Schema.",
    ),
    stdout: bool = typer.Option(
        False,
        "--stdout",
        help="Write the schema to stdout instead of a file.",
    ),
) -> None:
    """Generate ``frontmatter.schema.json`` from the models + active profile."""
    prof = _resolve_profile(profile)
    schema = frontmatter_json_schema(prof)
    payload = json.dumps(schema, indent=2, ensure_ascii=False) + "\n"

    if stdout:
        typer.echo(payload, nl=False)
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")
    typer.secho(
        f"Wrote JSON Schema for profile '{prof.name}' to {out}",
        fg=typer.colors.GREEN,
    )


# --- reads -----------------------------------------------------------------------


@app.command("list")
def list_cmd(
    type: str | None = typer.Option(None, "--type", help="Filter by memory type."),
    tags: str | None = typer.Option(None, "--tags", help="Comma-separated tags (AND-match)."),
    status: str | None = typer.Option(None, "--status", help="Filter by status."),
    visibility: str | None = typer.Option(None, "--visibility", help="Filter by visibility."),
    json_out: bool = typer.Option(False, "--json", help="Emit frontmatter summaries as JSON."),
) -> None:
    """List memories, optionally filtered by frontmatter dimensions."""
    hub = _open_hub()
    try:
        docs = hub.filter(type=type, tags=_split_csv(tags), status=status, visibility=visibility)
    except LoadError as exc:
        _fail(str(exc))
    if json_out:
        _echo_json([_doc_summary(doc, hub.profile) for doc in docs])
    else:
        _print_table(docs)


@app.command("get")
def get_cmd(
    id: str = typer.Argument(..., help="Memory id."),
    json_out: bool = typer.Option(False, "--json", help="Emit frontmatter + body + path as JSON."),
) -> None:
    """Show a single memory (raw markdown, or --json)."""
    hub = _open_hub()
    try:
        doc = hub.get(id)
    except LoadError as exc:
        _fail(str(exc))
    except KeyError as exc:
        _fail(str(exc.args[0]))
    if json_out:
        payload = _doc_summary(doc, hub.profile)
        payload["body"] = doc.body
        _echo_json(payload)
    else:
        typer.echo(serialize(doc, hub.profile), nl=False)


@app.command("find")
def find_cmd(
    term: str = typer.Argument(..., help="Search term (literal unless --regex)."),
    regex: bool = typer.Option(False, "--regex", help="Treat the term as a regular expression."),
    field: list[str] | None = typer.Option(
        None, "--field", help="Restrict the searched fields (repeatable)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit matches as JSON."),
) -> None:
    """Full-text search, ranked by match count."""
    hub = _open_hub()
    kwargs: dict[str, Any] = {"regex": regex}
    if field:
        kwargs["fields"] = tuple(field)
    try:
        docs = hub.fulltext(term, **kwargs)
    except LoadError as exc:
        _fail(str(exc))
    except re.error as exc:
        _fail(f"invalid regex {term!r}: {exc}", code=2)
    if json_out:
        _echo_json([_doc_summary(doc, hub.profile) for doc in docs])
    else:
        _print_table(docs)


@contextlib.contextmanager
def _surfaced_index_warnings() -> Iterator[None]:
    """Show IndexWarnings (fulltext fallback, stale index) to the user instead of swallowing."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IndexWarning)
        yield
    for warning in caught:
        if issubclass(warning.category, IndexWarning):
            typer.secho(f"warning: {warning.message}", fg=typer.colors.YELLOW, err=True)


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Natural-language search query."),
    mode: str = typer.Option(
        "hybrid", "--mode", help="Search mode: hybrid (default), vector, or text."
    ),
    type: str | None = typer.Option(None, "--type", help="Filter by memory type."),
    tags: str | None = typer.Option(None, "--tags", help="Comma-separated tags (AND-match)."),
    status: str | None = typer.Option(None, "--status", help="Filter by status."),
    visibility: str | None = typer.Option(None, "--visibility", help="Filter by visibility."),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum number of results."),
    json_out: bool = typer.Option(False, "--json", help="Emit matches as JSON."),
) -> None:
    """Semantic search over the store (vector + fulltext fused by default).

    Falls back to plain fulltext — with a warning — when the vectors extra isn't installed or
    no index has been built yet (`hub reindex`).
    """
    hub = _open_hub()
    try:
        with _surfaced_index_warnings():
            docs = hub.search(
                query,
                mode=mode,
                limit=limit,
                type=type,
                tags=_split_csv(tags),
                status=status,
                visibility=visibility,
            )
    except ValueError as exc:  # unknown mode
        _fail(str(exc), code=2)
    except (LoadError, EmbeddingError) as exc:
        _fail(str(exc))
    if json_out:
        _echo_json([_doc_summary(doc, hub.profile) for doc in docs])
    else:
        _print_table(docs)


def _print_bundle_manifest(bundle: Bundle) -> None:
    """Show the pack's manifest (what was included at which level) + exclusions, on stderr."""
    typer.secho(
        f"\n{bundle.total_tokens}/{bundle.budget} tokens "
        f"({len(bundle.manifest)} included, {len(bundle.excluded)} excluded, "
        f"counter: {bundle.counter})",
        fg=typer.colors.CYAN,
        err=True,
    )
    if bundle.manifest:
        rank_w = max(len(str(item.rank)) for item in bundle.manifest)
        id_w = max(len(item.id) for item in bundle.manifest)
        level_w = max(len(item.level) for item in bundle.manifest)
        for item in bundle.manifest:
            typer.secho(
                f"  #{item.rank:<{rank_w}}  {item.id:<{id_w}}  "
                f"{item.level:<{level_w}}  {item.tokens:>5} tok",
                dim=True,
                err=True,
            )
    for ex in bundle.excluded:
        where = f"#{ex.rank} " if ex.rank is not None else ""
        typer.secho(f"  excluded {where}{ex.id} ({ex.reason})", fg=typer.colors.YELLOW, err=True)


@app.command("bundle")
def bundle_cmd(
    task: str = typer.Argument(..., help="What you need context for (used as query + header)."),
    budget: int = typer.Option(2000, "--budget", "-b", help="Token budget; never exceeded."),
    type: str | None = typer.Option(None, "--type", help="Filter by memory type."),
    tags: str | None = typer.Option(None, "--tags", help="Comma-separated tags (AND-match)."),
    json_out: bool = typer.Option(False, "--json", help="Emit the whole bundle as JSON."),
) -> None:
    """Build a task-scoped context pack that fits a token budget (richest memories first).

    Prints the markdown pack to stdout and a manifest (levels + token costs + exclusions) to
    stderr, so `hub bundle ... > pack.md` captures just the pack. Falls back to plain fulltext —
    with a warning — when the vectors extra isn't installed or no index has been built yet.
    """
    hub = _open_hub()
    try:
        with _surfaced_index_warnings():
            bundle = hub.recall_bundle(task, budget, type=type, tags=_split_csv(tags))
    except (LoadError, EmbeddingError) as exc:
        _fail(str(exc))
    if json_out:
        _echo_json(bundle.model_dump())
        return
    typer.echo(bundle.text)
    _print_bundle_manifest(bundle)


@app.command("reindex")
def reindex_cmd(
    full: bool = typer.Option(
        False, "--full", help="Rebuild from scratch: re-embed every doc, not just changed ones."
    ),
) -> None:
    """Build or update the vector index over the store (needs the vectors extra + a backend)."""
    hub = _open_hub()
    try:
        stats = hub.reindex(full=full)
    except ModuleNotFoundError as exc:
        _fail(str(exc), code=2)
    except (LoadError, EmbeddingError) as exc:
        _fail(str(exc))
    typer.secho(str(stats), fg=typer.colors.GREEN)


def _emit_report(report: StoreReport, json_out: bool, ok_message: str) -> None:
    """Print a validation report (issues, then warnings) and exit non-zero if it failed."""
    if json_out:
        _echo_json(report.to_dict())
    else:
        for issue in report.issues:
            typer.secho(str(issue), fg=typer.colors.RED, err=True)
        for warning in report.warnings:
            typer.secho(f"warning: {warning}", fg=typer.colors.YELLOW, err=True)
        if report.ok:
            typer.secho(ok_message, fg=typer.colors.GREEN)
        else:
            typer.secho(
                f"FAILED: {len(report.issues)} problem(s) across {report.checked} file(s)",
                fg=typer.colors.RED,
                err=True,
            )
    if not report.ok:
        raise typer.Exit(code=1)


@app.command("validate")
def validate_cmd(
    json_out: bool = typer.Option(False, "--json", help="Emit the aggregated report as JSON."),
) -> None:
    """Validate the whole store; exits non-zero if any file fails.

    Unresolved `related` ids are reported as warnings (forward refs are allowed) and do not
    affect the exit code.
    """
    hub = _open_hub()
    report = hub.validate()
    _emit_report(report, json_out, f"OK: {report.checked} file(s) valid")


# --- graph -----------------------------------------------------------------------


def _open_graph() -> Graph:
    hub = _open_hub()
    try:
        return hub.graph()
    except LoadError as exc:
        _fail(str(exc))


def _statuses(option: str | None) -> list[str]:
    return _split_csv(option) or list(ACTIONABLE)


def _counts_line(counts: dict[str, int]) -> str:
    """`done 5 · planned 2`, in node-lifecycle order."""
    order = {name: i for i, name in enumerate(TYPE_STATUSES[NODE])}
    items = sorted(counts.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0]))
    return " · ".join(f"{name} {count}" for name, count in items) or "(no nodes)"


@graph_app.command("next")
def graph_next_cmd(
    scope: str = typer.Argument(..., help="Project or supernode id."),
    status: str | None = typer.Option(
        None, "--status", help="Comma-separated node statuses (default: every actionable one)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit the payload as JSON."),
) -> None:
    """The single next actionable node in SCOPE, with the subnodes the acting skill needs.

    A ready node is one whose status matches and whose every `depends_on` id is `done`. An idle
    scope is not an error: it prints `null` (`--json`) and exits 0.
    """
    graph = _open_graph()
    try:
        payload = graph.next(scope, _statuses(status))
    except GraphError as exc:
        _fail(str(exc))
    if json_out:
        _echo_json(payload)
        return
    if payload is None:
        typer.secho("(nothing ready)", dim=True)
        return
    node = payload["node"]
    typer.secho(f"{node['id']}  {node['status']}  {node['title']}", fg=typer.colors.GREEN)
    typer.echo(f"  supernode:  {payload['supernode']}")
    typer.echo(f"  repository: {payload['repository']}")
    typer.echo(f"  reads:      {', '.join(payload['reads']) or '(none)'}")


@graph_app.command("ready")
def graph_ready_cmd(
    scope: str = typer.Argument(..., help="Project or supernode id."),
    status: str | None = typer.Option(
        None, "--status", help="Comma-separated node statuses (default: every actionable one)."
    ),
    include_blocked: bool = typer.Option(
        False,
        "--include-blocked",
        help="Also list nodes whose dependencies aren't done (the human queue: a person's "
        "decision doesn't wait on unbuilt code).",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Emit each node as a `graph next` payload (body, repository, reads) instead of a "
        "frontmatter summary — one call to brief a whole queue. Implies --json.",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit the nodes as JSON."),
) -> None:
    """Every ready node in SCOPE, in walk order (`next` returns the first of these)."""
    graph = _open_graph()
    try:
        docs = graph.ready(scope, _statuses(status), include_blocked=include_blocked)
    except GraphError as exc:
        _fail(str(exc))
    if full:
        _echo_json([graph.brief(doc) for doc in docs])
    elif json_out:
        _echo_json([_doc_summary(doc, graph.profile) for doc in docs])
    else:
        _print_table(docs)


@graph_app.command("status")
def graph_status_cmd(
    scope: str = typer.Argument(..., help="Project or supernode id."),
    json_out: bool = typer.Option(False, "--json", help="Emit the counts as JSON."),
) -> None:
    """Node counts by status, per supernode — the PROGRESS.md replacement."""
    graph = _open_graph()
    try:
        report = graph.status(scope)
    except GraphError as exc:
        _fail(str(exc))
    if json_out:
        _echo_json(report)
        return
    typer.secho(
        f"{report['scope']}  ({report['type']}, {report['status']})  {report['title']}", bold=True
    )
    id_w = max([len(row["id"]) for row in report["supernodes"]] + [len("TOTAL")])
    for row in report["supernodes"]:
        typer.echo(
            f"  {row['id']:<{id_w}}  {row['status']:<12}  {row['total']:>3} nodes  "
            f"{row['ready']:>3} ready   {_counts_line(row['counts'])}"
        )
    typer.secho(
        f"  {'TOTAL':<{id_w}}  {'':<12}  {report['total']:>3} nodes  "
        f"{report['ready']:>3} ready   {_counts_line(report['counts'])}",
        fg=typer.colors.CYAN,
    )


@graph_app.command("validate")
def graph_validate_cmd(
    json_out: bool = typer.Option(False, "--json", help="Emit the aggregated report as JSON."),
) -> None:
    """Check the graph invariants; exits non-zero on any problem.

    Dangling references, dependency cycles, edges into superseded nodes, unfinished
    supersessions, node status vs the newest subnode's verdict, `attempt` vs impl-subnode
    count, id/filename agreement, per-type status validity and required fields (the profile can
    only declare one store-wide status enum and no per-type requirements), a work record for
    every started node, and subnodes their node actually lists.
    """
    graph = _open_graph()
    report = graph.validate()
    _emit_report(report, json_out, f"OK: {report.checked} document(s), graph invariants hold")


# --- export ----------------------------------------------------------------------


@app.command("export")
def export_cmd(
    dest: Path = typer.Option(
        ...,
        "--dest",
        "--to",
        help="Public export repo to sync into (e.g. ../personal-memory-public).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would change without writing anything."
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Publish despite sensitive-content warnings (skip the prompt)."
    ),
) -> None:
    """Sync public content into a public store (only `visibility: public` + `status: active`).

    A deterministic full sync: stale files in the destination are deleted, a README index and
    a minimal read-only hub.toml are generated, and a second run yields zero diff. Refuses to
    run while `hub validate` fails; emails/phone numbers/API-key shapes in exported text
    require confirmation. Review the destination diff before committing/pushing it.
    """
    hub = _open_hub()
    if yes:
        export_module.confirm_callback = lambda _prompt: True
    else:
        export_module.confirm_callback = typer.confirm
    try:
        report = hub.export(dest, dry_run=dry_run)
    except ExportError as exc:
        _fail(str(exc))
    except LoadError as exc:
        _fail(str(exc))
    if dry_run or yes:
        for hit in report.hits:
            typer.secho(f"warning: {hit}", fg=typer.colors.YELLOW, err=True)
    for rel in report.written:
        typer.echo(f"write  {rel}")
    for rel in report.deleted:
        typer.echo(f"delete {rel}")
    typer.secho(str(report), fg=typer.colors.GREEN)


# --- writes ----------------------------------------------------------------------


@app.command("new")
def new_cmd(
    type: str = typer.Argument(..., help="Memory type (from the profile vocabulary)."),
    title: str = typer.Option(..., "--title", prompt="Title"),
    description: str = typer.Option("", "--description", prompt="Description"),
    tags: str | None = typer.Option(None, "--tags", help="Comma-separated tags."),
) -> None:
    """Scaffold a new (draft) memory into the right folder; edit the file to fill in the body."""
    hub = _open_hub()
    with _guarded_write():
        doc = hub.add(
            type=type,
            title=title,
            description=description,
            tags=_split_csv(tags) or [],
            body=f"TODO: describe this {type}.",
        )
    typer.secho(
        f"Created {doc.id} at {doc.path} (status: {doc.frontmatter.status})",
        fg=typer.colors.GREEN,
    )
    extras = hub.profile.fields_for(type)
    if extras:
        typer.echo(f"Type-specific fields you can add: {', '.join(extras)}")


@app.command("add")
def add_cmd(
    type: str = typer.Option(..., "--type", help="Memory type (from the profile vocabulary)."),
    title: str = typer.Option(..., "--title"),
    description: str = typer.Option("", "--description"),
    tags: str | None = typer.Option(None, "--tags", help="Comma-separated tags."),
    status: str | None = typer.Option(
        None, "--status", help="Defaults to the profile's default status (else 'draft')."
    ),
    visibility: str | None = typer.Option(
        None, "--visibility", help="Defaults to the profile's default visibility (else 'private')."
    ),
    related: str | None = typer.Option(None, "--related", help="Comma-separated related ids."),
    set_: list[str] | None = typer.Option(
        None, "--set", help="Extra field as key=value (repeatable; YAML-typed values)."
    ),
    body_file: Path | None = typer.Option(None, "--body-file", help="Read the body from a file."),
) -> None:
    """Add a memory (validated before disk; atomic write)."""
    hub = _open_hub()
    fields = _parse_sets(set_)
    fields.update(type=type, title=title, description=description)
    if tags is not None:
        fields["tags"] = _split_csv(tags) or []
    if status is not None:
        fields["status"] = status
    if visibility is not None:
        fields["visibility"] = visibility
    if related is not None:
        fields["related"] = _split_csv(related) or []
    body = _read_body_file(body_file) or ""
    with _guarded_write():
        doc = hub.add(**fields, body=body)
    typer.secho(f"Created {doc.id} at {doc.path}", fg=typer.colors.GREEN)


@app.command("update")
def update_cmd(
    id: str = typer.Argument(..., help="Memory id."),
    set_: list[str] | None = typer.Option(
        None, "--set", help="Field as key=value (repeatable; YAML-typed values)."
    ),
    body_file: Path | None = typer.Option(
        None, "--body-file", help="Read the new body from a file."
    ),
) -> None:
    """Update a memory (bumps `updated`, re-validates, atomic write)."""
    hub = _open_hub()
    fields = _parse_sets(set_)
    body = _read_body_file(body_file)
    if not fields and body is None:
        _fail("nothing to update: pass --set and/or --body-file")
    with _guarded_write():
        doc = hub.update(id, **fields, body=body) if body is not None else hub.update(id, **fields)
    typer.secho(f"Updated {doc.id} at {doc.path}", fg=typer.colors.GREEN)


@app.command("rm")
def rm_cmd(id: str = typer.Argument(..., help="Memory id.")) -> None:
    """Soft-delete a memory (moves the file to `.trash/`)."""
    hub = _open_hub()
    with _guarded_write():
        hub.delete(id)
    typer.secho(f"Moved {id} to .trash/", fg=typer.colors.GREEN)


# --- MCP -------------------------------------------------------------------------


@app.command("mcp")
def mcp_cmd() -> None:
    """Serve the store to agents over MCP (stdio transport; blocks until the client hangs up).

    Write tools are only exposed when `allow_agent_writes = true` in hub.toml, and agent
    writes always land as private drafts. Requires the extra: pip install "memoryhub[mcp]".
    """
    try:
        from .mcp_server import build_server, warm_search_stack
    except ModuleNotFoundError as exc:
        _fail(f'MCP support is not installed ({exc}); run: pip install "memoryhub[mcp]"', code=2)
    hub = _open_hub()
    # Load the native search stack (numpy/lancedb/torch) here, on the main thread, before the
    # event loop starts — importing it lazily inside the first search deadlocks the stdio server
    # on the loader lock (see warm_search_stack). This is the entry point `hub mcp` actually runs.
    warm_search_stack(hub)
    build_server(hub).run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    app()
