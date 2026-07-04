"""``hub export`` — deterministic sync of public content into a public store.

Selection is strict: only ``visibility: public`` **and** ``status: active`` docs export
(drafts/archived/aspirational never do, whatever their visibility). Exported frontmatter is kept
identical except that ``related`` entries pointing at non-exported ids are stripped (with an
inline YAML comment noting how many); bodies pass through untouched. The destination therefore
remains a valid, read-only hub store of its own.

The sync is a deterministic full sync: stable key ordering and ``\\n`` newlines (via
:func:`memoryhub.loader.serialize`), files in the destination that left the export set are
deleted, and running the export twice yields zero diff. A root ``README.md`` index (grouped by
type) is regenerated on every run, and a minimal read-only ``hub.toml`` is created once if the
destination doesn't have one.

Safety gates: the export refuses to run while ``hub validate`` fails, refuses destinations that
overlap the source store, and scans exported text for emails / phone numbers / API-key shapes —
hits require confirmation (personal contact details in a ``bio`` may be intentional, hence
warn-and-confirm rather than block).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from . import loader
from .models import MemoryDoc
from .profiles import Profile, load_profile

if TYPE_CHECKING:
    from .config import Config


class ExportError(RuntimeError):
    """Raised when an export is refused (invalid store, unsafe destination, unconfirmed hits)."""


#: Optional confirmation hook for the sensitive-content scan (the CLI wires this to a prompt).
#: When the scan hits and no callback is registered, the export refuses rather than publishing.
confirm_callback: Callable[[str], bool] | None = None

#: Shapes worth a second look before publishing. Deliberately warn-not-block precision:
#: a false positive costs one confirmation, a false negative publishes a secret.
_SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "email address",
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
    ),
    ("phone number", re.compile(r"(?<![\w.+-])\+\d[\d ().-]{7,}\d")),
    (
        "API key",
        re.compile(
            r"\bsk-[A-Za-z0-9_-]{20,}"  # OpenAI / OpenRouter (sk-or-...)
            r"|\bAKIA[0-9A-Z]{16}\b"  # AWS access key id
            r"|\bgh[pousr]_[A-Za-z0-9]{36,}"  # GitHub tokens
            r"|\bxox[abprs]-[A-Za-z0-9-]{10,}"  # Slack tokens
            r"|\bAIza[0-9A-Za-z_-]{35}"  # Google API keys
        ),
    ),
    (
        "secret assignment",
        re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*['\"]?[^\s'\"]{8,}"),
    ),
)


@dataclass(frozen=True)
class ScanHit:
    """One sensitive-content match in a doc queued for export."""

    id: str
    kind: str
    excerpt: str

    def __str__(self) -> str:
        return f"{self.id}: possible {self.kind}: {self.excerpt!r}"


@dataclass
class ExportReport:
    """Outcome of one export run; paths are relative to the destination root."""

    dest: Path
    written: list[Path] = field(default_factory=list)
    deleted: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)
    hits: list[ScanHit] = field(default_factory=list)
    dry_run: bool = False

    @property
    def exported(self) -> int:
        """Number of files in the export set (written or already up to date)."""
        return len(self.written) + len(self.unchanged)

    def __str__(self) -> str:
        summary = (
            f"export -> {self.dest}: {len(self.written)} written, "
            f"{len(self.deleted)} deleted, {len(self.unchanged)} unchanged"
        )
        return f"{summary} (dry run - nothing touched)" if self.dry_run else summary


def _select(docs: list[MemoryDoc]) -> list[MemoryDoc]:
    """The export set: public **and** active, sorted by id for stable ordering."""
    keep = [
        doc
        for doc in docs
        if doc.frontmatter.visibility == "public" and doc.frontmatter.status == "active"
    ]
    return sorted(keep, key=lambda doc: doc.id)


def _annotate_related(text: str, note: str) -> str:
    """Append an inline YAML comment to the ``related:`` line of the frontmatter block."""
    lines = text.split("\n")
    for i, line in enumerate(lines[1:], start=1):
        if line == "---":  # end of frontmatter; body is never touched
            break
        if line.startswith("related:"):
            lines[i] = f"{line}  {note}"
            break
    return "\n".join(lines)


def _render(doc: MemoryDoc, exported_ids: set[str], profile: Profile) -> str:
    """Serialize ``doc`` for export: prune non-exported ``related`` ids, note the count."""
    kept = [ref for ref in doc.frontmatter.related if ref in exported_ids]
    stripped = len(doc.frontmatter.related) - len(kept)
    fm = doc.frontmatter.model_copy(update={"related": kept})
    text = loader.serialize(MemoryDoc(frontmatter=fm, body=doc.body), profile)
    if stripped:
        note = f"# {stripped} link(s) to non-exported memories removed by `hub export`"
        text = _annotate_related(text, note)
    return text


def _scan(doc_id: str, text: str) -> list[ScanHit]:
    hits = []
    for kind, pattern in _SENSITIVE_PATTERNS:
        for match in pattern.finditer(text):
            hits.append(ScanHit(id=doc_id, kind=kind, excerpt=match.group(0)[:60]))
    return hits


def _readme(docs: list[MemoryDoc], profile: Profile, hub_name: str, content_dir: str) -> str:
    """A generated table-of-contents README, grouped by type in profile order."""
    lines = [
        f"# {hub_name} — public memory export",
        "",
        "**This repository is generated by `hub export` — do not edit by hand.** It contains",
        "the `visibility: public`, `status: active` subset of a private MemoryHub store and is",
        "itself a valid read-only hub store. Manual edits are overwritten on the next export.",
        "",
        f"{len(docs)} memories.",
    ]
    by_type: dict[str, list[MemoryDoc]] = {}
    for doc in docs:
        by_type.setdefault(doc.type, []).append(doc)
    for type_name in profile.type_names:
        group = by_type.get(type_name)
        if not group:
            continue
        lines += ["", f"## {type_name} ({len(group)})", ""]
        for doc in sorted(group, key=lambda d: d.id):
            link = PurePosixPath(content_dir) / type_name / f"{doc.id}.md"
            entry = f"- [{doc.frontmatter.title}]({link})"
            if doc.frontmatter.description:
                entry += f" — {doc.frontmatter.description}"
            lines.append(entry)
    return "\n".join(lines) + "\n"


def _minimal_hub_toml(config: Config) -> str:
    return (
        "# hub.toml — created by `hub export`; this repo is a read-only export target.\n"
        "[hub]\n"
        f'name = "{config.hub.name}-public"\n'
        f'content_root = "{config.hub.content_root}"\n'
        f'profile = "{config.hub.profile}"\n'
        "\n"
        "[write]\n"
        "allow_agent_writes = false\n"
    )


def _dest_content_root(config: Config, dest: Path) -> Path:
    """Mirror the source's relative content root under ``dest`` (fallback: ``memory``)."""
    rel = Path(config.hub.content_root)
    if rel.is_absolute() or ".." in rel.parts or rel == Path("."):
        return dest / "memory"
    return dest / rel


def _check_dest(config: Config, dest: Path, dest_root: Path) -> None:
    src_root = config.content_root
    if (
        dest_root == src_root
        or dest_root.is_relative_to(src_root)
        or src_root.is_relative_to(dest_root)
        or (config.root is not None and dest == config.root.resolve())
    ):
        raise ExportError(f"refusing to export into {dest}: it overlaps the source store")


def _confirm_or_refuse(hits: list[ScanHit]) -> None:
    listing = "\n".join(f"  {hit}" for hit in hits)
    if confirm_callback is None:
        raise ExportError(
            f"refusing to export: {len(hits)} possible sensitive item(s) in public content "
            f"and no confirmation is available:\n{listing}"
        )
    prompt = (
        f"{len(hits)} possible sensitive item(s) would be published:\n{listing}\n" "Export anyway?"
    )
    if not confirm_callback(prompt):
        raise ExportError("export aborted: sensitive-content hits were not confirmed")


def export_store(config: Config, dest: str | Path, *, dry_run: bool = False) -> ExportReport:
    """Sync the public subset of the store into ``dest``; see the module docstring for rules.

    Returns an :class:`ExportReport`; with ``dry_run=True`` the report describes what would
    change and nothing is touched (the sensitive scan still runs but does not prompt).

    Raises:
        ExportError: if the store fails validation, ``dest`` overlaps the source store, or
            sensitive-content hits are not confirmed.
    """
    dest = Path(dest).resolve()
    dest_root = _dest_content_root(config, dest)
    _check_dest(config, dest, dest_root)

    validation = loader.validate_store(config)
    if not validation.ok:
        raise ExportError(
            f"refusing to export: the store has {len(validation.issues)} validation "
            "problem(s); run `hub validate` and fix them first"
        )

    profile = load_profile(config.hub.profile)
    selected = _select(loader.load_all(config))
    exported_ids = {doc.id for doc in selected}
    content_dir = dest_root.relative_to(dest).as_posix()

    # Plan every managed file (dest-relative path -> content) before touching anything.
    planned: dict[Path, str] = {}
    report = ExportReport(dest=dest, dry_run=dry_run)
    for doc in selected:
        text = _render(doc, exported_ids, profile)
        planned[Path(content_dir) / doc.type / f"{doc.id}.md"] = text
        report.hits.extend(_scan(doc.id, text))
    planned[Path("README.md")] = _readme(selected, profile, config.hub.name, content_dir)

    stale = [
        path.relative_to(dest)
        for path in loader.iter_store_paths(dest_root)
        if path.relative_to(dest) not in planned
    ]

    if report.hits and not dry_run:
        _confirm_or_refuse(report.hits)

    for rel, text in sorted(planned.items()):
        target = dest / rel
        try:
            current: str | None = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            current = None
        if current == text:
            report.unchanged.append(rel)
            continue
        report.written.append(rel)
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", newline="\n")

    for rel in sorted(stale):
        report.deleted.append(rel)
        if not dry_run:
            (dest / rel).unlink()

    if not dry_run:
        _prune_empty_dirs(dest_root)
        toml_path = dest / "hub.toml"
        if not toml_path.exists():
            toml_path.write_text(_minimal_hub_toml(config), encoding="utf-8", newline="\n")

    return report


def _prune_empty_dirs(dest_root: Path) -> None:
    """Drop now-empty type directories left behind by deletions (never ``dest_root`` itself)."""
    if not dest_root.is_dir():
        return
    subdirs = [p for p in dest_root.rglob("*") if p.is_dir()]
    for directory in sorted(subdirs, key=lambda p: len(p.parts), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()
