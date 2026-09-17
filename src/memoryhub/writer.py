"""Add/update/delete memory files: atomic, validated-before-disk, and guarded.

Design invariants:
  * validate everything **before** touching disk; on failure, no file changes;
  * atomic writes (temp file + ``os.replace``, same directory); single-file writes only;
  * paths confined under ``content_root`` (reject traversal);
  * refuse writes when ``config.write.allow_agent_writes`` is false;
  * ``related`` ids that don't resolve **warn** (:class:`WriteWarning`); duplicate explicit ids
    hard-fail; generated ids get a ``-2``/``-3`` suffix instead.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import frontmatter as frontmatter_lib
from pydantic import ValidationError

from . import ids as ids_module
from .loader import flat_frontmatter, iter_store_paths, load_all, load_one, serialize, split_fields
from .models import Frontmatter, MemoryDoc, validate_against_profile
from .profiles import Profile, load_profile

if TYPE_CHECKING:
    from .config import Config


class WriteError(RuntimeError):
    """Raised when a write is refused (policy/guardrail) or would violate an invariant."""


class WriteWarning(UserWarning):
    """Non-fatal problem with an otherwise valid write (e.g. an unresolved ``related`` id)."""


#: Optional confirmation hook for interactive use (the CLI wires this to a prompt). When
#: ``config.write.require_confirmation`` is true and no callback is registered, writes refuse.
confirm_callback: Callable[[str], bool] | None = None


def _check_policy(config: Config, action: str) -> None:
    if not config.write.allow_agent_writes:
        raise WriteError(f"refusing to {action}: allow_agent_writes is false in hub.toml")
    if config.write.require_confirmation:
        if confirm_callback is None:
            raise WriteError(
                f"refusing to {action}: require_confirmation is set but no confirmer is available"
            )
        if not confirm_callback(f"Confirm {action}?"):
            raise WriteError(f"{action} aborted: not confirmed")


def canonical_relpath(profile: Profile, fields: Mapping[str, Any]) -> Path:
    """Where a document belongs relative to ``content_root``: ``<type>/[<subfolder>/]<id>.md``.

    The subfolder comes from the profile's ``layout`` rule for the type (:class:`.LayoutRule`);
    a profile that declares none puts every document straight in its type folder, exactly as
    before layout existed.
    """
    type_ = str(fields["type"])
    subfolder = profile.subfolder_for(type_, fields)
    parent = Path(type_) / subfolder if subfolder else Path(type_)
    return parent / f"{fields['id']}.md"


def _confine(config: Config, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(config.content_root):
        raise WriteError(f"path {resolved} escapes content_root {config.content_root}")
    return resolved


def scan_ids(config: Config) -> dict[str, Path]:
    """Cheap id → path scan of the store (frontmatter ``id`` if parseable, else the file stem).

    Deliberately tolerant: a broken file elsewhere in the store must not block writes; full
    validation belongs to the loader.
    """
    found: dict[str, Path] = {}
    for path in iter_store_paths(config.content_root):
        doc_id = path.stem
        try:
            metadata = frontmatter_lib.loads(path.read_text(encoding="utf-8")).metadata
            candidate = metadata.get("id")
            if isinstance(candidate, str) and candidate:
                doc_id = candidate
        except Exception:  # noqa: BLE001 - fall back to the stem for unparseable files
            pass
        found.setdefault(doc_id, path)
    return found


def _validate_or_raise(
    fields: dict[str, Any], extra: dict[str, Any], profile: Profile
) -> Frontmatter:
    try:
        fm = Frontmatter(**fields, extra=extra)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'frontmatter'}: {err['msg']}"
            for err in exc.errors()
        )
        raise WriteError(f"invalid frontmatter: {details}") from exc
    problems = validate_against_profile(fm, profile)
    if problems:
        raise WriteError("invalid frontmatter: " + "; ".join(problems))
    return fm


def _warn_unresolved_related(fm: Frontmatter, known_ids: set[str]) -> None:
    for ref in fm.related:
        if ref not in known_ids:
            warnings.warn(
                f"related id {ref!r} does not resolve to a memory in the store",
                WriteWarning,
                stacklevel=3,
            )


def _atomic_write(path: Path, text: str) -> None:
    """Write UTF-8 text with LF newlines via a same-directory temp file + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def add(config: Config, frontmatter_fields: dict[str, Any], body: str) -> MemoryDoc:
    """Create a new memory file (generate/validate id, set created/updated, atomic write).

    An explicit ``id`` that already exists hard-fails; a generated one (from ``type`` + ``title``)
    is suffixed to uniqueness. ``status``/``visibility`` fall back to the profile's ``defaults``,
    and without those to ``draft``/``private`` (safe defaults — flipping to active/public is a
    deliberate act).
    """
    _check_policy(config, "add")
    profile = load_profile(config.profile_ref)

    supplied = dict(frontmatter_fields)
    type_ = supplied.get("type")
    if not isinstance(type_, str) or not profile.is_known_type(type_):
        raise WriteError(
            f"add requires a 'type' from the profile vocabulary ({', '.join(profile.type_names)})"
        )
    title = supplied.get("title")
    if not isinstance(title, str) or not title.strip():
        raise WriteError("add requires a non-empty 'title'")

    existing = scan_ids(config)
    if supplied.get("id"):
        new_id = supplied["id"]
        if new_id in existing:
            raise WriteError(f"duplicate id {new_id!r} (already at {existing[new_id]})")
    else:
        new_id = ids_module.ensure_unique(ids_module.slugify(title, type_), existing)
    supplied["id"] = new_id

    today = date.today()
    supplied.setdefault("created", today)
    supplied.setdefault("updated", today)
    supplied.setdefault("status", profile.defaults.get("status", "draft"))
    supplied.setdefault("visibility", profile.defaults.get("visibility", "private"))
    supplied.setdefault("description", "")

    known, extra = split_fields(supplied)
    fm = _validate_or_raise(known, extra, profile)
    _warn_unresolved_related(fm, set(existing) | {new_id})

    path = _confine(config, config.content_root / canonical_relpath(profile, supplied))
    if path.exists():
        raise WriteError(f"refusing to overwrite existing file {path}")
    doc = MemoryDoc(frontmatter=fm, body=body, path=path)
    _atomic_write(path, serialize(doc, profile))
    return doc


def _find_existing(config: Config, id: str) -> Path:
    existing = scan_ids(config)
    if id not in existing:
        raise WriteError(f"no memory with id {id!r}")
    return existing[id]


#: Patch-only key (never written to disk). CLI ``--allow-subnode-removal`` threads it here.
_ALLOW_SUBNODE_REMOVAL = "allow_subnode_removal"


def _refuse_dropped_subnodes(doc_id: str, on_disk: Any, incoming: Any) -> None:
    """Refuse a ``subnodes`` replacement that omits an id already present on disk."""
    if not isinstance(on_disk, list):
        return
    kept = set(incoming) if isinstance(incoming, list) else set()
    dropped = [sid for sid in on_disk if isinstance(sid, str) and sid not in kept]
    if not dropped:
        return
    n = len(dropped)
    noun = "id" if n == 1 else "ids"
    include = "it" if n == 1 else "them"
    raise WriteError(
        f"{doc_id}: subnodes drops {n} {noun} present on disk: {', '.join(dropped)} — "
        f"re-read the node and include {include} (the list replaces, it does not append)"
    )


def update(
    config: Config,
    id: str,
    *,
    fields: dict[str, Any] | None = None,
    body: str | None = None,
) -> MemoryDoc:
    """Load, patch, bump ``updated``, re-validate, and atomically write.

    ``id`` and ``type`` may not change (a rename/move is not a single-file write). Setting a
    type-specific field to ``None`` removes it. ``updated`` is bumped to today unless the patch
    sets it explicitly. A ``subnodes`` patch that omits an id already on disk is refused unless
    ``allow_subnode_removal`` is true in the patch (CLI ``--allow-subnode-removal``); that key is
    never written to the file.
    """
    _check_policy(config, f"update {id!r}")
    profile = load_profile(config.profile_ref)

    path = _find_existing(config, id)
    doc = load_one(path, profile)

    patch = dict(fields or {})
    allow_subnode_removal = bool(patch.pop(_ALLOW_SUBNODE_REMOVAL, False))
    if "id" in patch and patch["id"] != doc.frontmatter.id:
        raise WriteError("changing 'id' is not supported (delete and re-add instead)")
    if "type" in patch and patch["type"] != doc.frontmatter.type:
        raise WriteError("changing 'type' is not supported (the file would have to move)")
    if "subnodes" in patch and not allow_subnode_removal:
        _refuse_dropped_subnodes(id, doc.frontmatter.extra.get("subnodes"), patch["subnodes"])

    known_patch, extra_patch = split_fields(patch)
    known = doc.frontmatter.model_dump(exclude={"extra"})
    known.update(known_patch)
    if "updated" not in known_patch:
        known["updated"] = date.today()

    extra = dict(doc.frontmatter.extra)
    for key, value in extra_patch.items():
        if value is None:
            extra.pop(key, None)
        else:
            extra[key] = value

    fm = _validate_or_raise(known, extra, profile)
    _warn_unresolved_related(fm, set(scan_ids(config)))

    new_body = doc.body if body is None else body
    updated_doc = MemoryDoc(frontmatter=fm, body=new_body, path=doc.path)
    assert doc.path is not None
    _atomic_write(_confine(config, doc.path), serialize(updated_doc, profile))
    return updated_doc


def delete(config: Config, id: str) -> None:
    """Soft-delete: move the file into ``content_root/.trash/`` rather than hard-removing."""
    _check_policy(config, f"delete {id!r}")
    path = _find_existing(config, id)
    trash = config.content_root / ".trash"
    trash.mkdir(parents=True, exist_ok=True)
    dest = trash / path.name
    if dest.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = trash / f"{path.stem}-{stamp}{path.suffix}"
    os.replace(path, _confine(config, dest))


@dataclass(frozen=True)
class Move:
    """One file that is not at its canonical path (paths are relative to ``content_root``)."""

    id: str
    src: Path
    dest: Path

    def __str__(self) -> str:
        return f"{self.src.as_posix()} -> {self.dest.as_posix()}"


@dataclass
class RelayoutReport:
    """What :func:`relayout` found, and whether it was applied."""

    moves: list[Move] = field(default_factory=list)
    unchanged: int = 0
    applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "unchanged": self.unchanged,
            "moves": [
                {"id": m.id, "src": m.src.as_posix(), "dest": m.dest.as_posix()} for m in self.moves
            ],
        }

    def __str__(self) -> str:
        verb = "Moved" if self.applied else "Would move"
        return f"{verb} {len(self.moves)} file(s); {self.unchanged} already canonical."


def _prune_empty_dirs(root: Path) -> None:
    """Remove directories the moves emptied (deepest first); dot-directories are left alone."""
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts) or not path.is_dir():
            continue
        with contextlib.suppress(OSError):
            path.rmdir()


def relayout(config: Config, *, apply: bool = False) -> RelayoutReport:
    """Move existing files to the canonical paths the profile's ``layout`` rules imply.

    Dry-run by default: nothing is touched unless ``apply`` is true. Idempotent by construction
    — a file already at its canonical path is never moved, so a second run reports no moves.
    Planning is complete before the first rename, so a refusal leaves the store untouched.

    Raises:
        WriteError: a target path is already occupied by another file.
        LoadError: the store does not parse/validate — including the id collision that is the
            only way two documents can claim one canonical path (the filename *is* the id, so
            duplicate ids are what a collision looks like, and the loader reports them).
    """
    if apply:
        _check_policy(config, "relayout")
    profile = load_profile(config.profile_ref)
    root = config.content_root

    report = RelayoutReport(applied=False)
    for doc in sorted(load_all(config), key=lambda d: d.id):
        assert doc.path is not None
        src = doc.path.resolve()
        fields = flat_frontmatter(doc.frontmatter, profile)
        dest = _confine(config, root / canonical_relpath(profile, fields))
        if dest == src:
            report.unchanged += 1
        else:
            report.moves.append(Move(doc.id, src.relative_to(root), dest.relative_to(root)))

    for move in report.moves:
        if (root / move.dest).exists():
            raise WriteError(
                f"refusing to relayout: {move.dest.as_posix()} already exists and is not the "
                f"file {move.id!r} lives in"
            )

    if apply and report.moves:
        for move in report.moves:
            target = root / move.dest
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(root / move.src, target)
        _prune_empty_dirs(root)
    report.applied = apply
    return report
