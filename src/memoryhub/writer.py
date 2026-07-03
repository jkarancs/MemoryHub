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
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import frontmatter as frontmatter_lib
from pydantic import ValidationError

from . import ids as ids_module
from .loader import iter_store_paths, load_one, serialize, split_fields
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
    is suffixed to uniqueness. ``status`` defaults to ``draft`` and ``visibility`` to ``private``
    (safe defaults — flipping to active/public is a deliberate act).
    """
    _check_policy(config, "add")
    profile = load_profile(config.hub.profile)

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
    supplied.setdefault("status", "draft")
    supplied.setdefault("visibility", "private")
    supplied.setdefault("description", "")

    known, extra = split_fields(supplied)
    fm = _validate_or_raise(known, extra, profile)
    _warn_unresolved_related(fm, set(existing) | {new_id})

    path = _confine(config, config.content_root / type_ / f"{new_id}.md")
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
    sets it explicitly.
    """
    _check_policy(config, f"update {id!r}")
    profile = load_profile(config.hub.profile)

    path = _find_existing(config, id)
    doc = load_one(path, profile)

    patch = dict(fields or {})
    if "id" in patch and patch["id"] != doc.frontmatter.id:
        raise WriteError("changing 'id' is not supported (delete and re-add instead)")
    if "type" in patch and patch["type"] != doc.frontmatter.type:
        raise WriteError("changing 'type' is not supported (the file would have to move)")

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
