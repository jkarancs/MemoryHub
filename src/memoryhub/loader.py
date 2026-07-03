"""Parse/serialize markdown + frontmatter and validate against models + profile.

Serialization is Windows-safe by construction: output is always UTF-8 text with ``\\n`` newlines
(whatever the platform or the input file's line endings) and frontmatter keys are emitted in a
stable order — the profile's ``common_required`` order first, then type-specific fields, then
``related``/``source`` — so agent writes produce minimal diffs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import frontmatter as frontmatter_lib
import yaml
from pydantic import ValidationError

from .models import Frontmatter, MemoryDoc, validate_against_profile
from .profiles import Profile, load_profile

if TYPE_CHECKING:
    from .config import Config

#: Common (model-level) fields in canonical serialization order, excluding the trailing pair.
_COMMON_FIELDS = (
    "id",
    "title",
    "type",
    "description",
    "tags",
    "status",
    "visibility",
    "created",
    "updated",
)
#: Always emitted last, per the stable-ordering contract.
_TRAILING_FIELDS = ("related", "source")
#: Flat frontmatter keys that map onto Frontmatter model fields (everything else is `extra`).
_MODEL_FIELDS = frozenset(_COMMON_FIELDS) | frozenset(_TRAILING_FIELDS)


@dataclass(frozen=True)
class ValidationIssue:
    """One problem found while loading/validating a store: which file, which field, why."""

    path: Path | None
    field: str
    reason: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "path": str(self.path) if self.path is not None else None,
            "field": self.field,
            "reason": self.reason,
        }

    def __str__(self) -> str:
        where = str(self.path) if self.path is not None else "(store)"
        return f"{where}: {self.field}: {self.reason}"


@dataclass
class StoreReport:
    """Aggregated result of validating a whole store (consumed by ``hub validate``)."""

    issues: list[ValidationIssue]
    warnings: list[ValidationIssue]
    checked: int

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.ok,
            "checked": self.checked,
            "issues": [issue.to_dict() for issue in self.issues],
            "warnings": [warning.to_dict() for warning in self.warnings],
        }


class LoadError(ValueError):
    """Aggregated validation report raised when loading a store surfaces one or more problems."""

    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = list(issues)
        lines = [str(issue) for issue in self.issues]
        super().__init__(
            f"{len(self.issues)} validation problem(s):\n" + "\n".join(f"  {ln}" for ln in lines)
        )


def split_fields(mapping: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split flat frontmatter keys into (model fields, type-specific extras)."""
    known = {k: v for k, v in mapping.items() if k in _MODEL_FIELDS}
    extra = {k: v for k, v in mapping.items() if k not in _MODEL_FIELDS}
    return known, extra


def _normalize_body(body: str) -> str:
    """LF-only line endings, no leading/trailing blank lines, exactly one trailing newline."""
    text = body.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    return text + "\n" if text else ""


def _build_doc(
    text: str, path: Path | None, profile: Profile | None
) -> tuple[MemoryDoc | None, list[ValidationIssue]]:
    """Parse one file's text into a doc, collecting (not raising) every problem found."""
    try:
        post = frontmatter_lib.loads(text)
    except yaml.YAMLError as exc:
        return None, [ValidationIssue(path, "frontmatter", f"invalid YAML frontmatter: {exc}")]

    metadata = dict(post.metadata)
    if not metadata:
        return None, [ValidationIssue(path, "frontmatter", "missing frontmatter block")]

    known, extra = split_fields(metadata)
    try:
        fm = Frontmatter(**known, extra=extra)
    except ValidationError as exc:
        issues = [
            ValidationIssue(
                path,
                ".".join(str(part) for part in err["loc"]) or "frontmatter",
                err["msg"],
            )
            for err in exc.errors()
        ]
        return None, issues

    issues = []
    if profile is not None:
        problems = validate_against_profile(fm, profile)
        issues = [ValidationIssue(path, "frontmatter", problem) for problem in problems]
    if issues:
        return None, issues

    doc = MemoryDoc(frontmatter=fm, body=_normalize_body(post.content), path=path)
    return doc, []


def iter_store_paths(content_root: Path) -> list[Path]:
    """All memory files under ``content_root``, skipping dot-directories (``.trash`` etc.)."""
    if not content_root.is_dir():
        return []
    paths = []
    for path in sorted(content_root.rglob("*.md")):
        rel = path.relative_to(content_root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        paths.append(path)
    return paths


def _load_store(config: Config) -> tuple[list[MemoryDoc], list[ValidationIssue], int]:
    """Load every file, collecting docs + issues (duplicate ids included). Never raises."""
    profile = load_profile(config.hub.profile)
    root = config.content_root
    if not root.is_dir():
        issue = ValidationIssue(None, "content_root", f"content root {root} does not exist")
        return [], [issue], 0

    docs: list[MemoryDoc] = []
    issues: list[ValidationIssue] = []
    seen: dict[str, Path | None] = {}
    paths = iter_store_paths(root)
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            issues.append(ValidationIssue(path, "file", f"unreadable: {exc}"))
            continue
        doc, file_issues = _build_doc(text, path, profile)
        if file_issues:
            issues.extend(file_issues)
            continue
        assert doc is not None
        if doc.id in seen:
            issues.append(
                ValidationIssue(path, "id", f"duplicate id {doc.id!r} (also in {seen[doc.id]})")
            )
            continue
        seen[doc.id] = path
        docs.append(doc)
    return docs, issues, len(paths)


def load_all(config: Config) -> list[MemoryDoc]:
    """Walk ``config.content_root``, parse+validate every ``.md`` file, attach its path.

    Collects all errors and raises a single aggregated :class:`LoadError` (file + field + reason)
    rather than failing on the first bad file.
    """
    docs, issues, _ = _load_store(config)
    if issues:
        raise LoadError(issues)
    return docs


def validate_store(config: Config) -> StoreReport:
    """Validate the whole store without raising; unresolved ``related`` ids are warnings.

    Per the schema rules, ``related`` ids that don't resolve warn (forward refs are allowed)
    while everything else — including duplicate ids — is a hard issue.
    """
    docs, issues, checked = _load_store(config)
    known_ids = {doc.id for doc in docs}
    warnings = [
        ValidationIssue(doc.path, "related", f"related id {ref!r} does not resolve")
        for doc in docs
        for ref in doc.frontmatter.related
        if ref not in known_ids
    ]
    return StoreReport(issues=issues, warnings=warnings, checked=checked)


def load_one(path: str | Path, profile: Profile | None = None) -> MemoryDoc:
    """Parse and validate a single memory file.

    Profile-dependent checks (type vocabulary, allowed extras) run only when ``profile`` is given;
    store-level checks (id uniqueness, related resolution) need :func:`load_all`.
    """
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    doc, issues = _build_doc(text, file_path, profile)
    if issues:
        raise LoadError(issues)
    assert doc is not None
    return doc


def flat_frontmatter(fm: Frontmatter, profile: Profile | None = None) -> dict[str, Any]:
    """The frontmatter as a flat, stably-ordered mapping (as written to files).

    Order: profile ``common_required`` first (falling back to the canonical common-field order),
    then type-specific extras in profile declaration order, then ``related``/``source``.
    """
    order: list[str] = []
    candidates = list(profile.common_required) if profile is not None else []
    for key in [*candidates, *_COMMON_FIELDS]:
        if key in _MODEL_FIELDS and key not in _TRAILING_FIELDS and key not in order:
            order.append(key)

    data: dict[str, Any] = {key: getattr(fm, key) for key in order}

    extra_order = [f for f in (profile.fields_for(fm.type) if profile else []) if f in fm.extra]
    extra_order += [k for k in fm.extra if k not in extra_order]
    for key in extra_order:
        data[key] = fm.extra[key]

    data["related"] = fm.related
    data["source"] = fm.source
    return data


def serialize(doc: MemoryDoc, profile: Profile | None = None) -> str:
    """Round-trip a :class:`MemoryDoc` back to ``frontmatter + body`` with stable key ordering.

    Always UTF-8-safe text with ``\\n`` newlines regardless of platform or input line endings.
    """
    data = flat_frontmatter(doc.frontmatter, profile)
    yaml_text = yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=None,  # scalar lists inline: `tags: [python, async]`
        width=1000,  # don't wrap long values; wrapping churns diffs
    )
    body = _normalize_body(doc.body)
    text = f"---\n{yaml_text}---\n"
    if body:
        text += f"\n{body}"
    return text
