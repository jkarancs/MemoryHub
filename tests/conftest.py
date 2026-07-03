"""Shared pytest fixtures for the MemoryHub test suite."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from memoryhub import writer

HUB_TOML = """\
[hub]
name = "test"
content_root = "./memory"
profile = "personal"

[write]
allow_agent_writes = true
require_confirmation = false

[embeddings]
backend = "local"
[embeddings.local]
model = "BAAI/bge-m3"
device = "cuda"
[embeddings.api]
base_url = "https://openrouter.ai/api/v1"
model = "openai/text-embedding-3-large"
api_key_env = "OPENROUTER_API_KEY"

[index]
backend = "lancedb"
path = ".index"
"""


def memory_text(
    *,
    id: str,
    type: str,
    title: str | None = None,
    description: str = "A description.",
    tags: str = "[]",
    status: str = "active",
    visibility: str = "private",
    created: str = "2026-01-01",
    updated: str = "2026-01-02",
    related: str = "[]",
    source: str = "self",
    body: str = "Some body text.",
    extras: dict[str, Any] | None = None,
) -> str:
    """Hand-rolled memory file text (list-ish values are passed as raw YAML strings)."""
    lines = [
        "---",
        f"id: {id}",
        f"title: {title or id}",
        f"type: {type}",
        f"description: {description}",
        f"tags: {tags}",
        f"status: {status}",
        f"visibility: {visibility}",
        f"created: {created}",
        f"updated: {updated}",
    ]
    for key, value in (extras or {}).items():
        lines.append(f"{key}: {value}")
    lines += [f"related: {related}", f"source: {source}", "---", "", body, ""]
    return "\n".join(lines)


def write_memory(repo: Path, **kwargs: Any) -> Path:
    """Write a memory file into ``repo/memory/<type>/<id>.md`` and return its path."""
    path = repo / "memory" / kwargs["type"] / f"{kwargs['id']}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(memory_text(**kwargs), encoding="utf-8")
    return path


@pytest.fixture
def content_repo(tmp_path: Path) -> Path:
    """A minimal temp content repo: a hub.toml plus an (empty) memory/ tree."""
    (tmp_path / "hub.toml").write_text(HUB_TOML, encoding="utf-8")
    (tmp_path / "memory").mkdir()
    return tmp_path


@pytest.fixture
def seeded_repo(content_repo: Path) -> Path:
    """A ~10-file valid store covering every type, with varied tags/statuses/dates."""
    write_memory(
        content_repo,
        id="bio-me",
        type="bio",
        title="About me",
        tags="[bio]",
        visibility="public",
    )
    write_memory(
        content_repo,
        id="skill-async-python",
        type="skill",
        title="Async Python",
        description="asyncio and structured concurrency in production services",
        tags="[python, async]",
        updated="2026-03-01",
        related="[project-hub]",
        extras={"proficiency": "advanced"},
        body="Deep experience with asyncio event loops and asyncio task groups.",
    )
    write_memory(
        content_repo,
        id="skill-sql",
        type="skill",
        title="SQL",
        tags="[data, sql]",
        extras={"proficiency": "intermediate"},
    )
    write_memory(
        content_repo,
        id="skill-docker",
        type="skill",
        title="Docker",
        tags="[devops]",
        extras={"proficiency": "beginner"},
    )
    write_memory(
        content_repo,
        id="project-hub",
        type="project",
        title="MemoryHub",
        tags="[python, tooling]",
        updated="2026-02-01",
        related="[skill-async-python]",
        extras={"stack": "[python, typer]", "url": "https://example.com", "role": "author"},
        body="A markdown memory engine; the loader uses asyncio nowhere, but this line does.",
    )
    write_memory(
        content_repo,
        id="project-site",
        type="project",
        title="Personal site",
        tags="[web]",
        visibility="public",
        extras={"stack": "[astro]", "role": "author"},
    )
    write_memory(
        content_repo,
        id="goal-ai-role",
        type="goal",
        title="Land an AI engineering role",
        tags="[career]",
        status="draft",
        extras={"horizon": "2026", "blocking_skills": "[skill-sql]"},
    )
    write_memory(
        content_repo,
        id="preference-remote",
        type="preference",
        title="Remote-first work",
        tags="[work-style]",
    )
    write_memory(
        content_repo,
        id="education-phd",
        type="education",
        title="PhD in Physics",
        tags="[physics]",
        extras={
            "institution": "Some University",
            "degree": "PhD",
            "field": "physics",
            "year": 2020,
        },
    )
    write_memory(
        content_repo,
        id="experience-acme",
        type="experience",
        title="Engineer at Acme",
        tags="[python, backend]",
        extras={
            "org": "Acme",
            "role": "Engineer",
            "start": "2021-01-01",
            "end": "2024-01-01",
            "stack": "[python]",
            "highlights": "[shipped the ingestion service, cut costs 40 percent]",
        },
    )
    write_memory(
        content_repo,
        id="writing-blog-post",
        type="writing",
        title="On typed Python",
        tags="[python, writing]",
        status="archived",
        visibility="public",
    )
    return content_repo


@pytest.fixture(autouse=True)
def _reset_confirm_callback() -> Iterator[None]:
    """Keep the module-level confirmation hook from leaking between tests."""
    yield
    writer.confirm_callback = None


def block_module(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    """Make ``import <missing>`` raise ModuleNotFoundError even if the package is installed."""
    import builtins

    real_import = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == missing:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


class TopicEmbedder:
    """Deterministic fake embedder: each topic is an axis, keyed by a group of synonyms.

    Texts mentioning words from the same group get parallel vectors — that's the "semantic"
    matching the hybrid-search tests rely on (e.g. a doc saying only "lancedb" is nearest to
    the query "vector databases"). Texts with no topic share a junk axis. Records everything
    it embeds so incremental-reindex tests can count embeddings.
    """

    topics = (
        ("vector", "lancedb", "embedding"),
        ("async", "concurrency"),
        ("docker", "container"),
        ("physics",),
    )

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = [float(sum(lowered.count(word) for word in group)) for group in self.topics]
            vector.append(0.0)
            if not any(vector):
                vector[-1] = 1.0
            vectors.append(vector)
        return vectors
