"""Hub facade tests: reads, writes, and cache invalidation (write + mtime)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from conftest import memory_text
from memoryhub import Hub, LoadError, load_config


@pytest.fixture
def hub(seeded_repo: Path) -> Hub:
    return Hub(load_config(seeded_repo))


def test_all_and_get(hub: Hub) -> None:
    assert len(hub.all()) == 11
    doc = hub.get("skill-async-python")
    assert doc.frontmatter.title == "Async Python"


def test_get_missing_raises_keyerror(hub: Hub) -> None:
    with pytest.raises(KeyError, match="nope"):
        hub.get("nope")


def test_filter_and_fulltext_delegate(hub: Hub) -> None:
    assert {d.id for d in hub.filter(type="skill")} == {
        "skill-async-python",
        "skill-sql",
        "skill-docker",
    }
    assert hub.fulltext("asyncio")[0].id == "skill-async-python"


def test_list_types_is_profile_vocabulary(hub: Hub) -> None:
    assert hub.list_types() == [
        "bio",
        "skill",
        "experience",
        "education",
        "project",
        "goal",
        "preference",
        "writing",
    ]


def test_list_tags_counts_most_used_first(hub: Hub) -> None:
    tags = hub.list_tags()
    assert tags["python"] == 4
    assert list(tags)[0] == "python"


def test_validate_reports_clean_store(hub: Hub) -> None:
    report = hub.validate()
    assert report.ok and report.checked == 11


def test_cache_serves_repeat_reads(hub: Hub) -> None:
    first = hub.all()
    second = hub.all()
    assert [d.id for d in first] == [d.id for d in second]


def test_cache_invalidates_on_hub_write(hub: Hub) -> None:
    hub.all()
    hub.add(type="skill", title="Terraform", body="IaC.")
    assert "skill-terraform" in {d.id for d in hub.all()}
    hub.update("skill-terraform", description="infrastructure as code")
    assert hub.get("skill-terraform").frontmatter.description == "infrastructure as code"
    hub.delete("skill-terraform")
    assert "skill-terraform" not in {d.id for d in hub.all()}


def test_cache_invalidates_on_external_file_change(hub: Hub, seeded_repo: Path) -> None:
    hub.all()  # populate the cache
    path = seeded_repo / "memory" / "skill" / "skill-sql.md"
    text = path.read_text(encoding="utf-8").replace("title: SQL", "title: SQL and more")
    path.write_text(text, encoding="utf-8")
    os.utime(path)  # ensure the mtime moves even on coarse filesystem clocks
    assert hub.get("skill-sql").frontmatter.title == "SQL and more"


def test_cache_invalidates_on_new_external_file(hub: Hub, seeded_repo: Path) -> None:
    hub.all()
    new = seeded_repo / "memory" / "bio" / "bio-short.md"
    new.write_text(memory_text(id="bio-short", type="bio"), encoding="utf-8")
    assert "bio-short" in {d.id for d in hub.all()}


def test_read_on_broken_store_raises_loaderror(hub: Hub, seeded_repo: Path) -> None:
    bad = seeded_repo / "memory" / "bio" / "bad.md"
    bad.write_text("---\nid: bad\n---\n", encoding="utf-8")
    with pytest.raises(LoadError):
        hub.all()
    bad.unlink()
    assert len(hub.all()) == 11  # recovers once the bad file is gone
