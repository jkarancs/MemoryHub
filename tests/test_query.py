"""Query tests: every filter dimension, AND-tags, fulltext ranking, regex."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from memoryhub import load_config, loader, query


@pytest.fixture
def docs(seeded_repo: Path):
    return loader.load_all(load_config(seeded_repo))


# --- filter -----------------------------------------------------------------------


def test_filter_by_type(docs) -> None:
    assert {d.id for d in query.filter(docs, type="skill")} == {
        "skill-async-python",
        "skill-sql",
        "skill-docker",
    }


def test_filter_by_status(docs) -> None:
    assert {d.id for d in query.filter(docs, status="draft")} == {"goal-ai-role"}
    assert {d.id for d in query.filter(docs, status="archived")} == {"writing-blog-post"}


def test_filter_by_visibility(docs) -> None:
    assert {d.id for d in query.filter(docs, visibility="public")} == {
        "bio-me",
        "project-site",
        "writing-blog-post",
    }


def test_filter_tags_are_and_matched(docs) -> None:
    assert {d.id for d in query.filter(docs, tags=["python"])} == {
        "skill-async-python",
        "project-hub",
        "experience-acme",
        "writing-blog-post",
    }
    assert {d.id for d in query.filter(docs, tags=["python", "async"])} == {"skill-async-python"}


def test_filter_proficiency_gte(docs) -> None:
    # beginner=1 intermediate=2 advanced=3 expert=4; docs without proficiency are excluded.
    assert {d.id for d in query.filter(docs, proficiency_gte=3)} == {"skill-async-python"}
    assert {d.id for d in query.filter(docs, proficiency_gte=2)} == {
        "skill-async-python",
        "skill-sql",
    }
    assert query.filter(docs, type="bio", proficiency_gte=1) == []


def test_filter_updated_after_is_strict(docs) -> None:
    after = {d.id for d in query.filter(docs, updated_after=date(2026, 1, 2))}
    assert after == {"skill-async-python", "project-hub"}
    # Strictly after: a doc updated exactly on the cutoff is excluded.
    assert query.filter(docs, updated_after=date(2026, 3, 1)) == []


def test_filter_dimensions_combine(docs) -> None:
    hits = query.filter(docs, type="project", visibility="public")
    assert [d.id for d in hits] == ["project-site"]


def test_filter_no_criteria_returns_all(docs) -> None:
    assert len(query.filter(docs)) == len(docs)


# --- fulltext ---------------------------------------------------------------------


def test_fulltext_ranks_by_match_count(docs) -> None:
    hits = query.fulltext(docs, "asyncio")
    # skill-async-python: 1 in description + 2 in body; project-hub: 1 in body.
    assert [d.id for d in hits] == ["skill-async-python", "project-hub"]


def test_fulltext_is_case_insensitive(docs) -> None:
    assert query.fulltext(docs, "ASYNCIO")
    assert query.fulltext(docs, "AsyncIO") == query.fulltext(docs, "asyncio")


def test_fulltext_searches_tags_by_default(docs) -> None:
    assert [d.id for d in query.fulltext(docs, "devops")] == ["skill-docker"]


def test_fulltext_field_restriction(docs) -> None:
    assert query.fulltext(docs, "asyncio", fields=("title",)) == []
    only_body = query.fulltext(docs, "asyncio", fields=("body",))
    assert {d.id for d in only_body} == {"skill-async-python", "project-hub"}


def test_fulltext_searches_extra_fields_when_named(docs) -> None:
    hits = query.fulltext(docs, "advanced", fields=("proficiency",))
    assert [d.id for d in hits] == ["skill-async-python"]


def test_fulltext_regex(docs) -> None:
    hits = query.fulltext(docs, r"async\w+", regex=True)
    assert hits and hits[0].id == "skill-async-python"


def test_fulltext_invalid_regex_raises(docs) -> None:
    with pytest.raises(re.error):
        query.fulltext(docs, "[unclosed", regex=True)


def test_fulltext_no_match_is_empty(docs) -> None:
    assert query.fulltext(docs, "zzz-not-there") == []


# --- rrf_fuse ---------------------------------------------------------------------


def by_id(docs, *ids: str) -> list:
    lookup = {d.id: d for d in docs}
    return [lookup[i] for i in ids]


def test_rrf_agreement_beats_single_list_wins(docs) -> None:
    # skill-sql is mid-ranked in both lists; the two exclusive top docs each score in one only.
    fused = query.rrf_fuse(
        [
            by_id(docs, "skill-async-python", "skill-sql"),
            by_id(docs, "project-hub", "skill-sql"),
        ]
    )
    assert [d.id for d in fused][0] == "skill-sql"


def test_rrf_handles_docs_absent_from_one_ranking(docs) -> None:
    fused = query.rrf_fuse([by_id(docs, "skill-sql"), []])
    assert [d.id for d in fused] == ["skill-sql"]


def test_rrf_ties_break_by_id(docs) -> None:
    fused = query.rrf_fuse([by_id(docs, "skill-sql", "bio-me"), by_id(docs, "bio-me", "skill-sql")])
    # Symmetric ranks → equal scores → alphabetical by id.
    assert [d.id for d in fused] == ["bio-me", "skill-sql"]


def test_rrf_empty_input(docs) -> None:
    assert query.rrf_fuse([]) == []
    assert query.rrf_fuse([[], []]) == []
