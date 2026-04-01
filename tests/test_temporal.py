"""Tests for temporal fact validity — the DuckDB-impossible pattern.

DuckDB has no triggers, no temporal constraints, no AS OF queries.
EpisodicDB provides auto-supersession and point-in-time queries
at the application layer.
"""
from datetime import datetime, timedelta, timezone

import pytest

from episodicdb.db import EpisodicDB


@pytest.fixture
def db():
    with EpisodicDB(agent_id="test", path=":memory:") as db:
        yield db


# --- record_fact + auto-supersession ---

def test_record_fact_returns_uuid(db):
    fact_id = db.record_fact(key="theme", value="dark")
    assert isinstance(fact_id, str)
    assert len(fact_id) == 36


def test_record_fact_auto_supersession(db):
    """When a new fact with the same key is recorded, the old one is closed."""
    t1 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc)

    db.record_fact(key="theme", value="dark", valid_from=t1)
    db.record_fact(key="theme", value="light", valid_from=t2)

    history = db.fact_history("theme")
    assert len(history) == 2

    # First fact should have valid_until == t2
    assert history[0]["value"] == "dark"
    assert history[0]["valid_until"].astimezone(timezone.utc) == t2

    # Second fact should still be open (valid_until is None)
    assert history[1]["value"] == "light"
    assert history[1]["valid_until"] is None


def test_different_keys_independent(db):
    """Facts with different keys don't affect each other."""
    t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)

    db.record_fact(key="theme", value="dark", valid_from=t1)
    db.record_fact(key="language", value="en", valid_from=t1)

    theme_history = db.fact_history("theme")
    lang_history = db.fact_history("language")

    assert len(theme_history) == 1
    assert theme_history[0]["valid_until"] is None

    assert len(lang_history) == 1
    assert lang_history[0]["valid_until"] is None


def test_triple_supersession(db):
    """Three consecutive values for the same key."""
    t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2025, 2, 1, tzinfo=timezone.utc)
    t3 = datetime(2025, 3, 1, tzinfo=timezone.utc)

    db.record_fact(key="model", value="gpt-4", valid_from=t1)
    db.record_fact(key="model", value="claude-3", valid_from=t2)
    db.record_fact(key="model", value="claude-4", valid_from=t3)

    history = db.fact_history("model")
    assert len(history) == 3
    assert history[0]["valid_until"].astimezone(timezone.utc) == t2
    assert history[1]["valid_until"].astimezone(timezone.utc) == t3
    assert history[2]["valid_until"] is None


# --- facts_as_of ---

def test_facts_as_of_returns_valid_facts(db):
    t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)

    db.record_fact(key="theme", value="dark", valid_from=t1)
    db.record_fact(key="theme", value="light", valid_from=t2)
    db.record_fact(key="language", value="ko", valid_from=t1)

    # Query at t1 + 1 day: theme=dark, language=ko
    snapshot = db.facts_as_of(datetime(2025, 1, 2, tzinfo=timezone.utc))
    facts = {f["key"]: f["value"] for f in snapshot}
    assert facts["theme"] == "dark"
    assert facts["language"] == "ko"

    # Query at t2 + 1 day: theme=light, language=ko
    snapshot = db.facts_as_of(datetime(2025, 6, 2, tzinfo=timezone.utc))
    facts = {f["key"]: f["value"] for f in snapshot}
    assert facts["theme"] == "light"
    assert facts["language"] == "ko"


def test_facts_as_of_before_any_fact(db):
    t1 = datetime(2025, 6, 1, tzinfo=timezone.utc)
    db.record_fact(key="theme", value="dark", valid_from=t1)

    # Query before any fact exists
    snapshot = db.facts_as_of(datetime(2025, 1, 1, tzinfo=timezone.utc))
    assert snapshot == []


def test_facts_as_of_agent_isolation(db):
    """Facts from other agents are not visible."""
    t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    db.record_fact(key="theme", value="dark", valid_from=t1)

    # Switch agent_id
    db.agent_id = "other-agent"
    snapshot = db.facts_as_of(datetime(2025, 6, 1, tzinfo=timezone.utc))
    assert snapshot == []


# --- fact_history ---

def test_fact_history_empty(db):
    history = db.fact_history("nonexistent")
    assert history == []


def test_fact_history_with_episode_link(db):
    ep_id = db.record_episode(status="success")
    t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    db.record_fact(key="preference", value="verbose", valid_from=t1, episode_id=ep_id)

    history = db.fact_history("preference")
    assert len(history) == 1
    assert history[0]["episode_id"] == ep_id
