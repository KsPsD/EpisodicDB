import random
import datetime

import pytest


def _make_embedding(seed: int = 0) -> list[float]:
    random.seed(seed)
    return [random.uniform(-1, 1) for _ in range(1536)]


# --- top_failing_tools ---

def test_top_failing_tools_order(seeded_db):
    results = seeded_db.top_failing_tools(days=30, limit=5)
    assert len(results) > 0
    failures = [r["failures"] for r in results]
    assert failures == sorted(failures, reverse=True)
    assert "tool_name" in results[0]
    assert "failures" in results[0]


def test_top_failing_tools_days_filter(seeded_db):
    results = seeded_db.top_failing_tools(days=0)
    assert results == []


# --- never_succeeded_tools ---

def test_never_succeeded_tools(seeded_db):
    results = seeded_db.never_succeeded_tools()
    assert isinstance(results, list)
    for tool in results:
        assert isinstance(tool, str)


# --- hourly_failure_rate ---

def test_hourly_failure_rate_structure(seeded_db):
    results = seeded_db.hourly_failure_rate(days=30)
    assert isinstance(results, list)
    for r in results:
        assert "hour" in r
        assert "total" in r
        assert "failures" in r
        assert 0 <= r["hour"] <= 23
        assert r["failures"] <= r["total"]


# --- compare_periods ---

def test_compare_periods_returns_dict(seeded_db):
    result = seeded_db.compare_periods(metric="failure_rate", days=7)
    assert "period_a" in result
    assert "period_b" in result
    assert "delta" in result
    assert abs(result["delta"] - (result["period_a"] - result["period_b"])) < 1e-6


def test_compare_periods_episode_count(seeded_db):
    result = seeded_db.compare_periods(metric="episode_count", days=7)
    assert result["period_a"] >= 0
    assert result["period_b"] >= 0


# --- before_failure_sequence ---

def test_before_failure_sequence(db):
    """Verify Bash appears before Edit failures."""
    ep_id = db.record_episode(status="failure")
    now = datetime.datetime.now(datetime.timezone.utc)
    delta = datetime.timedelta

    db.record_tool_call(ep_id, "Bash", "success",
                        called_at_override=now - delta(seconds=30))
    db.record_tool_call(ep_id, "Edit", "failure",
                        called_at_override=now - delta(seconds=20))

    ep2 = db.record_episode(status="failure")
    db.record_tool_call(ep2, "Read", "success",
                        called_at_override=now - delta(seconds=10))
    db.record_tool_call(ep2, "Edit", "failure",
                        called_at_override=now - delta(seconds=5))

    results = db.before_failure_sequence("Edit", lookback=1)
    assert len(results) > 0
    tool_names = [r["prev_tool"] for r in results]
    assert "Bash" in tool_names or "Read" in tool_names
    for r in results:
        assert r["count"] > 0
        assert "prev_tool" in r


# --- similar_episodes ---

def test_similar_episodes_returns_sorted(db):
    q = _make_embedding(0)
    for i in range(5):
        db.record_episode(
            status="success" if i % 2 == 0 else "failure",
            embedding=_make_embedding(i),
        )

    results = db.similar_episodes(embedding=q, limit=3)
    assert len(results) <= 3
    dists = [r["distance"] for r in results]
    assert dists == sorted(dists)


def test_similar_episodes_status_filter(db):
    q = _make_embedding(0)
    for i in range(6):
        db.record_episode(
            status="success" if i < 3 else "failure",
            embedding=_make_embedding(i),
        )

    results = db.similar_episodes(embedding=q, status="success", limit=10)
    for r in results:
        assert r["status"] == "success"


def test_similar_episodes_excludes_null_embedding(db):
    q = _make_embedding(0)
    db.record_episode(status="success")  # no embedding
    db.record_episode(status="success", embedding=_make_embedding(1))

    results = db.similar_episodes(embedding=q, limit=5)
    assert len(results) <= 1
