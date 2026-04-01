from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from episodicdb.db import EpisodicDB

_db: EpisodicDB | None = None
_default_agent_id: str = ""


def _get_db() -> EpisodicDB:
    assert _db is not None, "Server not initialized"
    return _db


@contextmanager
def _agent_scope(agent_id: str | None):
    """Temporarily override the DB's agent_id, restoring it on exit."""
    db = _get_db()
    original = db.agent_id
    db.agent_id = agent_id if agent_id is not None else _default_agent_id
    try:
        yield db
    finally:
        db.agent_id = original


def _serialize_timestamps(rows: list[dict], keys: list[str]) -> list[dict]:
    """Convert datetime fields to ISO strings for JSON serialization."""
    for r in rows:
        for k in keys:
            if r.get(k):
                r[k] = r[k].isoformat()
    return rows


mcp_server = FastMCP("episodicdb")


# --- Writer tools ---


@mcp_server.tool()
def record_episode(
    status: str,
    task_type: str | None = None,
    context: dict | None = None,
    embedding: list[float] | None = None,
    tags: list[str] | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    agent_id: str | None = None,
) -> str:
    """Record an episode (task/session) for an agent."""
    with _agent_scope(agent_id) as db:
        return db.record_episode(
            status=status,
            task_type=task_type,
            context=context,
            embedding=embedding,
            tags=tags,
            started_at=datetime.fromisoformat(started_at) if started_at else None,
            ended_at=datetime.fromisoformat(ended_at) if ended_at else None,
        )


@mcp_server.tool()
def record_tool_call(
    episode_id: str,
    tool_name: str,
    outcome: str,
    parameters: dict | None = None,
    result: dict | None = None,
    duration_ms: int | None = None,
    error_message: str | None = None,
    called_at_override: str | None = None,
    agent_id: str | None = None,
) -> str:
    """Record a tool call within an episode."""
    with _agent_scope(agent_id) as db:
        return db.record_tool_call(
            episode_id=episode_id,
            tool_name=tool_name,
            outcome=outcome,
            parameters=parameters,
            result=result,
            duration_ms=duration_ms,
            error_message=error_message,
            called_at_override=datetime.fromisoformat(called_at_override) if called_at_override else None,
        )


@mcp_server.tool()
def record_decision(
    episode_id: str,
    rationale: str,
    decision_type: str | None = None,
    alternatives: list | None = None,
    outcome: str | None = None,
    agent_id: str | None = None,
) -> str:
    """Record a decision made during an episode."""
    with _agent_scope(agent_id) as db:
        return db.record_decision(
            episode_id=episode_id,
            rationale=rationale,
            decision_type=decision_type,
            alternatives=alternatives,
            outcome=outcome,
        )


@mcp_server.tool()
def record_fact(
    key: str,
    value: str,
    episode_id: str | None = None,
    valid_from: str | None = None,
    agent_id: str | None = None,
) -> str:
    """Record a fact with automatic supersession of previous values."""
    with _agent_scope(agent_id) as db:
        return db.record_fact(
            key=key,
            value=value,
            episode_id=episode_id,
            valid_from=datetime.fromisoformat(valid_from) if valid_from else None,
        )


# --- Analytics tools ---


@mcp_server.tool()
def top_failing_tools(
    days: int = 7,
    limit: int = 5,
    agent_id: str | None = None,
) -> str:
    """Get tools with the most failures, ranked by failure count."""
    with _agent_scope(agent_id) as db:
        return json.dumps(db.top_failing_tools(days=days, limit=limit))


@mcp_server.tool()
def never_succeeded_tools(
    agent_id: str | None = None,
) -> str:
    """List tools that have never had a successful outcome."""
    with _agent_scope(agent_id) as db:
        return json.dumps(db.never_succeeded_tools())


@mcp_server.tool()
def hourly_failure_rate(
    days: int = 7,
    agent_id: str | None = None,
) -> str:
    """Get failure counts grouped by hour of day."""
    with _agent_scope(agent_id) as db:
        return json.dumps(db.hourly_failure_rate(days=days))


@mcp_server.tool()
def compare_periods(
    metric: str,
    days: int = 7,
    agent_id: str | None = None,
) -> str:
    """Compare a metric between two consecutive time periods."""
    with _agent_scope(agent_id) as db:
        return json.dumps(db.compare_periods(metric=metric, days=days))


@mcp_server.tool()
def before_failure_sequence(
    tool_name: str,
    lookback: int = 3,
    agent_id: str | None = None,
) -> str:
    """Find which tools commonly precede failures of a given tool."""
    with _agent_scope(agent_id) as db:
        return json.dumps(db.before_failure_sequence(tool_name=tool_name, lookback=lookback))


@mcp_server.tool()
def similar_episodes(
    embedding: list[float],
    status: str | None = None,
    limit: int = 5,
    agent_id: str | None = None,
) -> str:
    """Find episodes most similar to a given embedding vector."""
    with _agent_scope(agent_id) as db:
        results = db.similar_episodes(embedding=embedding, status=status, limit=limit)
        return json.dumps(_serialize_timestamps(results, ["started_at", "ended_at"]))


# --- Temporal tools ---


@mcp_server.tool()
def facts_as_of(
    as_of: str,
    agent_id: str | None = None,
) -> str:
    """Return all facts that were valid at a specific point in time."""
    with _agent_scope(agent_id) as db:
        results = db.facts_as_of(as_of=datetime.fromisoformat(as_of))
        return json.dumps(_serialize_timestamps(results, ["valid_from", "valid_until"]))


@mcp_server.tool()
def fact_history(
    key: str,
    agent_id: str | None = None,
) -> str:
    """Return the full change history of a fact key."""
    with _agent_scope(agent_id) as db:
        results = db.fact_history(key=key)
        return json.dumps(_serialize_timestamps(results, ["valid_from", "valid_until"]))


def serve(agent_id: str, db_path: str | None = None) -> None:
    global _db, _default_agent_id
    _default_agent_id = agent_id
    _db = EpisodicDB(agent_id=agent_id, path=db_path)
    try:
        mcp_server.run(transport="stdio")
    finally:
        _db.close()
