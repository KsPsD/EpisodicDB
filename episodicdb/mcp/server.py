from __future__ import annotations

import json
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from episodicdb.db import EpisodicDB

_db: EpisodicDB | None = None
_default_agent_id: str = ""


def _get_db() -> EpisodicDB:
    assert _db is not None, "Server not initialized"
    return _db


def _resolve_agent_id(agent_id: str | None) -> str:
    return agent_id if agent_id is not None else _default_agent_id


mcp_server = FastMCP("episodicdb")


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
    db = _get_db()
    original = db.agent_id
    db.agent_id = _resolve_agent_id(agent_id)
    try:
        sa = datetime.fromisoformat(started_at) if started_at else None
        ea = datetime.fromisoformat(ended_at) if ended_at else None
        return db.record_episode(
            status=status,
            task_type=task_type,
            context=context,
            embedding=embedding,
            tags=tags,
            started_at=sa,
            ended_at=ea,
        )
    finally:
        db.agent_id = original


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
    db = _get_db()
    original = db.agent_id
    db.agent_id = _resolve_agent_id(agent_id)
    try:
        cao = datetime.fromisoformat(called_at_override) if called_at_override else None
        return db.record_tool_call(
            episode_id=episode_id,
            tool_name=tool_name,
            outcome=outcome,
            parameters=parameters,
            result=result,
            duration_ms=duration_ms,
            error_message=error_message,
            called_at_override=cao,
        )
    finally:
        db.agent_id = original


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
    db = _get_db()
    original = db.agent_id
    db.agent_id = _resolve_agent_id(agent_id)
    try:
        return db.record_decision(
            episode_id=episode_id,
            rationale=rationale,
            decision_type=decision_type,
            alternatives=alternatives,
            outcome=outcome,
        )
    finally:
        db.agent_id = original


@mcp_server.tool()
def top_failing_tools(
    days: int = 7,
    limit: int = 5,
    agent_id: str | None = None,
) -> str:
    """Get tools with the most failures, ranked by failure count."""
    db = _get_db()
    original = db.agent_id
    db.agent_id = _resolve_agent_id(agent_id)
    try:
        return json.dumps(db.top_failing_tools(days=days, limit=limit))
    finally:
        db.agent_id = original


@mcp_server.tool()
def never_succeeded_tools(
    agent_id: str | None = None,
) -> str:
    """List tools that have never had a successful outcome."""
    db = _get_db()
    original = db.agent_id
    db.agent_id = _resolve_agent_id(agent_id)
    try:
        return json.dumps(db.never_succeeded_tools())
    finally:
        db.agent_id = original


@mcp_server.tool()
def hourly_failure_rate(
    days: int = 7,
    agent_id: str | None = None,
) -> str:
    """Get failure counts grouped by hour of day."""
    db = _get_db()
    original = db.agent_id
    db.agent_id = _resolve_agent_id(agent_id)
    try:
        return json.dumps(db.hourly_failure_rate(days=days))
    finally:
        db.agent_id = original


@mcp_server.tool()
def compare_periods(
    metric: str,
    days: int = 7,
    agent_id: str | None = None,
) -> str:
    """Compare a metric between two consecutive time periods."""
    db = _get_db()
    original = db.agent_id
    db.agent_id = _resolve_agent_id(agent_id)
    try:
        return json.dumps(db.compare_periods(metric=metric, days=days))
    finally:
        db.agent_id = original


@mcp_server.tool()
def before_failure_sequence(
    tool_name: str,
    lookback: int = 3,
    agent_id: str | None = None,
) -> str:
    """Find which tools commonly precede failures of a given tool."""
    db = _get_db()
    original = db.agent_id
    db.agent_id = _resolve_agent_id(agent_id)
    try:
        return json.dumps(db.before_failure_sequence(tool_name=tool_name, lookback=lookback))
    finally:
        db.agent_id = original


@mcp_server.tool()
def similar_episodes(
    embedding: list[float],
    status: str | None = None,
    limit: int = 5,
    agent_id: str | None = None,
) -> str:
    """Find episodes most similar to a given embedding vector."""
    db = _get_db()
    original = db.agent_id
    db.agent_id = _resolve_agent_id(agent_id)
    try:
        results = db.similar_episodes(embedding=embedding, status=status, limit=limit)
        for r in results:
            if r.get("started_at"):
                r["started_at"] = r["started_at"].isoformat()
            if r.get("ended_at"):
                r["ended_at"] = r["ended_at"].isoformat()
        return json.dumps(results)
    finally:
        db.agent_id = original


def serve(agent_id: str, db_path: str | None = None) -> None:
    global _db, _default_agent_id
    _default_agent_id = agent_id
    _db = EpisodicDB(agent_id=agent_id, path=db_path)
    try:
        mcp_server.run(transport="stdio")
    finally:
        _db.close()
