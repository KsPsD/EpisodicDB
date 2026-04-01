from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

_EMBEDDING_DIM = 1536


class WriterMixin:
    """Mixin providing write methods. Requires self._conn and self.agent_id."""

    def record_episode(
        self,
        status: Literal["success", "failure", "partial", "aborted"],
        task_type: str | None = None,
        context: dict | None = None,
        embedding: list[float] | None = None,
        tags: list[str] | None = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
    ) -> str:
        if embedding is not None and len(embedding) != _EMBEDDING_DIM:
            raise ValueError(
                f"Expected {_EMBEDDING_DIM} dimensions, got {len(embedding)}"
            )

        context_json = json.dumps(context) if context is not None else None

        row = self._conn.execute(
            """
            INSERT INTO episodes
                (agent_id, status, task_type, context, context_embedding, tags,
                 started_at, ended_at)
            VALUES ($1, $2, $3, $4, $5, $6,
                    COALESCE($7, NOW()), $8)
            RETURNING id::TEXT
            """,
            [
                self.agent_id,
                status,
                task_type,
                context_json,
                embedding,
                tags,
                started_at,
                ended_at,
            ],
        ).fetchone()
        return row[0]

    def record_tool_call(
        self,
        episode_id: str,
        tool_name: str,
        outcome: Literal["success", "failure", "timeout", "error"],
        parameters: dict | None = None,
        result: dict | None = None,
        duration_ms: int | None = None,
        error_message: str | None = None,
        called_at_override: datetime | None = None,
    ) -> str:
        params_json = json.dumps(parameters) if parameters is not None else None
        result_json = json.dumps(result) if result is not None else None

        row = self._conn.execute(
            """
            INSERT INTO tool_calls
                (episode_id, tool_name, outcome, parameters, result,
                 duration_ms, error_message, called_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7,
                    COALESCE($8, NOW()))
            RETURNING id::TEXT
            """,
            [
                episode_id,
                tool_name,
                outcome,
                params_json,
                result_json,
                duration_ms,
                error_message,
                called_at_override,
            ],
        ).fetchone()
        return row[0]

    def record_decision(
        self,
        episode_id: str,
        rationale: str,
        decision_type: str | None = None,
        alternatives: list | None = None,
        outcome: str | None = None,
    ) -> str:
        alts_json = json.dumps(alternatives) if alternatives is not None else None

        row = self._conn.execute(
            """
            INSERT INTO decisions
                (episode_id, rationale, decision_type, alternatives, outcome)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id::TEXT
            """,
            [episode_id, rationale, decision_type, alts_json, outcome],
        ).fetchone()
        return row[0]

    def record_fact(
        self,
        key: str,
        value: str,
        episode_id: str | None = None,
        valid_from: datetime | None = None,
    ) -> str:
        """Record a fact with automatic supersession.

        If a fact with the same key already exists (for this agent) and has no
        valid_until, its validity window is closed at the new fact's valid_from.
        This is the temporal supersession pattern that DuckDB cannot enforce
        natively (no triggers, no temporal constraints).
        """
        now = valid_from if valid_from is not None else None

        # Close the currently-active fact for this key (auto-supersession)
        self._conn.execute(
            """
            UPDATE facts
            SET valid_until = COALESCE($3, NOW())
            WHERE agent_id = $1
              AND key = $2
              AND valid_until IS NULL
            """,
            [self.agent_id, key, now],
        )

        row = self._conn.execute(
            """
            INSERT INTO facts
                (agent_id, key, value, valid_from, episode_id)
            VALUES ($1, $2, $3, COALESCE($4, NOW()), $5)
            RETURNING id::TEXT
            """,
            [self.agent_id, key, value, now, episode_id],
        ).fetchone()
        return row[0]
