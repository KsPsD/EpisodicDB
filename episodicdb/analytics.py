from __future__ import annotations

from typing import Literal

from episodicdb.schema import EMBEDDING_DIM


class AnalyticsMixin:
    """Mixin providing analytics methods. Requires self._conn and self.agent_id."""

    def top_failing_tools(
        self,
        days: int = 7,
        limit: int = 5,
    ) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT tc.tool_name, COUNT(*) AS failures
            FROM tool_calls tc
            JOIN episodes e ON e.id = tc.episode_id
            WHERE tc.outcome = 'failure'
              AND e.agent_id = $1
              AND tc.called_at >= NOW() - INTERVAL (CAST($2 AS VARCHAR) || ' days')
            GROUP BY tc.tool_name
            ORDER BY failures DESC
            LIMIT $3
            """,
            [self.agent_id, days, limit],
        ).fetchall()
        return [{"tool_name": r[0], "failures": r[1]} for r in rows]

    def never_succeeded_tools(self) -> list[str]:
        rows = self._conn.execute(
            """
            SELECT DISTINCT tc.tool_name
            FROM tool_calls tc
            JOIN episodes e ON e.id = tc.episode_id
            WHERE e.agent_id = $1
              AND tc.tool_name NOT IN (
                SELECT DISTINCT tc2.tool_name
                FROM tool_calls tc2
                JOIN episodes e2 ON e2.id = tc2.episode_id
                WHERE e2.agent_id = $1
                  AND tc2.outcome = 'success'
            )
            ORDER BY tc.tool_name
            """,
            [self.agent_id],
        ).fetchall()
        return [r[0] for r in rows]

    def hourly_failure_rate(
        self,
        days: int = 7,
    ) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT
                hour(tc.called_at) AS hour,
                COUNT(*) AS total,
                SUM(CASE WHEN tc.outcome = 'failure' THEN 1 ELSE 0 END) AS failures
            FROM tool_calls tc
            JOIN episodes e ON e.id = tc.episode_id
            WHERE e.agent_id = $1
              AND tc.called_at >= NOW() - INTERVAL (CAST($2 AS VARCHAR) || ' days')
            GROUP BY hour(tc.called_at)
            ORDER BY hour(tc.called_at)
            """,
            [self.agent_id, days],
        ).fetchall()
        return [{"hour": r[0], "total": r[1], "failures": r[2]} for r in rows]

    def compare_periods(
        self,
        metric: Literal["failure_rate", "episode_count", "tool_calls"],
        days: int = 7,
    ) -> dict:
        if metric == "failure_rate":
            row = self._conn.execute(
                """
                SELECT
                    AVG(CASE WHEN tc.called_at >= NOW() - INTERVAL (CAST($2 AS VARCHAR) || ' days')
                             THEN CASE WHEN tc.outcome = 'failure' THEN 1.0 ELSE 0.0 END
                             ELSE NULL END) AS period_a,
                    AVG(CASE WHEN tc.called_at >= NOW() - INTERVAL (CAST($3 AS VARCHAR) || ' days')
                              AND tc.called_at <  NOW() - INTERVAL (CAST($2 AS VARCHAR) || ' days')
                             THEN CASE WHEN tc.outcome = 'failure' THEN 1.0 ELSE 0.0 END
                             ELSE NULL END) AS period_b
                FROM tool_calls tc
                JOIN episodes e ON e.id = tc.episode_id
                WHERE e.agent_id = $1
                  AND tc.called_at >= NOW() - INTERVAL (CAST($3 AS VARCHAR) || ' days')
                """,
                [self.agent_id, days, days * 2],
            ).fetchone()
            a = row[0] or 0.0
            b = row[1] or 0.0
        elif metric == "episode_count":
            row = self._conn.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE started_at >= NOW() - INTERVAL (CAST($2 AS VARCHAR) || ' days')) AS period_a,
                    COUNT(*) FILTER (WHERE started_at >= NOW() - INTERVAL (CAST($3 AS VARCHAR) || ' days')
                                       AND started_at <  NOW() - INTERVAL (CAST($2 AS VARCHAR) || ' days')) AS period_b
                FROM episodes
                WHERE agent_id = $1
                  AND started_at >= NOW() - INTERVAL (CAST($3 AS VARCHAR) || ' days')
                """,
                [self.agent_id, days, days * 2],
            ).fetchone()
            a = float(row[0] or 0)
            b = float(row[1] or 0)
        else:  # tool_calls
            row = self._conn.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE tc.called_at >= NOW() - INTERVAL (CAST($2 AS VARCHAR) || ' days')) AS period_a,
                    COUNT(*) FILTER (WHERE tc.called_at >= NOW() - INTERVAL (CAST($3 AS VARCHAR) || ' days')
                                       AND tc.called_at <  NOW() - INTERVAL (CAST($2 AS VARCHAR) || ' days')) AS period_b
                FROM tool_calls tc
                JOIN episodes e ON e.id = tc.episode_id
                WHERE e.agent_id = $1
                  AND tc.called_at >= NOW() - INTERVAL (CAST($3 AS VARCHAR) || ' days')
                """,
                [self.agent_id, days, days * 2],
            ).fetchone()
            a = float(row[0] or 0)
            b = float(row[1] or 0)

        return {"period_a": round(a, 4), "period_b": round(b, 4), "delta": round(a - b, 4)}

    def before_failure_sequence(
        self,
        tool_name: str,
        lookback: int = 3,
    ) -> list[dict]:
        """Aggregate tool calls immediately before tool_name failures."""
        lag_cols = ", ".join(
            f"LAG(tc.tool_name, {i}) OVER (ORDER BY tc.called_at) AS prev_{i}"
            for i in range(1, lookback + 1)
        )
        prev_selects = " UNION ALL ".join(
            f"SELECT prev_{i} AS prev_tool FROM failures WHERE prev_{i} IS NOT NULL"
            for i in range(1, lookback + 1)
        )
        rows = self._conn.execute(
            f"""
            WITH sequenced AS (
                SELECT
                    tc.tool_name,
                    tc.outcome,
                    {lag_cols}
                FROM tool_calls tc
                JOIN episodes e ON e.id = tc.episode_id
                WHERE e.agent_id = $1
            ),
            failures AS (
                SELECT * FROM sequenced
                WHERE tool_name = $2 AND outcome = 'failure'
            )
            SELECT prev_tool, COUNT(*) AS cnt FROM (
                {prev_selects}
            )
            GROUP BY prev_tool
            ORDER BY cnt DESC
            """,
            [self.agent_id, tool_name],
        ).fetchall()
        return [{"prev_tool": r[0], "count": r[1]} for r in rows]

    def similar_episodes(
        self,
        embedding: list[float],
        status: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """SQL predicate + vector similarity in a single execution plan."""
        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(f"Expected {EMBEDDING_DIM} dimensions, got {len(embedding)}")

        params: list = [embedding, self.agent_id]
        status_clause = ""
        if status is not None:
            status_clause = "AND status = $3"
            params.append(status)
        params.append(limit)
        limit_param = f"${len(params)}"

        rows = self._conn.execute(
            f"""
            SELECT
                id::TEXT,
                agent_id,
                status,
                task_type,
                started_at,
                ended_at,
                array_cosine_distance(context_embedding, $1::FLOAT[{EMBEDDING_DIM}]) AS distance
            FROM episodes
            WHERE context_embedding IS NOT NULL
              AND agent_id = $2
              {status_clause}
            ORDER BY distance ASC
            LIMIT {limit_param}
            """,
            params,
        ).fetchall()

        return [
            {
                "id": r[0],
                "agent_id": r[1],
                "status": r[2],
                "task_type": r[3],
                "started_at": r[4],
                "ended_at": r[5],
                "distance": r[6],
            }
            for r in rows
        ]
