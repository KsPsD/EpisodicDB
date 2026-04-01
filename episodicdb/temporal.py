from __future__ import annotations

from datetime import datetime


class TemporalMixin:
    """Mixin providing temporal fact queries. Requires self._conn and self.agent_id."""

    def facts_as_of(
        self,
        as_of: datetime,
    ) -> list[dict]:
        """Return all facts that were valid at a specific point in time.

        This is the temporal point-in-time query that DuckDB has no native
        syntax for (no AS OF, no temporal tables). Every caller would need
        to manually write the valid_from/valid_until predicate.
        """
        rows = self._conn.execute(
            """
            SELECT key, value, valid_from, valid_until, episode_id::TEXT
            FROM facts
            WHERE agent_id = $1
              AND valid_from <= $2
              AND (valid_until IS NULL OR valid_until > $2)
            ORDER BY key
            """,
            [self.agent_id, as_of],
        ).fetchall()
        return [
            {
                "key": r[0],
                "value": r[1],
                "valid_from": r[2],
                "valid_until": r[3],
                "episode_id": r[4],
            }
            for r in rows
        ]

    def fact_history(
        self,
        key: str,
    ) -> list[dict]:
        """Return the full change history of a fact key, ordered chronologically."""
        rows = self._conn.execute(
            """
            SELECT value, valid_from, valid_until, episode_id::TEXT
            FROM facts
            WHERE agent_id = $1 AND key = $2
            ORDER BY valid_from ASC
            """,
            [self.agent_id, key],
        ).fetchall()
        return [
            {
                "value": r[0],
                "valid_from": r[1],
                "valid_until": r[2],
                "episode_id": r[3],
            }
            for r in rows
        ]
