"""Selectivity estimator for hybrid query planning.

Collects table statistics and estimates what fraction of rows
will pass a given set of predicates.
"""

from __future__ import annotations

import time


class SelectivityEstimator:
    """Estimates predicate selectivity using cached column statistics."""

    def __init__(self, conn, cache_ttl: float = 60.0):
        self._conn = conn
        self._cache_ttl = cache_ttl
        self._stats: dict[str, dict[str, int]] = {}  # {column: {value: count}}
        self._total_with_embedding: int = 0
        self._last_refresh: float = 0.0

    def estimate(self, agent_id: str, predicates: dict[str, str]) -> float:
        """Estimate combined selectivity for a set of equality predicates.

        Returns a float between 0.0 and 1.0 representing the fraction
        of embedding-bearing rows that pass all predicates.

        Assumes predicate independence (multiply individual selectivities).
        """
        if not predicates:
            return 1.0

        self._maybe_refresh(agent_id)

        if self._total_with_embedding == 0:
            return 0.0  # no embeddings → nothing can match

        combined = 1.0
        for col, val in predicates.items():
            col_stats = self._stats.get(col, {})
            count = col_stats.get(val, 0)
            sel = count / self._total_with_embedding
            combined *= sel

        return combined

    def get_filtered_count(self, agent_id: str, predicates: dict[str, str]) -> int:
        """Estimate how many rows will pass the predicates."""
        sel = self.estimate(agent_id, predicates)
        return max(1, int(sel * self._total_with_embedding))

    @property
    def total_with_embedding(self) -> int:
        return self._total_with_embedding

    def invalidate(self) -> None:
        """Force refresh on next estimate call."""
        self._last_refresh = 0.0

    def _maybe_refresh(self, agent_id: str) -> None:
        now = time.monotonic()
        if now - self._last_refresh < self._cache_ttl:
            return
        self._refresh(agent_id)
        self._last_refresh = now

    def _refresh(self, agent_id: str) -> None:
        """Collect column value distributions for episodes with embeddings."""
        # Total rows with embeddings
        row = self._conn.execute(
            "SELECT COUNT(*) FROM episodes "
            "WHERE context_embedding IS NOT NULL AND agent_id = $1",
            [agent_id],
        ).fetchone()
        self._total_with_embedding = row[0] if row else 0

        # Status distribution (most common filter)
        rows = self._conn.execute(
            "SELECT status, COUNT(*) FROM episodes "
            "WHERE context_embedding IS NOT NULL AND agent_id = $1 "
            "GROUP BY status",
            [agent_id],
        ).fetchall()
        self._stats["status"] = {r[0]: r[1] for r in rows}

        # Task type distribution
        rows = self._conn.execute(
            "SELECT task_type, COUNT(*) FROM episodes "
            "WHERE context_embedding IS NOT NULL AND agent_id = $1 "
            "GROUP BY task_type",
            [agent_id],
        ).fetchall()
        self._stats["task_type"] = {r[0]: r[1] for r in rows}
