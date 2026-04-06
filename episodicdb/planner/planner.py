"""HybridPlanner: the main entry point for optimized vector+SQL queries.

Sits between similar_episodes() and DuckDB execution.
Estimates selectivity, chooses strategy, executes.
"""

from __future__ import annotations

import logging

from episodicdb.planner.cost_model import choose_strategy, compute_oversampling
from episodicdb.planner.selectivity import SelectivityEstimator
from episodicdb.planner import strategies

logger = logging.getLogger(__name__)


class HybridPlanner:
    """Hybrid query planner for filtered vector search.

    Automatically chooses the fastest execution strategy based on
    predicate selectivity estimation.
    """

    def __init__(self, conn, cache_ttl: float = 60.0):
        self._conn = conn
        self._estimator = SelectivityEstimator(conn, cache_ttl=cache_ttl)

    @property
    def estimator(self) -> SelectivityEstimator:
        return self._estimator

    def execute(
        self,
        agent_id: str,
        embedding: list[float],
        predicates: dict[str, str] | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """Execute a filtered vector similarity query with automatic strategy selection."""
        if not predicates:
            return strategies.no_filter(self._conn, agent_id, embedding, limit)

        # Estimate selectivity
        selectivity = self._estimator.estimate(agent_id, predicates)
        n_total = self._estimator.total_with_embedding

        # Choose strategy
        strategy = choose_strategy(n_total, selectivity, limit)

        logger.debug(
            "planner: selectivity=%.4f n_total=%d strategy=%s",
            selectivity, n_total, strategy,
        )

        # Execute
        if strategy == "no_filter":
            return strategies.no_filter(self._conn, agent_id, embedding, limit)
        elif strategy == "filter_first":
            return strategies.filter_first(
                self._conn, agent_id, embedding, predicates, limit,
            )
        else:  # vector_first
            fetch_limit = int(compute_oversampling(limit, selectivity))
            return strategies.vector_first(
                self._conn, agent_id, embedding, predicates, limit,
                fetch_limit=fetch_limit,
            )

    def invalidate_stats(self) -> None:
        """Force statistics refresh on next query."""
        self._estimator.invalidate()
