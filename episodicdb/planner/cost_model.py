"""Cost model for choosing between execution strategies.

Constants are calibrated from Phase 1.5 benchmark results on Apple M3 Pro.
"""

from __future__ import annotations

import math

# Calibrated from Phase 1.5 benchmarks (DuckDB 1.4.3, M3 Pro)
# Filter-First costs
SQL_SCAN_MS_PER_1K = 0.05     # columnar scan cost per 1K rows
NUMPY_COSINE_MS_PER_1K = 0.8  # NumPy vectorized cosine per 1K rows (1536-dim)
EMBEDDING_FETCH_MS_PER_1K = 0.3  # I/O: fetching embeddings from DuckDB to Python

# Vector-First costs
HNSW_BASE_MS = 2.0            # base HNSW search overhead
HNSW_LOG_FACTOR_MS = 3.0      # per log2(N) factor
POST_FILTER_MS_PER_1K = 0.01  # trivial dict comparison

# Strategy selection thresholds
FILTER_FIRST_THRESHOLD = 0.05   # use filter-first below this selectivity
VECTOR_FIRST_THRESHOLD = 0.50   # use vector-first above this selectivity


def filter_first_cost(n_total: int, selectivity: float) -> float:
    """Estimate cost in ms for Filter-First strategy."""
    n_filtered = max(1, int(n_total * selectivity))

    # Cost = SQL scan for filter + fetch embeddings + NumPy cosine
    scan_cost = n_total / 1000 * SQL_SCAN_MS_PER_1K
    fetch_cost = n_filtered / 1000 * EMBEDDING_FETCH_MS_PER_1K
    cosine_cost = n_filtered / 1000 * NUMPY_COSINE_MS_PER_1K

    return scan_cost + fetch_cost + cosine_cost


def vector_first_cost(n_total: int, limit: int, selectivity: float) -> float:
    """Estimate cost in ms for Vector-First strategy."""
    if selectivity <= 0:
        return float("inf")

    oversampling = min(limit / selectivity, n_total)

    # Cost = HNSW search + post-filter
    hnsw_cost = HNSW_BASE_MS + HNSW_LOG_FACTOR_MS * math.log2(max(n_total, 2))
    # HNSW cost scales with fetch limit somewhat
    hnsw_cost *= math.log2(max(oversampling, 2)) / math.log2(max(limit, 2))
    filter_cost = oversampling / 1000 * POST_FILTER_MS_PER_1K

    return hnsw_cost + filter_cost


def choose_strategy(
    n_total: int,
    selectivity: float,
    limit: int,
) -> str:
    """Choose the best execution strategy.

    Returns: "filter_first", "vector_first", or "no_filter"
    """
    if selectivity >= 1.0:
        return "no_filter"

    ff_cost = filter_first_cost(n_total, selectivity)
    vf_cost = vector_first_cost(n_total, limit, selectivity)

    # Hard thresholds as guardrails (override cost model in extreme cases)
    if selectivity < FILTER_FIRST_THRESHOLD:
        return "filter_first"
    if selectivity > VECTOR_FIRST_THRESHOLD:
        return "vector_first"

    # In the middle range, use cost model
    return "filter_first" if ff_cost < vf_cost else "vector_first"


def compute_oversampling(limit: int, selectivity: float) -> float:
    """Compute oversampling factor for Vector-First strategy.

    We need to fetch enough candidates so that after filtering,
    we still have at least `limit` results with high probability.
    """
    if selectivity <= 0:
        return 100.0

    # Fetch limit/selectivity * safety_margin candidates
    # Safety margin accounts for selectivity estimation error
    safety_margin = 2.0
    raw = limit / selectivity * safety_margin
    return min(max(raw, limit * 2), 100_000)
