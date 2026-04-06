"""Cost model for choosing between execution strategies.

Constants calibrated from Phase 2a benchmark (DuckDB 1.4.3, M3 Pro).
The critical insight: fetching 1536-dim embeddings from DuckDB to Python
is ~100x more expensive than originally estimated. Filter-First is only
viable when very few rows survive the filter (< 1-2% of embeddings).
"""

from __future__ import annotations

import math

# Calibrated from Phase 2a benchmarks
# Filter-First costs (dominated by embedding I/O)
SQL_SCAN_MS_PER_1K = 0.05       # columnar scan cost per 1K rows
NUMPY_COSINE_MS_PER_1K = 0.8    # NumPy vectorized cosine per 1K rows (1536-dim)
EMBEDDING_FETCH_MS_PER_1K = 30.0  # I/O: fetching 1536-dim float[] from DuckDB to Python
                                   # This dominates filter-first cost.
                                   # Measured: 10K filtered rows ≈ 300ms fetch

# Vector-First costs (DuckDB-internal, much cheaper)
HNSW_BASE_MS = 5.0              # base HNSW search overhead
HNSW_LOG_FACTOR_MS = 8.0        # per log2(N) factor (calibrated: ~20ms at 10K, ~150ms at 100K)
POST_FILTER_MS_PER_1K = 0.01    # trivial dict comparison

# Strategy selection thresholds
# Filter-First is only worth it when embedding fetch is tiny
FILTER_FIRST_THRESHOLD = 0.02   # use filter-first below 2% selectivity


def filter_first_cost(n_total: int, selectivity: float) -> float:
    """Estimate cost in ms for Filter-First strategy."""
    n_filtered = max(1, int(n_total * selectivity))

    scan_cost = n_total / 1000 * SQL_SCAN_MS_PER_1K
    fetch_cost = n_filtered / 1000 * EMBEDDING_FETCH_MS_PER_1K
    cosine_cost = n_filtered / 1000 * NUMPY_COSINE_MS_PER_1K

    return scan_cost + fetch_cost + cosine_cost


def vector_first_cost(n_total: int, limit: int, selectivity: float) -> float:
    """Estimate cost in ms for Vector-First strategy."""
    if selectivity <= 0:
        return float("inf")

    oversampling = min(limit / selectivity, n_total)

    # HNSW cost: base + log scaling with dataset size
    hnsw_cost = HNSW_BASE_MS + HNSW_LOG_FACTOR_MS * math.log2(max(n_total, 2))
    # Additional cost for fetching more candidates
    if oversampling > limit:
        hnsw_cost *= 1.0 + 0.3 * math.log2(oversampling / max(limit, 1))
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

    # Very low selectivity: filter-first is clearly better
    # (fetch only a handful of embeddings)
    if selectivity < FILTER_FIRST_THRESHOLD:
        return "filter_first"

    # Everything else: vector-first with oversampling
    # The embedding fetch cost makes filter-first uncompetitive
    # for selectivity > 2%
    return "vector_first"


def compute_oversampling(limit: int, selectivity: float) -> float:
    """Compute how many candidates to fetch for Vector-First strategy.

    Returns the total number of candidates (not a ratio).
    """
    if selectivity <= 0:
        return 100.0

    # Fetch limit/selectivity * safety_margin candidates
    safety_margin = 2.0
    raw = limit / selectivity * safety_margin
    return min(max(raw, limit * 2), 100_000)
