"""Benchmark query definitions — 7 categories matching the EpisodicDB thesis.

Each query function takes an EpisodicDB instance and returns its result.
The runner wraps each in timing + statistics collection.

Categories:
1. similarity    — vector similarity search (similar_episodes)
2. aggregation   — GROUP BY analytics (top_failing_tools)
3. time_series   — hourly bucketed analysis (hourly_failure_rate)
4. causal_trace  — sequential pattern mining (before_failure_sequence)
5. comparison    — period-over-period delta (compare_periods)
6. absence       — anti-join / NOT IN (never_succeeded_tools)
7. temporal      — point-in-time + history (facts_as_of, fact_history)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

from episodicdb.db import EpisodicDB
from episodicdb.schema import EMBEDDING_DIM

from benchmarks.datagen import _random_embedding, TOOL_NAMES, FACT_KEYS


@dataclass
class BenchmarkQuery:
    name: str
    category: str
    description: str
    fn: Callable[[EpisodicDB], object]
    variants: list[Callable[[EpisodicDB], object]] = field(default_factory=list)


def _make_queries(seed: int = 42) -> list[BenchmarkQuery]:
    """Build the full set of benchmark queries."""
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)

    queries: list[BenchmarkQuery] = []

    # ── 1. Similarity Search ──
    for k in [5, 10, 20]:
        emb = _random_embedding(cluster_id=rng.randint(0, 19))
        queries.append(BenchmarkQuery(
            name=f"similar_episodes_top{k}",
            category="similarity",
            description=f"Find {k} most similar episodes by embedding",
            fn=lambda db, e=emb, lim=k: db.similar_episodes(embedding=e, limit=lim),
        ))

    # Similarity with status filter
    emb = _random_embedding(cluster_id=rng.randint(0, 19))
    queries.append(BenchmarkQuery(
        name="similar_episodes_filtered",
        category="similarity",
        description="Similar episodes filtered by status='failure'",
        fn=lambda db, e=emb: db.similar_episodes(embedding=e, status="failure", limit=10),
    ))

    # ── 2. Aggregation ──
    for days in [7, 30, 90]:
        queries.append(BenchmarkQuery(
            name=f"top_failing_tools_{days}d",
            category="aggregation",
            description=f"Top 10 failing tools in last {days} days",
            fn=lambda db, d=days: db.top_failing_tools(days=d, limit=10),
        ))

    # ── 3. Time Series ──
    for days in [7, 30, 90]:
        queries.append(BenchmarkQuery(
            name=f"hourly_failure_rate_{days}d",
            category="time_series",
            description=f"Hourly failure rate over {days} days",
            fn=lambda db, d=days: db.hourly_failure_rate(days=d),
        ))

    # ── 4. Causal Trace ──
    # Pick tools known to fail often
    for tool in ["Deploy", "RunTests", "APICall"]:
        for lookback in [3, 5]:
            queries.append(BenchmarkQuery(
                name=f"before_failure_{tool}_lb{lookback}",
                category="causal_trace",
                description=f"Tools preceding {tool} failures (lookback={lookback})",
                fn=lambda db, t=tool, lb=lookback: db.before_failure_sequence(
                    tool_name=t, lookback=lb
                ),
            ))

    # ── 5. Comparison (period-over-period) ──
    for metric in ["failure_rate", "episode_count", "tool_calls"]:
        for days in [7, 30]:
            queries.append(BenchmarkQuery(
                name=f"compare_{metric}_{days}d",
                category="comparison",
                description=f"Compare {metric} between current and previous {days}-day period",
                fn=lambda db, m=metric, d=days: db.compare_periods(metric=m, days=d),
            ))

    # ── 6. Absence ──
    queries.append(BenchmarkQuery(
        name="never_succeeded_tools",
        category="absence",
        description="Tools that have never succeeded",
        fn=lambda db: db.never_succeeded_tools(),
    ))

    # ── 7. Temporal ──
    # Point-in-time at various offsets
    for days_ago in [1, 7, 30, 60]:
        as_of = now - timedelta(days=days_ago)
        queries.append(BenchmarkQuery(
            name=f"facts_as_of_{days_ago}d_ago",
            category="temporal",
            description=f"All facts valid {days_ago} days ago",
            fn=lambda db, t=as_of: db.facts_as_of(as_of=t),
        ))

    # Fact history for specific keys
    for key in ["preferred_language", "current_project", "deploy_target"]:
        queries.append(BenchmarkQuery(
            name=f"fact_history_{key}",
            category="temporal",
            description=f"Full change history of '{key}'",
            fn=lambda db, k=key: db.fact_history(key=k),
        ))

    return queries


# Module-level singleton
QUERIES = _make_queries()
CATEGORIES = sorted(set(q.category for q in QUERIES))
