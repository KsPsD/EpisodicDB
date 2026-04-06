"""Benchmark runner with statistical timing and reporting.

Runs each query multiple times, collects p50/p95/p99 latencies,
and outputs results as JSON + formatted table.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from episodicdb.db import EpisodicDB

from benchmarks.queries import CATEGORIES, QUERIES, BenchmarkQuery


@dataclass
class QueryResult:
    name: str
    category: str
    iterations: int
    latencies_ms: list[float] = field(default_factory=list)
    result_count: int | None = None
    error: str | None = None

    @property
    def p50(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        idx = int(len(sorted_l) * 0.5)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def p95(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def p99(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        idx = int(len(sorted_l) * 0.99)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def mean(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.latencies_ms) if len(self.latencies_ms) > 1 else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "iterations": self.iterations,
            "p50_ms": round(self.p50, 3),
            "p95_ms": round(self.p95, 3),
            "p99_ms": round(self.p99, 3),
            "mean_ms": round(self.mean, 3),
            "stdev_ms": round(self.stdev, 3),
            "result_count": self.result_count,
            "error": self.error,
        }


@dataclass
class BenchmarkResult:
    scale: int
    data_stats: dict
    query_results: list[QueryResult] = field(default_factory=list)
    system_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scale": self.scale,
            "data_stats": self.data_stats,
            "system_info": self.system_info,
            "queries": [qr.to_dict() for qr in self.query_results],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))


def _collect_system_info() -> dict:
    import platform
    import duckdb
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "duckdb": duckdb.__version__,
        "cpu": platform.processor(),
        "machine": platform.machine(),
    }


def run_query(
    db: EpisodicDB,
    query: BenchmarkQuery,
    iterations: int = 50,
    warmup: int = 3,
) -> QueryResult:
    """Run a single query multiple times and collect latency stats."""
    result = QueryResult(
        name=query.name,
        category=query.category,
        iterations=iterations,
    )

    # Warmup (discard results)
    for _ in range(warmup):
        try:
            query.fn(db)
        except Exception:
            pass

    # Timed runs
    for _ in range(iterations):
        try:
            t0 = time.perf_counter()
            res = query.fn(db)
            elapsed = (time.perf_counter() - t0) * 1000  # ms
            result.latencies_ms.append(elapsed)

            if result.result_count is None:
                if isinstance(res, list):
                    result.result_count = len(res)
                elif isinstance(res, dict):
                    result.result_count = 1
                else:
                    result.result_count = 0
        except Exception as e:
            result.error = str(e)
            break

    return result


def run_benchmark(
    db: EpisodicDB,
    data_stats: dict,
    scale: int,
    iterations: int = 50,
    warmup: int = 3,
    categories: list[str] | None = None,
) -> BenchmarkResult:
    """Run all benchmark queries and return collected results."""
    target_categories = set(categories) if categories else set(CATEGORIES)

    bench = BenchmarkResult(
        scale=scale,
        data_stats=data_stats,
        system_info=_collect_system_info(),
    )

    queries = [q for q in QUERIES if q.category in target_categories]
    total = len(queries)

    for i, query in enumerate(queries, 1):
        print(f"  [{i}/{total}] {query.category}/{query.name} ...", end=" ", flush=True)
        qr = run_query(db, query, iterations=iterations, warmup=warmup)
        bench.query_results.append(qr)

        if qr.error:
            print(f"ERROR: {qr.error}", flush=True)
        else:
            print(f"p50={qr.p50:.2f}ms  p95={qr.p95:.2f}ms  p99={qr.p99:.2f}ms", flush=True)

    return bench


def format_table(result: BenchmarkResult) -> str:
    """Format results as a readable ASCII table."""
    lines = []
    lines.append(f"\n{'='*90}")
    lines.append(f"EpisodicDB Benchmark — {result.scale:,} episodes")
    lines.append(f"{'='*90}")
    lines.append(f"System: {result.system_info.get('platform', 'unknown')}")
    lines.append(f"DuckDB: {result.system_info.get('duckdb', 'unknown')}")
    lines.append(f"Data: {result.data_stats.get('episodes', 0):,} episodes, "
                 f"{result.data_stats.get('tool_calls', 0):,} tool_calls, "
                 f"{result.data_stats.get('facts', 0):,} facts")
    lines.append(f"Write throughput: {result.data_stats.get('timing', {}).get('episodes_per_sec', 0):.0f} episodes/sec")
    lines.append(f"{'='*90}")

    header = f"{'Category':<14} {'Query':<35} {'p50':>8} {'p95':>8} {'p99':>8} {'mean':>8} {'rows':>6}"
    lines.append(header)
    lines.append("-" * 90)

    current_cat = ""
    for qr in sorted(result.query_results, key=lambda q: (q.category, q.name)):
        cat_display = qr.category if qr.category != current_cat else ""
        current_cat = qr.category

        d = qr.to_dict()
        rows = str(d["result_count"]) if d["result_count"] is not None else "-"
        error_marker = " *ERR*" if d["error"] else ""

        lines.append(
            f"{cat_display:<14} {qr.name:<35} "
            f"{d['p50_ms']:>7.2f} {d['p95_ms']:>7.2f} {d['p99_ms']:>7.2f} "
            f"{d['mean_ms']:>7.2f} {rows:>6}{error_marker}"
        )

    lines.append("=" * 90)

    # Category summary
    lines.append(f"\n{'Category Summary':}")
    lines.append(f"{'Category':<14} {'Avg p50':>10} {'Avg p95':>10} {'Avg p99':>10} {'Queries':>8}")
    lines.append("-" * 55)

    for cat in sorted(set(qr.category for qr in result.query_results)):
        cat_results = [qr for qr in result.query_results if qr.category == cat and not qr.error]
        if cat_results:
            avg_p50 = statistics.mean(qr.p50 for qr in cat_results)
            avg_p95 = statistics.mean(qr.p95 for qr in cat_results)
            avg_p99 = statistics.mean(qr.p99 for qr in cat_results)
            lines.append(
                f"{cat:<14} {avg_p50:>9.2f} {avg_p95:>9.2f} {avg_p99:>9.2f} {len(cat_results):>8}"
            )

    lines.append("")
    return "\n".join(lines)
