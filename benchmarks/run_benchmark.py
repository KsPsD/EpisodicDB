#!/usr/bin/env python3
"""EpisodicDB Phase 1.5 Benchmark — CLI entry point.

Usage:
    python -m benchmarks.run_benchmark                    # default 10K
    python -m benchmarks.run_benchmark --scale 50000
    python -m benchmarks.run_benchmark --scale 10000 50000 100000  # multi-scale
    python -m benchmarks.run_benchmark --categories similarity temporal
    python -m benchmarks.run_benchmark --iterations 100

Results are saved to benchmarks/results/<scale>_episodes.json
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from episodicdb.db import EpisodicDB

from benchmarks.datagen import generate
from benchmarks.runner import BenchmarkResult, format_table, run_benchmark

RESULTS_DIR = Path(__file__).parent / "results"
TEMP_DB_DIR = Path(__file__).parent / ".tmp"


def run_single_scale(
    scale: int,
    iterations: int,
    categories: list[str] | None,
    seed: int,
) -> BenchmarkResult:
    """Generate data + run benchmark for one scale level."""
    db_path = TEMP_DB_DIR / f"bench_{scale}.db"
    TEMP_DB_DIR.mkdir(parents=True, exist_ok=True)

    # Clean previous run
    if db_path.exists():
        db_path.unlink()
    wal = db_path.with_suffix(".db.wal")
    if wal.exists():
        wal.unlink()

    print(f"\n{'#'*60}")
    print(f"# Scale: {scale:,} episodes")
    print(f"{'#'*60}")

    # Generate data
    print(f"\n[1/2] Generating {scale:,} synthetic episodes...")
    db = EpisodicDB(agent_id="benchmark", path=str(db_path))

    data_stats = generate(
        db,
        n_episodes=scale,
        seed=seed,
        embedding_ratio=0.3,
        n_clusters=20,
        fact_changes=max(200, scale // 50),
        progress_every=max(1000, scale // 10),
    )
    print(f"  Done: {data_stats['episodes']:,} episodes, "
          f"{data_stats['tool_calls']:,} tool_calls, "
          f"{data_stats['facts']:,} facts, "
          f"{data_stats['embeddings']:,} embeddings")
    print(f"  Write throughput: {data_stats['timing']['episodes_per_sec']:.0f} episodes/sec")

    # Run benchmark
    print(f"\n[2/2] Running queries ({iterations} iterations each)...")
    result = run_benchmark(
        db=db,
        data_stats=data_stats,
        scale=scale,
        iterations=iterations,
        categories=categories,
    )

    # Save
    out_path = RESULTS_DIR / f"{scale}_episodes.json"
    result.save(out_path)
    print(f"\n  Results saved: {out_path}")

    # Print table
    print(format_table(result))

    db.close()
    return result


def main():
    parser = argparse.ArgumentParser(description="EpisodicDB Phase 1.5 Benchmark")
    parser.add_argument(
        "--scale", nargs="+", type=int, default=[10_000],
        help="Episode counts to benchmark (default: 10000)",
    )
    parser.add_argument(
        "--iterations", type=int, default=50,
        help="Query iterations per benchmark (default: 50)",
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Only run specific categories (e.g. similarity temporal)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-db", action="store_true",
        help="Keep temporary DB files after benchmark",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total_start = time.monotonic()
    results = []

    for scale in args.scale:
        result = run_single_scale(
            scale=scale,
            iterations=args.iterations,
            categories=args.categories,
            seed=args.seed,
        )
        results.append(result)

    # Multi-scale comparison
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("MULTI-SCALE COMPARISON")
        print(f"{'='*70}")
        print(f"{'Scale':>10} {'Write eps/s':>12} {'Avg p50':>10} {'Avg p95':>10}")
        print("-" * 45)
        for r in results:
            all_p50 = [qr.p50 for qr in r.query_results if not qr.error]
            all_p95 = [qr.p95 for qr in r.query_results if not qr.error]
            import statistics
            print(
                f"{r.scale:>10,} "
                f"{r.data_stats.get('timing', {}).get('episodes_per_sec', 0):>11.0f} "
                f"{statistics.mean(all_p50):>9.2f} "
                f"{statistics.mean(all_p95):>9.2f}"
            )

    total = time.monotonic() - total_start
    print(f"\nTotal benchmark time: {total:.1f}s")

    # Cleanup
    if not args.keep_db and TEMP_DB_DIR.exists():
        shutil.rmtree(TEMP_DB_DIR)
        print("Temporary DB files cleaned up.")


if __name__ == "__main__":
    main()
