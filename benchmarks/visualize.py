#!/usr/bin/env python3
"""Generate benchmark visualization charts for README and paper."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 11

RESULTS_DIR = Path(__file__).parent / "results"
ASSETS_DIR = Path(__file__).parent.parent / "assets"


def load_results() -> dict[int, dict]:
    results = {}
    for f in sorted(RESULTS_DIR.glob("*_episodes.json")):
        data = json.loads(f.read_text())
        results[data["scale"]] = data
    return results


def _category_stats(data: dict) -> dict[str, dict]:
    """Compute per-category avg p50/p95/p99."""
    cats: dict[str, list] = {}
    for q in data["queries"]:
        if q.get("error"):
            continue
        cats.setdefault(q["category"], []).append(q)

    stats = {}
    for cat, qs in cats.items():
        stats[cat] = {
            "p50": statistics.mean(q["p50_ms"] for q in qs),
            "p95": statistics.mean(q["p95_ms"] for q in qs),
            "p99": statistics.mean(q["p99_ms"] for q in qs),
        }
    return stats


def chart_latency_by_category(results: dict[int, dict]) -> None:
    """Bar chart: p50 latency per category across scales."""
    scales = sorted(results.keys())
    all_stats = {s: _category_stats(results[s]) for s in scales}

    categories = [
        "temporal", "comparison", "aggregation", "absence",
        "time_series", "causal_trace", "similarity",
    ]
    cat_labels = [
        "Temporal", "Comparison", "Aggregation", "Absence",
        "Time-series", "Causal Trace", "Similarity",
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(categories))
    width = 0.25
    colors = ["#4C78A8", "#F58518", "#E45756"]

    for i, scale in enumerate(scales):
        vals = [all_stats[scale].get(cat, {}).get("p50", 0) for cat in categories]
        offset = (i - 1) * width
        bars = ax.bar([xi + offset for xi in x], vals, width,
                      label=f"{scale//1000}K episodes", color=colors[i], alpha=0.9)
        for bar, val in zip(bars, vals):
            if val > 5:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Query Category")
    ax.set_ylabel("Latency (ms, p50)")
    ax.set_title("EpisodicDB Query Latency by Category (p50)")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    ASSETS_DIR.mkdir(exist_ok=True)
    fig.savefig(ASSETS_DIR / "bench_latency_by_category.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {ASSETS_DIR / 'bench_latency_by_category.png'}")


def chart_scaling_factor(results: dict[int, dict]) -> None:
    """Line chart: scaling factor (100K/10K ratio) per category."""
    if 10000 not in results or 100000 not in results:
        print("  Skipping scaling chart (need 10K and 100K)")
        return

    stats_10k = _category_stats(results[10000])
    stats_100k = _category_stats(results[100000])

    categories = [
        "temporal", "comparison", "aggregation", "absence",
        "time_series", "causal_trace", "similarity",
    ]
    cat_labels = [
        "Temporal", "Comparison", "Aggregation", "Absence",
        "Time-series", "Causal Trace", "Similarity",
    ]

    ratios_p50 = []
    ratios_p95 = []
    for cat in categories:
        p50_10k = stats_10k.get(cat, {}).get("p50", 0.001)
        p50_100k = stats_100k.get(cat, {}).get("p50", 0.001)
        p95_10k = stats_10k.get(cat, {}).get("p95", 0.001)
        p95_100k = stats_100k.get(cat, {}).get("p95", 0.001)
        ratios_p50.append(p50_100k / p50_10k if p50_10k > 0 else 0)
        ratios_p95.append(p95_100k / p95_10k if p95_10k > 0 else 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(categories))

    ax.bar([xi - 0.15 for xi in x], ratios_p50, 0.3,
           label="p50 ratio", color="#4C78A8", alpha=0.9)
    ax.bar([xi + 0.15 for xi in x], ratios_p95, 0.3,
           label="p95 ratio", color="#F58518", alpha=0.9)

    # Reference lines
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.5, label="Linear scaling (10x)")
    ax.axhline(y=1, color="green", linestyle="--", alpha=0.3)

    for xi, (r50, r95) in enumerate(zip(ratios_p50, ratios_p95)):
        ax.text(xi - 0.15, r50 + 0.15, f"{r50:.1f}x", ha="center", fontsize=8)
        ax.text(xi + 0.15, r95 + 0.15, f"{r95:.1f}x", ha="center", fontsize=8)

    ax.set_xlabel("Query Category")
    ax.set_ylabel("Scaling Factor (100K / 10K)")
    ax.set_title("Scaling Behavior: 10x Data → ?x Latency")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, rotation=15, ha="right")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "bench_scaling_factor.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {ASSETS_DIR / 'bench_scaling_factor.png'}")


def chart_write_throughput(results: dict[int, dict]) -> None:
    """Bar chart: write throughput at each scale."""
    scales = sorted(results.keys())

    eps_per_sec = [results[s]["data_stats"]["timing"]["episodes_per_sec"] for s in scales]
    labels = [f"{s//1000}K" for s in scales]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, eps_per_sec, color=["#4C78A8", "#F58518", "#E45756"], alpha=0.9)

    for bar, val in zip(bars, eps_per_sec):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Dataset Scale")
    ax.set_ylabel("Episodes / sec")
    ax.set_title("Write Throughput (episodes + tool_calls + decisions)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0, top=max(eps_per_sec) * 1.2)

    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "bench_write_throughput.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {ASSETS_DIR / 'bench_write_throughput.png'}")


def main():
    results = load_results()
    if not results:
        print("No results found in benchmarks/results/")
        return

    print(f"Loaded results for scales: {sorted(results.keys())}")

    chart_latency_by_category(results)
    chart_scaling_factor(results)
    chart_write_throughput(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
