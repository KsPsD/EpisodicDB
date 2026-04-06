#!/usr/bin/env python3
"""Generate Phase 1.5 vs Phase 2a comparison charts."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 11

RESULTS_DIR = Path(__file__).parent / "results"
ASSETS_DIR = Path(__file__).parent.parent / "assets"


def load_phase(tag: str) -> dict[int, dict]:
    phase_dir = RESULTS_DIR / tag
    results = {}
    for f in sorted(phase_dir.glob("*_episodes.json")):
        data = json.loads(f.read_text())
        results[data["scale"]] = data
    return results


def _get_query(data: dict, name: str) -> dict | None:
    for q in data["queries"]:
        if q["name"] == name:
            return q
    return None


def _category_avg(data: dict, category: str, metric: str = "p50_ms") -> float:
    qs = [q for q in data["queries"] if q["category"] == category and not q.get("error")]
    return statistics.mean(q[metric] for q in qs) if qs else 0


def chart_filtered_similarity_comparison(p15: dict, p2a: dict) -> None:
    """Bar chart: filtered similarity p50 across scales, baseline vs planner."""
    scales = sorted(set(p15.keys()) & set(p2a.keys()))
    labels = [f"{s//1000}K" for s in scales]

    p15_vals = []
    p2a_vals = []
    for s in scales:
        q15 = _get_query(p15[s], "similar_episodes_filtered")
        q2a = _get_query(p2a[s], "similar_episodes_filtered")
        p15_vals.append(q15["p50_ms"] if q15 else 0)
        p2a_vals.append(q2a["p50_ms"] if q2a else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(scales))
    width = 0.35

    bars1 = ax.bar([xi - width/2 for xi in x], p15_vals, width,
                   label="Phase 1.5 (baseline)", color="#4C78A8", alpha=0.9)
    bars2 = ax.bar([xi + width/2 for xi in x], p2a_vals, width,
                   label="Phase 2a (planner)", color="#E45756", alpha=0.9)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 2,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Dataset Scale")
    ax.set_ylabel("Latency (ms, p50)")
    ax.set_title("Filtered Similarity Search: Baseline vs Planner")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    ASSETS_DIR.mkdir(exist_ok=True)
    fig.savefig(ASSETS_DIR / "compare_filtered_similarity.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {ASSETS_DIR / 'compare_filtered_similarity.png'}")


def chart_all_categories_comparison(p15: dict, p2a: dict) -> None:
    """Grouped bar chart: all categories at 100K, baseline vs planner."""
    scale = 100000
    if scale not in p15 or scale not in p2a:
        print("  Skipping: need 100K in both phases")
        return

    categories = [
        "temporal", "comparison", "aggregation", "absence",
        "time_series", "causal_trace", "similarity",
    ]
    cat_labels = [
        "Temporal", "Comparison", "Aggregation", "Absence",
        "Time-series", "Causal\nTrace", "Similarity",
    ]

    p15_vals = [_category_avg(p15[scale], cat) for cat in categories]
    p2a_vals = [_category_avg(p2a[scale], cat) for cat in categories]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(categories))
    width = 0.35

    bars1 = ax.bar([xi - width/2 for xi in x], p15_vals, width,
                   label="Phase 1.5 (baseline)", color="#4C78A8", alpha=0.9)
    bars2 = ax.bar([xi + width/2 for xi in x], p2a_vals, width,
                   label="Phase 2a (planner)", color="#E45756", alpha=0.9)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 3:
                ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Query Category")
    ax.set_ylabel("Latency (ms, p50) — 100K episodes")
    ax.set_title("Phase 1.5 vs Phase 2a: All Query Categories at 100K")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "compare_all_categories_100k.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {ASSETS_DIR / 'compare_all_categories_100k.png'}")


def chart_scaling_comparison(p15: dict, p2a: dict) -> None:
    """Line chart: similarity scaling across both phases."""
    scales = sorted(set(p15.keys()) & set(p2a.keys()))
    labels = [f"{s//1000}K" for s in scales]

    # Filtered similarity
    p15_filtered = []
    p2a_filtered = []
    # All similarity avg
    p15_sim_avg = []
    p2a_sim_avg = []

    for s in scales:
        q15 = _get_query(p15[s], "similar_episodes_filtered")
        q2a = _get_query(p2a[s], "similar_episodes_filtered")
        p15_filtered.append(q15["p50_ms"] if q15 else 0)
        p2a_filtered.append(q2a["p50_ms"] if q2a else 0)
        p15_sim_avg.append(_category_avg(p15[s], "similarity"))
        p2a_sim_avg.append(_category_avg(p2a[s], "similarity"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: filtered similarity
    ax1.plot(labels, p15_filtered, "o-", color="#4C78A8", linewidth=2,
             markersize=8, label="Phase 1.5")
    ax1.plot(labels, p2a_filtered, "s-", color="#E45756", linewidth=2,
             markersize=8, label="Phase 2a")
    for i, (v1, v2) in enumerate(zip(p15_filtered, p2a_filtered)):
        ax1.annotate(f"{v1:.0f}", (labels[i], v1), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, color="#4C78A8")
        ax1.annotate(f"{v2:.0f}", (labels[i], v2), textcoords="offset points",
                     xytext=(0, -15), ha="center", fontsize=9, color="#E45756")
    ax1.set_title("Filtered Similarity (status='failure')")
    ax1.set_xlabel("Scale")
    ax1.set_ylabel("Latency (ms, p50)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Right: all similarity avg
    ax2.plot(labels, p15_sim_avg, "o-", color="#4C78A8", linewidth=2,
             markersize=8, label="Phase 1.5")
    ax2.plot(labels, p2a_sim_avg, "s-", color="#E45756", linewidth=2,
             markersize=8, label="Phase 2a")
    for i, (v1, v2) in enumerate(zip(p15_sim_avg, p2a_sim_avg)):
        ax2.annotate(f"{v1:.0f}", (labels[i], v1), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, color="#4C78A8")
        ax2.annotate(f"{v2:.0f}", (labels[i], v2), textcoords="offset points",
                     xytext=(0, -15), ha="center", fontsize=9, color="#E45756")
    ax2.set_title("All Similarity (avg p50)")
    ax2.set_xlabel("Scale")
    ax2.set_ylabel("Latency (ms, p50)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "compare_similarity_scaling.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {ASSETS_DIR / 'compare_similarity_scaling.png'}")


def main():
    p15 = load_phase("phase1.5")
    p2a = load_phase("phase2a")

    if not p15 or not p2a:
        print("Need both phase1.5/ and phase2a/ in benchmarks/results/")
        return

    print(f"Phase 1.5: scales {sorted(p15.keys())}")
    print(f"Phase 2a:  scales {sorted(p2a.keys())}")

    chart_filtered_similarity_comparison(p15, p2a)
    chart_all_categories_comparison(p15, p2a)
    chart_scaling_comparison(p15, p2a)
    print("\nDone.")


if __name__ == "__main__":
    main()
