"""Synthetic data generator for EpisodicDB benchmarks.

Generates realistic agent memory data at configurable scale:
- Episodes with embeddings, tool calls, decisions, and facts
- Temporal fact chains (supersession patterns)
- Failure clusters for causal trace queries
"""

from __future__ import annotations

import math
import random
import time
from datetime import datetime, timedelta, timezone

from episodicdb.db import EpisodicDB
from episodicdb.schema import EMBEDDING_DIM

STATUSES = ["success", "failure", "partial", "aborted"]
STATUS_WEIGHTS = [0.6, 0.2, 0.15, 0.05]

TASK_TYPES = [
    "code_review", "bug_fix", "feature_dev", "refactor",
    "test_writing", "deployment", "debug", "docs",
    "investigation", "config_change",
]

TOOL_NAMES = [
    "Read", "Edit", "Write", "Bash", "Grep", "Glob",
    "WebSearch", "WebFetch", "Agent", "AskUser",
    "GitCommit", "GitPush", "RunTests", "Deploy",
    "DBQuery", "APICall",
]

OUTCOME_TYPES = ["success", "failure", "timeout", "error"]
OUTCOME_WEIGHTS = [0.7, 0.15, 0.1, 0.05]

FACT_KEYS = [
    "preferred_language", "current_project", "os_type", "editor",
    "deploy_target", "db_engine", "test_framework", "ci_tool",
    "branch_strategy", "code_style", "package_manager", "cloud_provider",
    "monitoring_tool", "auth_method", "api_version", "timezone",
]

FACT_VALUES = {
    "preferred_language": ["python", "typescript", "go", "rust", "java"],
    "current_project": ["alpha-push", "core-system", "cdp-analytics", "edge-lambda", "data-pipeline"],
    "os_type": ["macos", "linux", "windows"],
    "editor": ["cursor", "vscode", "neovim", "datagrip"],
    "deploy_target": ["ecs", "lambda", "k8s", "ec2"],
    "db_engine": ["postgresql", "duckdb", "redis", "dynamodb"],
    "test_framework": ["pytest", "jest", "vitest", "unittest"],
    "ci_tool": ["github-actions", "circleci", "jenkins"],
    "branch_strategy": ["trunk-based", "gitflow", "github-flow"],
    "code_style": ["black+ruff", "prettier+eslint", "gofmt"],
    "package_manager": ["pip", "pnpm", "npm", "poetry"],
    "cloud_provider": ["aws", "gcp", "azure"],
    "monitoring_tool": ["datadog", "grafana", "cloudwatch"],
    "auth_method": ["oauth2", "api-key", "jwt", "session"],
    "api_version": ["v1", "v2", "v3"],
    "timezone": ["Asia/Seoul", "UTC", "US/Pacific"],
}

# Tools that tend to fail together (for causal trace benchmarks)
FAILURE_CHAINS = [
    ["Bash", "RunTests", "Deploy"],       # test failure → deploy failure
    ["DBQuery", "APICall", "Write"],       # DB error → cascade
    ["GitPush", "Deploy", "APICall"],      # push conflict → deploy fail
    ["Read", "Edit", "RunTests"],          # bad edit → test fail
]


def _random_embedding(cluster_id: int = 0, noise: float = 0.3) -> list[float]:
    """Generate a pseudo-random embedding with optional clustering.

    Embeddings in the same cluster_id will be closer together in cosine space,
    which makes similarity search benchmarks meaningful.
    """
    rng = random.Random(cluster_id * 1000)
    base = [rng.gauss(0, 1) for _ in range(EMBEDDING_DIM)]
    # Add noise
    noisy = [b + random.gauss(0, noise) for b in base]
    # L2 normalize
    norm = math.sqrt(sum(x * x for x in noisy))
    if norm > 0:
        noisy = [x / norm for x in noisy]
    return noisy


def _random_datetime(start: datetime, end: datetime) -> datetime:
    delta = (end - start).total_seconds()
    offset = random.random() * delta
    return start + timedelta(seconds=offset)


def generate(
    db: EpisodicDB,
    n_episodes: int = 10_000,
    *,
    seed: int = 42,
    embedding_ratio: float = 0.3,
    n_clusters: int = 20,
    tools_per_episode: tuple[int, int] = (1, 8),
    decisions_per_episode: tuple[int, int] = (0, 3),
    fact_changes: int = 200,
    time_span_days: int = 90,
    progress_every: int = 1000,
) -> dict:
    """Generate synthetic benchmark data.

    Returns stats dict with counts and timing.
    """
    random.seed(seed)
    start_time = time.monotonic()

    now = datetime.now(timezone.utc)
    time_start = now - timedelta(days=time_span_days)

    stats = {
        "episodes": 0,
        "tool_calls": 0,
        "decisions": 0,
        "facts": 0,
        "embeddings": 0,
    }

    # --- Facts (temporal chains) ---
    fact_start = time.monotonic()
    for i in range(fact_changes):
        key = random.choice(FACT_KEYS)
        value = random.choice(FACT_VALUES[key])
        valid_from = _random_datetime(time_start, now)
        db.record_fact(key=key, value=value, valid_from=valid_from)
        stats["facts"] += 1
    fact_elapsed = time.monotonic() - fact_start

    # --- Episodes + tool calls + decisions ---
    episode_ids = []
    write_start = time.monotonic()

    for i in range(n_episodes):
        status = random.choices(STATUSES, STATUS_WEIGHTS)[0]
        task_type = random.choice(TASK_TYPES)
        started_at = _random_datetime(time_start, now)
        duration = timedelta(minutes=random.randint(1, 120))
        ended_at = started_at + duration if status in ("success", "failure") else None

        # Embedding for a fraction of episodes
        embedding = None
        if random.random() < embedding_ratio:
            cluster = random.randint(0, n_clusters - 1)
            embedding = _random_embedding(cluster)
            stats["embeddings"] += 1

        episode_id = db.record_episode(
            status=status,
            task_type=task_type,
            started_at=started_at,
            ended_at=ended_at,
            embedding=embedding,
            tags=random.sample(TASK_TYPES, k=random.randint(0, 3)),
        )
        episode_ids.append((episode_id, started_at, status))
        stats["episodes"] += 1

        # Tool calls
        n_tools = random.randint(*tools_per_episode)

        # Inject failure chains for causal trace data
        if status == "failure" and random.random() < 0.4:
            chain = random.choice(FAILURE_CHAINS)
            tool_time = started_at
            for j, tool in enumerate(chain):
                tool_time += timedelta(seconds=random.randint(1, 30))
                outcome = "failure" if j == len(chain) - 1 else "success"
                db.record_tool_call(
                    episode_id=episode_id,
                    tool_name=tool,
                    outcome=outcome,
                    duration_ms=random.randint(50, 5000),
                    called_at_override=tool_time,
                    error_message="chain failure" if outcome == "failure" else None,
                )
                stats["tool_calls"] += 1
            n_tools = max(0, n_tools - len(chain))

        # Regular tool calls
        tool_time = started_at
        for _ in range(n_tools):
            tool_time += timedelta(seconds=random.randint(1, 60))
            tool = random.choice(TOOL_NAMES)
            outcome = random.choices(OUTCOME_TYPES, OUTCOME_WEIGHTS)[0]
            db.record_tool_call(
                episode_id=episode_id,
                tool_name=tool,
                outcome=outcome,
                duration_ms=random.randint(10, 10000),
                called_at_override=tool_time,
                error_message=f"{tool} failed" if outcome != "success" else None,
            )
            stats["tool_calls"] += 1

        # Decisions
        n_decisions = random.randint(*decisions_per_episode)
        for _ in range(n_decisions):
            db.record_decision(
                episode_id=episode_id,
                rationale=f"Chose approach for {task_type}",
                decision_type=random.choice(["architecture", "tool_selection", "strategy", "retry"]),
                alternatives=[f"alt_{j}" for j in range(random.randint(1, 4))],
                outcome=random.choice(["good", "acceptable", "poor", None]),
            )
            stats["decisions"] += 1

        if (i + 1) % progress_every == 0:
            elapsed = time.monotonic() - write_start
            rate = (i + 1) / elapsed
            print(f"  [{i + 1:,}/{n_episodes:,}] {rate:.0f} episodes/sec", flush=True)

    write_elapsed = time.monotonic() - write_start
    total_elapsed = time.monotonic() - start_time

    stats["timing"] = {
        "facts_sec": round(fact_elapsed, 3),
        "writes_sec": round(write_elapsed, 3),
        "total_sec": round(total_elapsed, 3),
        "episodes_per_sec": round(n_episodes / write_elapsed, 1),
    }

    return stats
