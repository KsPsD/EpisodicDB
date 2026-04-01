# EpisodicDB

**An OLAP-based memory engine for AI agents.**

Existing agent memory systems (Mem0, Zep, Letta) are designed as *search systems*. EpisodicDB treats agent memory as an *analytics problem* — aggregation, time-series patterns, causal tracing, and vector similarity all in a single query engine.

```sql
-- "Which tools failed most this week?"
SELECT tool_name, COUNT(*) AS failures
FROM tool_calls
WHERE outcome = 'failure'
  AND called_at >= NOW() - INTERVAL '7 days'
GROUP BY tool_name
ORDER BY failures DESC;

-- "Find past episodes similar to this context that succeeded"
SELECT *, array_cosine_distance(context_embedding, ?) AS dist
FROM episodes
WHERE status = 'success'
ORDER BY dist ASC
LIMIT 5;
```

## The Problem

| Query type | Example | Vector search? |
|------------|---------|----------------|
| Similarity | "Seen a similar error before?" | ✅ |
| Aggregation | "How many tool failures this week?" | ❌ |
| Time-series | "Do failures spike in the afternoon?" | ❌ |
| Causal trace | "What tool ran right before failures?" | ❌ |
| Comparison | "Worse than last week?" | ❌ |
| Absence | "Tools that never succeeded?" | ❌ |

EpisodicDB answers all of them. Vector similarity is just another SQL operator.

## Architecture

```
EpisodicDB
├── WriterMixin          record_episode / record_tool_call / record_decision
└── AnalyticsMixin       6 analytics methods

Engine: DuckDB (OLAP) + VSS extension (HNSW vector index)
Schema: episodes + tool_calls + decisions
```

## Quick Start

```bash
pip install episodicdb  # coming soon — for now: pip install -e .
```

```python
from episodicdb import EpisodicDB

with EpisodicDB(agent_id="my-agent") as db:
    # Record what happened
    ep_id = db.record_episode(
        status="failure",
        task_type="file_edit",
        context={"file": "auth.py", "error": "permission denied"},
    )
    db.record_tool_call(ep_id, "Edit", "failure",
                        duration_ms=120, error_message="permission denied")
    db.record_tool_call(ep_id, "Bash", "success", duration_ms=50)

    # Analyze patterns
    print(db.top_failing_tools(days=7))
    # → [{"tool_name": "Edit", "failures": 5}, ...]

    print(db.before_failure_sequence("Edit"))
    # → [{"prev_tool": "Bash", "count": 4}, ...]

    print(db.compare_periods("failure_rate", days=7))
    # → {"period_a": 0.32, "period_b": 0.18, "delta": 0.14}
```

## API

### Writer

```python
db.record_episode(
    status,           # "success" | "failure" | "partial" | "aborted"
    task_type=None,   # str
    context=None,     # dict (stored as JSON)
    embedding=None,   # list[float] — 1536-dim, externally generated
    tags=None,        # list[str]
    started_at=None,  # datetime (defaults to NOW())
    ended_at=None,    # datetime
) -> str              # episode_id (UUID)

db.record_tool_call(episode_id, tool_name, outcome, ...) -> str
db.record_decision(episode_id, rationale, ...) -> str
```

### Analytics

| Method | Description |
|--------|-------------|
| `top_failing_tools(days, limit)` | Most-failed tools in the last N days |
| `hourly_failure_rate(days)` | Failure count by hour of day |
| `before_failure_sequence(tool_name, lookback)` | Tools that precede failures |
| `compare_periods(metric, days)` | Period A vs period B comparison |
| `never_succeeded_tools()` | Tools with zero successful calls |
| `similar_episodes(embedding, status, limit)` | Vector similarity + SQL filter |

### Persistence

```python
EpisodicDB(agent_id="my-agent")                    # ~/.episodicdb/my-agent.db
EpisodicDB(agent_id="my-agent", path="./x.db")    # explicit path
EpisodicDB(agent_id="my-agent", path=":memory:")  # in-memory (testing)
```

### Embeddings

EpisodicDB does not generate embeddings. Pass them in:

```python
import openai

response = openai.embeddings.create(
    model="text-embedding-3-small",
    input="what the agent was doing"
)
embedding = response.data[0].embedding  # list[float], 1536 dims

db.record_episode(status="success", embedding=embedding)
```

## Development

```bash
git clone https://github.com/KsPsD/EpisodicDB
cd EpisodicDB
pip install -e ".[dev]"
pytest -v
```

## Stack

- [DuckDB](https://duckdb.org/) — embedded OLAP engine
- [DuckDB VSS](https://duckdb.org/docs/extensions/vss) — HNSW vector index
- Python 3.11+

## Status

`alpha` — core DB layer implemented. MCP server interface coming next.

## License

MIT
