# EpisodicDB

**An OLAP-based memory engine for AI agents.**

Existing agent memory systems (Mem0, Zep, Letta) are designed as *search systems*. EpisodicDB treats agent memory as an *analytics problem* — aggregation, time-series patterns, causal tracing, temporal facts, and vector similarity all in a single query engine.

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
| Similarity | "Seen a similar error before?" | Yes |
| Aggregation | "How many tool failures this week?" | No |
| Time-series | "Do failures spike in the afternoon?" | No |
| Causal trace | "What tool ran right before failures?" | No |
| Comparison | "Worse than last week?" | No |
| Absence | "Tools that never succeeded?" | No |
| Temporal | "What was the user's timezone last Tuesday?" | No |

EpisodicDB answers all of them. Vector similarity is just another SQL operator.

## Architecture

```
EpisodicDB
├── WriterMixin      record_episode / record_tool_call / record_decision / record_fact
├── AnalyticsMixin   6 analytics methods + vector similarity
└── TemporalMixin    facts_as_of / fact_history

Engine: DuckDB (OLAP) + VSS extension (HNSW vector index)
Schema: episodes + tool_calls + decisions + facts
```

## Install

```bash
pip install episodicdb
```

## Quick Start

### Python SDK

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

    # Record temporal facts (auto-supersedes previous values)
    db.record_fact("user_timezone", "Asia/Seoul", episode_id=ep_id)
    db.record_fact("user_timezone", "America/New_York")  # closes previous

    # Analyze patterns
    print(db.top_failing_tools(days=7))
    # [{"tool_name": "Edit", "failures": 5}, ...]

    print(db.before_failure_sequence("Edit"))
    # [{"prev_tool": "Bash", "count": 4}, ...]

    print(db.compare_periods("failure_rate", days=7))
    # {"period_a": 0.32, "period_b": 0.18, "delta": 0.14}

    # Time-travel query
    from datetime import datetime
    print(db.facts_as_of(datetime(2025, 3, 15)))
    # [{"key": "user_timezone", "value": "Asia/Seoul", ...}]
```

### MCP Server (Claude, OpenAI Agents SDK)

EpisodicDB ships an MCP server with 12 tools over stdio.

```bash
episodicdb-mcp --agent-id my-agent
episodicdb-mcp --agent-id my-agent --db ./memory.db
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "episodicdb": {
      "command": "episodicdb-mcp",
      "args": ["--agent-id", "my-agent"]
    }
  }
}
```

**Claude Code** (`.mcp.json`):

```json
{
  "mcpServers": {
    "episodicdb": {
      "command": "episodicdb-mcp",
      "args": ["--agent-id", "my-agent"]
    }
  }
}
```

**OpenAI Agents SDK**:

```python
from agents import Agent
from agents.mcp import MCPServerStdio

agent = Agent(
    name="my-agent",
    instructions="You have access to episodic memory.",
    mcp_servers=[MCPServerStdio(
        command="episodicdb-mcp",
        args=["--agent-id", "my-agent"],
    )],
)
```

### Auto-recording (recommended)

Add these rules to your `CLAUDE.md` (or system prompt) so the agent records episodes automatically:

```markdown
## EpisodicDB — auto-recording rules

EpisodicDB MCP server is connected. Follow these rules to record work automatically.

### On session start
- Call `record_episode` (status: "partial", task_type: type of work)
- Remember the episode ID for subsequent tool call / decision records

### During work
- Record significant tool results with `record_tool_call` (success/failure, duration)
- Record important decisions with `record_decision` (rationale, alternatives)
- Record new user/project facts with `record_fact` (e.g. preferred_language, current_project)

### On session end
- Update the episode status to match the final outcome (success/failure/partial/aborted)

### Guidelines
- Don't interrupt the user's workflow to record — do it in the background
- Only record meaningful actions, not every small step
- Use english snake_case for fact keys (e.g. preferred_language, deploy_target)
```

## API

### Writer

```python
db.record_episode(status, task_type=None, context=None,
                  embedding=None, tags=None,
                  started_at=None, ended_at=None) -> str  # episode UUID

db.record_tool_call(episode_id, tool_name, outcome,
                    parameters=None, result=None,
                    duration_ms=None, error_message=None) -> str

db.record_decision(episode_id, rationale,
                   decision_type=None, alternatives=None,
                   outcome=None) -> str

db.record_fact(key, value, episode_id=None,
               valid_from=None) -> str  # auto-supersedes previous
```

### Analytics

| Method | Description |
|--------|-------------|
| `top_failing_tools(days, limit)` | Most-failed tools in the last N days |
| `hourly_failure_rate(days)` | Failure count by hour of day |
| `before_failure_sequence(tool_name, lookback)` | Tools that precede failures |
| `compare_periods(metric, days)` | Period-over-period comparison |
| `never_succeeded_tools()` | Tools with zero successful calls |
| `similar_episodes(embedding, status, limit)` | Vector similarity + SQL filter |

### Temporal Facts

Facts are key-value pairs with automatic temporal validity. Recording a new value for the same key closes the previous one.

```python
db.record_fact("preferred_model", "gpt-4o")
# later...
db.record_fact("preferred_model", "claude-sonnet")  # supersedes gpt-4o

db.facts_as_of(some_datetime)   # point-in-time snapshot
db.fact_history("preferred_model")  # full change log
```

### Persistence

```python
EpisodicDB(agent_id="my-agent")                    # ~/.episodicdb/my-agent.db
EpisodicDB(agent_id="my-agent", path="./x.db")    # explicit path
EpisodicDB(agent_id="my-agent", path=":memory:")  # in-memory (testing)
```

### Embeddings

Built-in helpers for popular providers (lazy imports, no hard dependencies):

```bash
pip install episodicdb[openai]   # OpenAI
pip install episodicdb[voyage]   # Voyage AI
pip install episodicdb[ollama]   # Ollama (local)
pip install episodicdb[all]      # all providers
```

```python
from episodicdb import embeddings

# OpenAI
vec = embeddings.openai("what the agent was doing")

# Voyage AI
vec = embeddings.voyage("what the agent was doing")

# Local Ollama
vec = embeddings.ollama("what the agent was doing")

db.record_episode(status="success", embedding=vec)
db.similar_episodes(vec, status="failure", limit=5)
```

Or bring your own:

```python
db.record_episode(status="success", embedding=your_1536_dim_list)
```

## Development

```bash
git clone https://github.com/KsPsD/EpisodicDB
cd EpisodicDB
pip install -e ".[dev]"
pytest
```

## Stack

- [DuckDB](https://duckdb.org/) — embedded OLAP engine
- [DuckDB VSS](https://duckdb.org/docs/extensions/vss) — HNSW vector index
- [MCP](https://modelcontextprotocol.io/) — Model Context Protocol server
- Python 3.11+

## License

MIT
