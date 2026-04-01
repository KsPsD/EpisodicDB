EMBEDDING_DIM = 1536

EPISODES_DDL = """
CREATE TABLE IF NOT EXISTS episodes (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id          TEXT NOT NULL,
    started_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at          TIMESTAMPTZ,
    status            TEXT NOT NULL
                        CHECK (status IN ('success', 'failure', 'partial', 'aborted')),
    task_type         TEXT,
    context           JSON,
    context_embedding FLOAT[1536],
    tags              TEXT[]
);
"""

TOOL_CALLS_DDL = """
CREATE TABLE IF NOT EXISTS tool_calls (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id    UUID NOT NULL REFERENCES episodes(id),
    called_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tool_name     TEXT NOT NULL,
    parameters    JSON,
    result        JSON,
    outcome       TEXT NOT NULL
                    CHECK (outcome IN ('success', 'failure', 'timeout', 'error')),
    duration_ms   INTEGER,
    error_message TEXT
);
"""

DECISIONS_DDL = """
CREATE TABLE IF NOT EXISTS decisions (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id     UUID NOT NULL REFERENCES episodes(id),
    decided_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    decision_type  TEXT,
    rationale      TEXT,
    alternatives   JSON,
    outcome        TEXT
);
"""

FACTS_DDL = """
CREATE TABLE IF NOT EXISTS facts (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id    TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    valid_from  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_until TIMESTAMPTZ,
    episode_id  UUID REFERENCES episodes(id)
);
"""

FACTS_KEY_IDX_DDL = """
CREATE INDEX IF NOT EXISTS facts_key_idx
ON facts (agent_id, key, valid_from);
"""

HNSW_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS episodes_embedding_idx
ON episodes USING HNSW (context_embedding);
"""

CALLED_AT_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS tool_calls_called_at_idx
ON tool_calls (called_at);
"""

STARTED_AT_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS episodes_started_at_idx
ON episodes (started_at);
"""
