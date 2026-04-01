from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.db import EpisodicDB


@pytest.fixture
def db():
    with EpisodicDB(agent_id="test", path=":memory:") as db:
        yield db


@pytest.fixture
def seeded_db(db):
    """20 episodes + 60 tool_calls. For time-series/aggregation tests."""
    now = datetime.now(tz=timezone.utc)

    tools = ["Edit", "Bash", "Read", "Write", "WebSearch"]
    outcomes = ["success", "failure", "success", "success", "failure"]

    for i in range(20):
        started = now - timedelta(days=13 - i % 14, hours=i % 24)
        ep_status = "failure" if i % 3 == 0 else "success"
        ep_id = db.record_episode(
            status=ep_status,
            task_type="file_edit" if i % 2 == 0 else "web_search",
            started_at=started,
            ended_at=started + timedelta(minutes=5),
        )

        for j in range(3):
            tool = tools[j % len(tools)]
            oc = outcomes[j % len(outcomes)]
            db.record_tool_call(
                episode_id=ep_id,
                tool_name=tool,
                outcome=oc,
                duration_ms=50 + j * 10,
                called_at_override=started + timedelta(seconds=j * 10),
            )

    return db
