import pytest
from src.db import EpisodicDB
from src import EpisodicDBError


def test_connect_in_memory():
    with EpisodicDB(agent_id="test", path=":memory:") as db:
        assert db._conn is not None


def test_connect_default_path(tmp_path):
    db = EpisodicDB(agent_id="agent-a", path=str(tmp_path / "test.db"))
    db.close()
    assert (tmp_path / "test.db").exists()


def test_record_episode_returns_uuid(db):
    ep_id = db.record_episode(status="success")
    assert isinstance(ep_id, str)
    assert len(ep_id) == 36


def test_record_episode_stores_agent_id(db):
    db.record_episode(status="failure", task_type="file_edit")
    row = db._conn.execute(
        "SELECT agent_id, task_type FROM episodes"
    ).fetchone()
    assert row[0] == "test"
    assert row[1] == "file_edit"


def test_record_episode_invalid_status(db):
    with pytest.raises(Exception):
        db.record_episode(status="invalid")


def test_record_episode_embedding_dimension_error(db):
    with pytest.raises(ValueError, match="Expected 1536 dimensions"):
        db.record_episode(status="success", embedding=[0.1, 0.2])


def test_record_tool_call_returns_uuid(db):
    ep_id = db.record_episode(status="success")
    tc_id = db.record_tool_call(
        episode_id=ep_id,
        tool_name="Edit",
        outcome="failure",
        duration_ms=120,
        error_message="permission denied",
    )
    assert isinstance(tc_id, str)
    assert len(tc_id) == 36


def test_record_tool_call_fk_constraint(db):
    with pytest.raises(Exception):
        db.record_tool_call(
            episode_id="00000000-0000-0000-0000-000000000000",
            tool_name="Edit",
            outcome="failure",
        )


def test_record_decision_returns_uuid(db):
    ep_id = db.record_episode(status="success")
    dec_id = db.record_decision(
        episode_id=ep_id,
        rationale="Read before modify",
        decision_type="safety_check",
        alternatives=["skip_read", "read_first"],
        outcome="read_first",
    )
    assert isinstance(dec_id, str)
    assert len(dec_id) == 36
