"""Tests for the daemon + client architecture."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest

from episodicdb.client import EpisodicDBClient
from episodicdb.daemon import read_daemon_info


@pytest.fixture()
def daemon_agent_id():
    """Unique agent_id per test run to avoid pidfile collisions."""
    return f"test-daemon-{os.getpid()}"


@pytest.fixture()
def daemon_port():
    """Use a unique port to avoid conflicts."""
    return 18900 + os.getpid() % 1000


@pytest.fixture()
def daemon_process(tmp_path, daemon_port, daemon_agent_id):
    """Start a daemon process for testing."""
    db_path = str(tmp_path / "test.db")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "episodicdb.daemon",
            "--agent-id", daemon_agent_id,
            "--port", str(daemon_port),
            "--db", db_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for daemon to be ready
    for _ in range(30):
        try:
            from urllib.request import urlopen
            urlopen(f"http://127.0.0.1:{daemon_port}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    yield proc
    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture()
def client(daemon_process, daemon_port, daemon_agent_id):
    """Create a client connected to the test daemon."""
    return EpisodicDBClient(
        agent_id=daemon_agent_id,
        port=daemon_port,
        auto_start=False,
    )


def test_health_check(daemon_process, daemon_port):
    from urllib.request import urlopen
    import json
    resp = urlopen(f"http://127.0.0.1:{daemon_port}/health")
    body = json.loads(resp.read())
    assert body == {"status": "ok"}


def test_record_and_query_episode(client):
    ep_id = client.record_episode(status="failure", task_type="test_task")
    assert isinstance(ep_id, str)
    assert len(ep_id) > 0


def test_record_tool_call(client):
    ep_id = client.record_episode(status="success")
    tc_id = client.record_tool_call(
        episode_id=ep_id,
        tool_name="grep",
        outcome="success",
        duration_ms=50,
    )
    assert isinstance(tc_id, str)


def test_record_decision(client):
    ep_id = client.record_episode(status="success")
    dec_id = client.record_decision(
        episode_id=ep_id,
        rationale="chose grep over find for speed",
        decision_type="tool_selection",
    )
    assert isinstance(dec_id, str)


def test_record_fact_and_query(client):
    client.record_fact(key="test_key", value="test_value")
    facts = client.facts_as_of(as_of=datetime.now(tz=timezone.utc))
    assert any(f["key"] == "test_key" and f["value"] == "test_value" for f in facts)


def test_fact_supersession(client):
    client.record_fact(key="color", value="red")
    client.record_fact(key="color", value="blue")
    history = client.fact_history(key="color")
    assert len(history) == 2
    assert history[0]["value"] == "red"
    assert history[1]["value"] == "blue"


def test_analytics_empty(client):
    result = client.top_failing_tools(days=7, limit=5)
    assert result == []

    result = client.never_succeeded_tools()
    assert result == []


def test_compare_periods(client):
    result = client.compare_periods(metric="failure_rate", days=7)
    assert "period_a" in result
    assert "period_b" in result
    assert "delta" in result


def test_two_clients_same_daemon(daemon_process, daemon_port, daemon_agent_id):
    """Two clients can connect to the same daemon simultaneously."""
    client1 = EpisodicDBClient(
        agent_id=daemon_agent_id, port=daemon_port, auto_start=False,
    )
    client2 = EpisodicDBClient(
        agent_id=daemon_agent_id, port=daemon_port, auto_start=False,
    )

    ep1 = client1.record_episode(status="success", task_type="from_client_1")
    ep2 = client2.record_episode(status="failure", task_type="from_client_2")

    assert ep1 != ep2

    # Both can query
    facts1 = client1.top_failing_tools(days=7)
    facts2 = client2.top_failing_tools(days=7)
    assert isinstance(facts1, list)
    assert isinstance(facts2, list)


def test_session_lifecycle(client):
    """start_session / end_session round-trip through daemon."""
    session_id = client.start_session(client_type="test-client")
    assert isinstance(session_id, str) and len(session_id) > 0

    # Record episode with session
    ep_id = client.record_episode(status="success", session_id=session_id)
    assert isinstance(ep_id, str)

    # End session should not raise
    client.end_session(session_id)


def test_unknown_method_returns_error(daemon_process, daemon_port):
    """Calling a non-existent method returns an error, not a crash."""
    import json
    from urllib.request import Request, urlopen

    body = json.dumps({"method": "no_such_method", "args": {}}).encode()
    req = Request(
        f"http://127.0.0.1:{daemon_port}/call",
        data=body,
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    resp = json.loads(urlopen(req, timeout=5).read())
    assert "error" in resp
    assert "Unknown method" in resp["error"]


def test_private_method_blocked(daemon_process, daemon_port):
    """Methods starting with _ are blocked."""
    import json
    from urllib.request import Request, urlopen

    body = json.dumps({"method": "_init_schema", "args": {}}).encode()
    req = Request(
        f"http://127.0.0.1:{daemon_port}/call",
        data=body,
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    resp = json.loads(urlopen(req, timeout=5).read())
    assert "error" in resp
