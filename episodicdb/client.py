"""EpisodicDB client — talks to the daemon over HTTP.

Same interface as EpisodicDB, but routes all calls through the daemon
to avoid DuckDB's single-writer lock.

If no daemon is running, auto-starts one in the background.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.error import URLError
from urllib.request import Request, urlopen

from episodicdb.daemon import _DEFAULT_PORT, _port_for_agent, read_daemon_info

_PIDFILE_DIR = Path.home() / ".episodicdb"


class EpisodicDBClient:
    """Drop-in replacement for EpisodicDB that routes through the daemon."""

    def __init__(
        self,
        agent_id: str,
        port: int | None = None,
        db_path: str | None = None,
        auto_start: bool = True,
    ) -> None:
        self.agent_id = agent_id
        self._db_path = db_path

        info = read_daemon_info(agent_id)
        if info is not None:
            self._port = info["port"]
        elif port is not None:
            self._port = port
        elif auto_start:
            self._port = _port_for_agent(agent_id)
            self._start_daemon()
        else:
            raise ConnectionError(
                f"No daemon running for agent_id={agent_id}. "
                "Start one with: python -m episodicdb.daemon --agent-id " + agent_id
            )

    def _start_daemon(self) -> None:
        """Start the daemon as a background process."""
        cmd = [
            sys.executable, "-m", "episodicdb.daemon",
            "--agent-id", self.agent_id,
            "--port", str(self._port),
        ]
        if self._db_path:
            cmd.extend(["--db", self._db_path])

        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Wait for daemon to be ready
        for _ in range(30):
            try:
                self._http("GET", "/health")
                return
            except (ConnectionError, URLError):
                time.sleep(0.1)

        raise ConnectionError("Daemon failed to start within 3 seconds")

    def _http(self, method: str, path: str, body: dict | None = None) -> dict:
        url = f"http://127.0.0.1:{self._port}{path}"
        data = json.dumps(body).encode() if body else None
        req = Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except URLError as e:
            raise ConnectionError(f"Cannot connect to daemon: {e}") from e

    def _call(self, method: str, **kwargs) -> Any:
        kwargs["agent_id"] = self.agent_id
        # Convert datetimes to ISO strings
        kwargs = {
            k: v.isoformat() if isinstance(v, datetime) else v
            for k, v in kwargs.items()
        }
        resp = self._http("POST", "/call", {"method": method, "args": kwargs})
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return resp["result"]

    # --- Session ---

    def start_session(
        self,
        client_type: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        return self._call("start_session", client_type=client_type, metadata=metadata)

    def end_session(self, session_id: str) -> None:
        self._call("end_session", session_id=session_id)

    # --- Writer ---

    def record_episode(
        self,
        status: Literal["success", "failure", "partial", "aborted"],
        task_type: str | None = None,
        context: dict | None = None,
        embedding: list[float] | None = None,
        tags: list[str] | None = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        session_id: str | None = None,
    ) -> str:
        return self._call(
            "record_episode",
            status=status, task_type=task_type, context=context,
            embedding=embedding, tags=tags,
            started_at=started_at, ended_at=ended_at,
            session_id=session_id,
        )

    def record_tool_call(
        self,
        episode_id: str,
        tool_name: str,
        outcome: Literal["success", "failure", "timeout", "error"],
        parameters: dict | None = None,
        result: dict | None = None,
        duration_ms: int | None = None,
        error_message: str | None = None,
        called_at_override: datetime | None = None,
    ) -> str:
        return self._call(
            "record_tool_call",
            episode_id=episode_id, tool_name=tool_name, outcome=outcome,
            parameters=parameters, result=result,
            duration_ms=duration_ms, error_message=error_message,
            called_at_override=called_at_override,
        )

    def record_decision(
        self,
        episode_id: str,
        rationale: str,
        decision_type: str | None = None,
        alternatives: list | None = None,
        outcome: str | None = None,
    ) -> str:
        return self._call(
            "record_decision",
            episode_id=episode_id, rationale=rationale,
            decision_type=decision_type, alternatives=alternatives,
            outcome=outcome,
        )

    def record_fact(
        self,
        key: str,
        value: str,
        episode_id: str | None = None,
        valid_from: datetime | None = None,
    ) -> str:
        return self._call(
            "record_fact",
            key=key, value=value,
            episode_id=episode_id, valid_from=valid_from,
        )

    # --- Analytics ---

    def top_failing_tools(self, days: int = 7, limit: int = 5) -> list[dict]:
        return self._call("top_failing_tools", days=days, limit=limit)

    def never_succeeded_tools(self) -> list[str]:
        return self._call("never_succeeded_tools")

    def hourly_failure_rate(self, days: int = 7) -> list[dict]:
        return self._call("hourly_failure_rate", days=days)

    def compare_periods(self, metric: str, days: int = 7) -> dict:
        return self._call("compare_periods", metric=metric, days=days)

    def before_failure_sequence(self, tool_name: str, lookback: int = 3) -> list[dict]:
        return self._call("before_failure_sequence", tool_name=tool_name, lookback=lookback)

    def similar_episodes(
        self,
        embedding: list[float],
        status: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        return self._call(
            "similar_episodes",
            embedding=embedding, status=status, limit=limit,
        )

    # --- Temporal ---

    def facts_as_of(self, as_of: datetime) -> list[dict]:
        return self._call("facts_as_of", as_of=as_of)

    def fact_history(self, key: str) -> list[dict]:
        return self._call("fact_history", key=key)

    # --- Lifecycle ---

    def close(self) -> None:
        """No-op — daemon manages its own lifecycle."""
        pass

    def stop_daemon(self) -> None:
        """Stop the daemon process."""
        info = read_daemon_info(self.agent_id)
        if info:
            try:
                os.kill(info["pid"], signal.SIGTERM)
            except OSError:
                pass

    def __enter__(self) -> "EpisodicDBClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()
