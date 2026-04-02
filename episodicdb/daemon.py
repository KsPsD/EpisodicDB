"""EpisodicDB daemon — single process that owns the DuckDB connection.

Multiple MCP servers (Claude Code, OpenClaw, etc.) connect to this daemon
instead of opening the DB directly, avoiding DuckDB's single-writer lock.

Usage:
    python -m episodicdb.daemon --agent-id my-agent
    python -m episodicdb.daemon --agent-id my-agent --port 7823
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Lock

from episodicdb.db import EpisodicDB

_PIDFILE_DIR = Path.home() / ".episodicdb"
_DEFAULT_PORT = 7823


def _port_for_agent(agent_id: str) -> int:
    """Derive a deterministic port from agent_id to avoid collisions."""
    import hashlib
    h = int(hashlib.sha256(agent_id.encode()).hexdigest(), 16)
    return 7823 + (h % 1000)

_db: EpisodicDB | None = None
_db_lock = Lock()


def _pidfile_path(agent_id: str) -> Path:
    return _PIDFILE_DIR / f"{agent_id}.daemon.json"


def _write_pidfile(agent_id: str, port: int) -> None:
    _PIDFILE_DIR.mkdir(parents=True, exist_ok=True)
    _pidfile_path(agent_id).write_text(
        json.dumps({"pid": os.getpid(), "port": port, "agent_id": agent_id})
    )


def _remove_pidfile(agent_id: str) -> None:
    try:
        _pidfile_path(agent_id).unlink()
    except FileNotFoundError:
        pass


def read_daemon_info(agent_id: str) -> dict | None:
    """Read daemon connection info. Returns {"pid", "port", "agent_id"} or None."""
    path = _pidfile_path(agent_id)
    if not path.exists():
        return None
    try:
        info = json.loads(path.read_text())
        # Check if process is alive
        os.kill(info["pid"], 0)
        return info
    except (OSError, json.JSONDecodeError, KeyError):
        # Stale pidfile
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return None


def _dispatch(method: str, args: dict) -> dict:
    """Call an EpisodicDB method and return the result."""
    assert _db is not None

    # Extract agent_id without mutating the caller's dict
    call_args = {k: v for k, v in args.items() if k != "agent_id"}
    agent_id = args.get("agent_id")

    with _db_lock:
        original = _db.agent_id
        if agent_id is not None:
            _db.agent_id = agent_id
        try:
            return _call_method(method, call_args)
        finally:
            _db.agent_id = original


def _call_method(method: str, args: dict) -> dict:
    """Route method name to EpisodicDB method."""
    assert _db is not None

    # Parse datetime strings
    for key in ("started_at", "ended_at", "called_at_override", "valid_from", "as_of"):
        if key in args and args[key] is not None:
            args[key] = datetime.fromisoformat(args[key])

    fn = getattr(_db, method, None)
    if fn is None or method.startswith("_"):
        return {"error": f"Unknown method: {method}"}

    try:
        result = fn(**args)
        return {"result": _serialize(result)}
    except Exception as e:
        return {"error": str(e)}


def _serialize(obj):
    """Make result JSON-serializable."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


class DaemonHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/call":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            method = body.get("method", "")
            args = body.get("args", {})
            result = _dispatch(method, args)
            self._respond(200, result)
        elif self.path == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not found"})

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not found"})

    def _respond(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        # Suppress request logging
        pass


def run_daemon(agent_id: str, port: int = _DEFAULT_PORT, db_path: str | None = None) -> None:
    global _db

    _db = EpisodicDB(agent_id=agent_id, path=db_path)
    _write_pidfile(agent_id, port)

    def cleanup(*_):
        _remove_pidfile(agent_id)
        if _db:
            _db.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    server = HTTPServer(("127.0.0.1", port), DaemonHandler)
    try:
        server.serve_forever()
    finally:
        cleanup()


def main():
    parser = argparse.ArgumentParser(description="EpisodicDB Daemon")
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    run_daemon(agent_id=args.agent_id, port=args.port, db_path=args.db)


if __name__ == "__main__":
    main()
