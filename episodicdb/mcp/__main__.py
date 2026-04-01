from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="EpisodicDB MCP Server")
    parser.add_argument("--agent-id", required=True, help="Default agent ID")
    parser.add_argument("--db", default=None, help="DuckDB file path (default: ~/.episodicdb/{agent_id}.db)")
    parser.add_argument("--daemon", action="store_true", help="Use daemon mode (allows multiple MCP sessions)")
    parser.add_argument("--client-type", default=None, help="Client identifier (e.g. claude-code, openclaw)")
    args = parser.parse_args()

    from episodicdb.mcp.server import serve
    serve(agent_id=args.agent_id, db_path=args.db, use_daemon=args.daemon, client_type=args.client_type)


if __name__ == "__main__":
    main()
