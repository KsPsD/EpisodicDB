from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="EpisodicDB MCP Server")
    parser.add_argument("--agent-id", required=True, help="Default agent ID")
    parser.add_argument("--db", default=None, help="DuckDB file path (default: ~/.episodicdb/{agent_id}.db)")
    args = parser.parse_args()

    from src.mcp.server import serve
    serve(agent_id=args.agent_id, db_path=args.db)


if __name__ == "__main__":
    main()
