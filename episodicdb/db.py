from __future__ import annotations

from pathlib import Path

import duckdb

from episodicdb.analytics import AnalyticsMixin
from episodicdb.writer import WriterMixin
from episodicdb.schema import (
    CALLED_AT_INDEX_DDL,
    DECISIONS_DDL,
    EPISODES_DDL,
    FACTS_DDL,
    FACTS_KEY_IDX_DDL,
    HNSW_INDEX_DDL,
    STARTED_AT_INDEX_DDL,
    TOOL_CALLS_DDL,
)

_DEFAULT_DIR = Path.home() / ".episodicdb"


class EpisodicDB(WriterMixin, AnalyticsMixin):
    def __init__(self, agent_id: str, path: str | None = None) -> None:
        self.agent_id = agent_id

        if path is None:
            _DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
            resolved = str(_DEFAULT_DIR / f"{agent_id}.db")
        else:
            resolved = path

        try:
            self._conn = duckdb.connect(resolved)
        except Exception as exc:
            from episodicdb import EpisodicDBError
            raise EpisodicDBError(f"Cannot open database: {resolved}") from exc

        self._load_vss()
        self._init_schema()

    def _load_vss(self) -> None:
        try:
            self._conn.execute("INSTALL vss; LOAD vss")
        except Exception as exc:
            from episodicdb import EpisodicDBError
            raise EpisodicDBError("VSS extension unavailable") from exc

    def _init_schema(self) -> None:
        self._conn.execute(EPISODES_DDL)
        self._conn.execute(TOOL_CALLS_DDL)
        self._conn.execute(DECISIONS_DDL)
        self._conn.execute(FACTS_DDL)
        try:
            self._conn.execute("SET hnsw_enable_experimental_persistence = true")
        except Exception:
            pass
        self._conn.execute(HNSW_INDEX_DDL)
        self._conn.execute(CALLED_AT_INDEX_DDL)
        self._conn.execute(STARTED_AT_INDEX_DDL)
        self._conn.execute(FACTS_KEY_IDX_DDL)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EpisodicDB":
        return self

    def __exit__(self, *_) -> None:
        self.close()
