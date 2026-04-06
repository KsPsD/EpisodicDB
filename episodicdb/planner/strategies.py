"""Execution strategies for hybrid vector + SQL queries.

Each strategy implements the same interface:
    execute(conn, agent_id, embedding, predicates, limit) -> list[dict]
"""

from __future__ import annotations

from episodicdb.schema import EMBEDDING_DIM

_ALLOWED_COLUMNS = frozenset({"status", "task_type"})


def _validate_predicates(predicates: dict[str, str]) -> None:
    for col in predicates:
        if col not in _ALLOWED_COLUMNS:
            raise ValueError(f"Unsupported predicate column: {col!r}")


def filter_first(
    conn,
    agent_id: str,
    embedding: list[float],
    predicates: dict[str, str],
    limit: int,
) -> list[dict]:
    """Filter with SQL first, then brute-force cosine on the small result set.

    Best when selectivity is low (< 5%) — few rows survive the filter,
    so brute-force distance computation is cheap.
    """
    _validate_predicates(predicates)

    try:
        import numpy as np
    except ImportError:
        raise ImportError(
            "numpy is required for the hybrid planner: pip install numpy"
        )

    # Build WHERE clause from predicates
    where_parts = ["context_embedding IS NOT NULL", "agent_id = $1"]
    params: list = [agent_id]
    for i, (col, val) in enumerate(predicates.items(), start=2):
        where_parts.append(f"{col} = ${i}")
        params.append(val)

    where_clause = " AND ".join(where_parts)

    # Fetch filtered rows with their embeddings
    rows = conn.execute(
        f"""
        SELECT
            id::TEXT, agent_id, status, task_type,
            started_at, ended_at, context_embedding
        FROM episodes
        WHERE {where_clause}
        """,
        params,
    ).fetchall()

    if not rows:
        return []

    # Compute cosine distances with NumPy (vectorized)
    query_vec = np.array(embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return []

    results = []
    for r in rows:
        row_emb = r[6]
        if row_emb is None:
            continue
        row_vec = np.array(row_emb, dtype=np.float32)
        row_norm = np.linalg.norm(row_vec)
        if row_norm == 0:
            continue
        # Cosine distance = 1 - cosine_similarity
        cos_sim = np.dot(query_vec, row_vec) / (query_norm * row_norm)
        distance = 1.0 - float(cos_sim)
        results.append({
            "id": r[0],
            "agent_id": r[1],
            "status": r[2],
            "task_type": r[3],
            "started_at": r[4],
            "ended_at": r[5],
            "distance": distance,
        })

    # Sort by distance, take top-K
    results.sort(key=lambda x: x["distance"])
    return results[:limit]


def vector_first(
    conn,
    agent_id: str,
    embedding: list[float],
    predicates: dict[str, str],
    limit: int,
    fetch_limit: int | None = None,
) -> list[dict]:
    """HNSW search without predicates, then post-filter in Python.

    Best when selectivity is high (> 50%) — most rows pass the filter,
    so oversampling gives enough candidates.
    """
    _validate_predicates(predicates)

    if fetch_limit is None:
        fetch_limit = limit * 10
    fetch_limit = min(fetch_limit, 100_000)

    rows = conn.execute(
        f"""
        SELECT
            id::TEXT, agent_id, status, task_type,
            started_at, ended_at,
            array_cosine_distance(context_embedding, $1::FLOAT[{EMBEDDING_DIM}]) AS distance
        FROM episodes
        WHERE context_embedding IS NOT NULL
          AND agent_id = $2
        ORDER BY distance ASC
        LIMIT $3
        """,
        [embedding, agent_id, fetch_limit],
    ).fetchall()

    # Post-filter with predicates
    results = []
    for r in rows:
        row_dict = {
            "id": r[0],
            "agent_id": r[1],
            "status": r[2],
            "task_type": r[3],
            "started_at": r[4],
            "ended_at": r[5],
            "distance": r[6],
        }

        # Check all predicates
        if all(row_dict.get(col) == val for col, val in predicates.items()):
            results.append(row_dict)

    # Re-sort after filtering (HNSW approximate ordering not guaranteed)
    results.sort(key=lambda x: x["distance"])
    return results[:limit]


def no_filter(
    conn,
    agent_id: str,
    embedding: list[float],
    limit: int,
) -> list[dict]:
    """Pure HNSW search with no predicates. Used when predicates is empty."""
    rows = conn.execute(
        f"""
        SELECT
            id::TEXT, agent_id, status, task_type,
            started_at, ended_at,
            array_cosine_distance(context_embedding, $1::FLOAT[{EMBEDDING_DIM}]) AS distance
        FROM episodes
        WHERE context_embedding IS NOT NULL
          AND agent_id = $2
        ORDER BY distance ASC
        LIMIT $3
        """,
        [embedding, agent_id, limit],
    ).fetchall()

    return [
        {
            "id": r[0],
            "agent_id": r[1],
            "status": r[2],
            "task_type": r[3],
            "started_at": r[4],
            "ended_at": r[5],
            "distance": r[6],
        }
        for r in rows
    ]
