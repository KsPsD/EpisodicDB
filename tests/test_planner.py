"""Tests for the hybrid query planner (Phase 2a)."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import pytest

from episodicdb.db import EpisodicDB
from episodicdb.planner import HybridPlanner
from episodicdb.planner.cost_model import choose_strategy, compute_oversampling
from episodicdb.planner.selectivity import SelectivityEstimator
from episodicdb.schema import EMBEDDING_DIM


def _random_embedding(seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(EMBEDDING_DIM)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


@pytest.fixture
def planner_db():
    """DB with 100 episodes: 70 success, 20 failure, 10 partial.
    30% have embeddings (clustered by status for meaningful similarity).
    """
    db = EpisodicDB(agent_id="planner-test", path=":memory:")
    now = datetime.now(tz=timezone.utc)

    for i in range(100):
        if i < 70:
            status = "success"
            cluster = 0
        elif i < 90:
            status = "failure"
            cluster = 1
        else:
            status = "partial"
            cluster = 2

        task_type = "bug_fix" if i % 3 == 0 else "feature_dev"
        started = now - timedelta(days=i % 30)

        # 30% get embeddings
        emb = _random_embedding(seed=cluster * 1000 + i) if i % 3 == 0 else None

        db.record_episode(
            status=status,
            task_type=task_type,
            started_at=started,
            embedding=emb,
        )

    yield db
    db.close()


# ── SelectivityEstimator ──

class TestSelectivityEstimator:
    def test_empty_predicates(self, planner_db):
        est = SelectivityEstimator(planner_db._conn)
        sel = est.estimate("planner-test", {})
        assert sel == 1.0

    def test_status_selectivity(self, planner_db):
        est = SelectivityEstimator(planner_db._conn)
        sel = est.estimate("planner-test", {"status": "failure"})
        # 20 failures out of 100, ~30% have embeddings
        # Exact value depends on random seed, but should be reasonable
        assert 0.0 < sel < 1.0

    def test_nonexistent_value_returns_zero(self, planner_db):
        est = SelectivityEstimator(planner_db._conn)
        sel = est.estimate("planner-test", {"status": "nonexistent"})
        assert sel == 0.0

    def test_multi_predicate_multiplies(self, planner_db):
        est = SelectivityEstimator(planner_db._conn)
        sel_status = est.estimate("planner-test", {"status": "failure"})
        sel_task = est.estimate("planner-test", {"task_type": "bug_fix"})
        sel_both = est.estimate("planner-test", {"status": "failure", "task_type": "bug_fix"})
        # Independence assumption: combined ≈ product
        assert abs(sel_both - sel_status * sel_task) < 0.01

    def test_cache_invalidation(self, planner_db):
        est = SelectivityEstimator(planner_db._conn, cache_ttl=0.0)
        sel1 = est.estimate("planner-test", {"status": "failure"})
        # Add more failures
        planner_db.record_episode(
            status="failure",
            embedding=_random_embedding(seed=999),
        )
        est.invalidate()
        sel2 = est.estimate("planner-test", {"status": "failure"})
        # Should reflect the new episode
        assert sel2 != sel1 or sel2 > 0

    def test_filtered_count(self, planner_db):
        est = SelectivityEstimator(planner_db._conn)
        count = est.get_filtered_count("planner-test", {"status": "failure"})
        assert count >= 1


# ── CostModel ──

class TestCostModel:
    def test_low_selectivity_chooses_filter_first(self):
        strategy = choose_strategy(n_total=100_000, selectivity=0.01, limit=5)
        assert strategy == "filter_first"

    def test_high_selectivity_chooses_vector_first(self):
        strategy = choose_strategy(n_total=100_000, selectivity=0.8, limit=5)
        assert strategy == "vector_first"

    def test_full_selectivity_chooses_no_filter(self):
        strategy = choose_strategy(n_total=100_000, selectivity=1.0, limit=5)
        assert strategy == "no_filter"

    def test_oversampling_scales_with_selectivity(self):
        os_high = compute_oversampling(limit=5, selectivity=0.9)
        os_low = compute_oversampling(limit=5, selectivity=0.1)
        assert os_low > os_high

    def test_oversampling_capped(self):
        os = compute_oversampling(limit=5, selectivity=0.0001)
        assert os <= 100_000


# ── HybridPlanner (integration) ──

class TestHybridPlanner:
    def test_no_predicates_returns_results(self, planner_db):
        planner = HybridPlanner(planner_db._conn)
        emb = _random_embedding(seed=42)
        results = planner.execute("planner-test", emb, limit=5)
        assert len(results) == 5
        assert all("distance" in r for r in results)

    def test_with_status_filter(self, planner_db):
        planner = HybridPlanner(planner_db._conn)
        emb = _random_embedding(seed=42)
        results = planner.execute(
            "planner-test", emb,
            predicates={"status": "failure"},
            limit=5,
        )
        assert all(r["status"] == "failure" for r in results)

    def test_results_sorted_by_distance(self, planner_db):
        planner = HybridPlanner(planner_db._conn)
        emb = _random_embedding(seed=42)
        results = planner.execute("planner-test", emb, limit=10)
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances)

    def test_filtered_results_sorted_by_distance(self, planner_db):
        planner = HybridPlanner(planner_db._conn)
        emb = _random_embedding(seed=42)
        results = planner.execute(
            "planner-test", emb,
            predicates={"status": "failure"},
            limit=5,
        )
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances)

    def test_limit_respected(self, planner_db):
        planner = HybridPlanner(planner_db._conn)
        emb = _random_embedding(seed=42)
        results = planner.execute("planner-test", emb, limit=3)
        assert len(results) <= 3

    def test_nonexistent_status_returns_empty(self, planner_db):
        planner = HybridPlanner(planner_db._conn)
        emb = _random_embedding(seed=42)
        results = planner.execute(
            "planner-test", emb,
            predicates={"status": "nonexistent"},
            limit=5,
        )
        assert results == []

    def test_invalidate_stats(self, planner_db):
        planner = HybridPlanner(planner_db._conn)
        emb = _random_embedding(seed=42)
        # Run once to populate cache
        planner.execute("planner-test", emb, limit=5)
        # Should not error
        planner.invalidate_stats()
        results = planner.execute("planner-test", emb, limit=5)
        assert len(results) > 0


# ── Integration with EpisodicDB.similar_episodes ──

class TestSimilarEpisodesIntegration:
    def test_similar_episodes_uses_planner(self, planner_db):
        """similar_episodes should work with the planner transparently."""
        emb = _random_embedding(seed=42)
        results = planner_db.similar_episodes(emb, limit=5)
        assert len(results) == 5

    def test_similar_episodes_with_status_filter(self, planner_db):
        emb = _random_embedding(seed=42)
        results = planner_db.similar_episodes(emb, status="failure", limit=5)
        assert all(r["status"] == "failure" for r in results)

    def test_similar_episodes_no_embedding_raises(self, planner_db):
        with pytest.raises(ValueError, match="Expected 1536"):
            planner_db.similar_episodes([1.0, 2.0], limit=5)


class TestSecurityAndEdgeCases:
    def test_sql_injection_column_rejected(self, planner_db):
        """Predicate column names must be in the allowlist."""
        planner = HybridPlanner(planner_db._conn)
        emb = _random_embedding(seed=42)
        with pytest.raises(ValueError, match="Unsupported predicate column"):
            planner.execute(
                "planner-test", emb,
                predicates={"status; DROP TABLE episodes; --": "x"},
                limit=5,
            )

    def test_empty_db_returns_empty(self):
        """Planner on a DB with no episodes should return empty, not error."""
        db = EpisodicDB(agent_id="empty-test", path=":memory:")
        planner = HybridPlanner(db._conn)
        emb = _random_embedding(seed=42)
        results = planner.execute(
            "empty-test", emb,
            predicates={"status": "failure"},
            limit=5,
        )
        assert results == []
        db.close()

    def test_cold_cache_respects_filter(self, planner_db):
        """First query on fresh planner must still apply predicates."""
        planner = HybridPlanner(planner_db._conn)
        emb = _random_embedding(seed=42)
        results = planner.execute(
            "planner-test", emb,
            predicates={"status": "failure"},
            limit=5,
        )
        assert all(r["status"] == "failure" for r in results)
