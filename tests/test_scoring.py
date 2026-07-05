"""Tests for the reusable scoring helpers (``src.scoring``)."""

from __future__ import annotations

import numpy as np
import pytest

from src import config, scoring
from src import generate_data as gen


def _events(label: str, seed: int) -> list[dict]:
    session = gen.generate_session(f"{label}-{seed}", label, 0.0, np.random.default_rng(seed), seed)
    return scoring.session_to_events(session)


def test_session_to_events_shape() -> None:
    events = _events("normal", 1)
    assert len(events) >= config.MIN_EVENTS
    assert set(events[0]) == {"tick", "x", "y", "z", "yaw", "pitch", "block_type"}


def test_feature_vector_length_and_finite() -> None:
    vec = scoring.feature_vector(_events("cheater", 2))
    assert vec.shape == (config.N_FEATURES,)
    assert np.isfinite(vec).all()


def test_top_contributions_sorted_by_abs_zscore() -> None:
    vector = np.arange(config.N_FEATURES, dtype=float)
    zs = np.linspace(-3, 3, config.N_FEATURES)
    top = scoring.top_contributions(vector, zs, limit=5)
    mags = [abs(c["zscore"]) for c in top]
    assert mags == sorted(mags, reverse=True)
    assert all(c["feature"] in config.FEATURE_NAMES for c in top)


@pytest.fixture(scope="module")
def demo_model():
    return scoring.train_demo_model(kind="isolation_forest", n_normal=120, n_cheater=30, seed=7)


def test_train_demo_model_is_fitted(demo_model) -> None:
    assert demo_model.threshold_ is not None
    assert demo_model.scaler_ is not None
    assert demo_model.kind == "isolation_forest"


def test_score_session_structure(demo_model) -> None:
    result = scoring.score_session(demo_model, _events("normal", 100))
    assert set(result) >= {
        "n_events",
        "anomaly_score",
        "is_anomaly",
        "threshold",
        "features",
        "top_features",
    }
    assert isinstance(result["is_anomaly"], bool)
    assert set(result["features"]) == set(config.FEATURE_NAMES)


def test_cheater_scores_higher_than_normal(demo_model) -> None:
    normal = scoring.score_session(demo_model, _events("normal", 100))
    cheater = scoring.score_session(demo_model, _events("cheater", 500))
    assert cheater["anomaly_score"] > normal["anomaly_score"]
