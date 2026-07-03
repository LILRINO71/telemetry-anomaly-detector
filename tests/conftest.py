"""Shared pytest fixtures and helpers.

The generator, feature extractor, model, and API are authored in parallel
against the same spec. These helpers build the tiny, deterministic datasets the
tests need directly from the concrete public contract:

* ``src.generate_data`` -- session dicts with a ``ticks`` array
  (keys ``t, x, y, z, yaw, pitch, dig, block_type``).
* ``src.features.extract_features`` -- per-session extractor over a per-tick
  DataFrame (columns ``tick, x, y, z, yaw, pitch, block_type``), returning a
  ``dict[str, float]`` keyed by ``config.FEATURE_NAMES`` (a Series / bare array
  is also tolerated).
* ``src.model.AnomalyModel`` -- one-class model with ``fit`` / ``score_samples``
  (higher == more anomalous) / ``predict`` (1 == cheater) / ``threshold_`` /
  ``save`` / ``load``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, generate_data


# ---------------------------------------------------------------------------
# Session synthesis (concrete generator contract)
# ---------------------------------------------------------------------------
def make_session_dict(label: str, seed: int, difficulty: float = 0.0) -> dict:
    """Generate one raw session dict for the given ``label`` and ``seed``.

    ``difficulty`` defaults to ``0.0`` (maximally separable populations) so the
    behavioural signal the tests rely on is unambiguous and fast to check.
    """
    rng = np.random.default_rng(seed)
    return generate_data.generate_session(
        session_id=f"{label}-{seed}",
        label=label,
        difficulty=difficulty,
        rng=rng,
        seed=seed,
    )


def session_to_frame(session: dict) -> pd.DataFrame:
    """Flatten a session dict's ``ticks`` array into the per-tick DataFrame.

    Column names match ``api.main.TelemetryEvent`` / the feature extractor's
    contract: the generator's ``t`` becomes ``tick``; ``block_type`` and the
    position/aim columns pass through unchanged.
    """
    rows = [
        {
            "tick": tick["t"],
            "x": tick["x"],
            "y": tick["y"],
            "z": tick["z"],
            "yaw": tick["yaw"],
            "pitch": tick["pitch"],
            "block_type": tick["block_type"],
        }
        for tick in session["ticks"]
    ]
    return pd.DataFrame(rows)


def make_session_frame(label: str, seed: int, difficulty: float = 0.0) -> pd.DataFrame:
    """Convenience: raw session dict -> per-tick DataFrame in one call."""
    return session_to_frame(make_session_dict(label, seed, difficulty))


# ---------------------------------------------------------------------------
# Feature extraction (normalise the extractor's return flavour)
# ---------------------------------------------------------------------------
def extract_vector(frame: pd.DataFrame) -> np.ndarray:
    """Return the ordered N_FEATURES float vector for one per-tick frame."""
    from src import features

    out = features.extract_features(frame)
    if isinstance(out, dict):
        return np.array([float(out[name]) for name in config.FEATURE_NAMES], dtype=float)
    if isinstance(out, pd.Series):
        if set(config.FEATURE_NAMES).issubset(set(out.index)):
            return out.reindex(config.FEATURE_NAMES).to_numpy(dtype=float)
        return out.to_numpy(dtype=float)
    return np.asarray(out, dtype=float).ravel()


def feature_matrix(label: str, n: int, base_seed: int) -> np.ndarray:
    """Stack feature vectors for ``n`` sessions of one label."""
    return np.vstack([extract_vector(make_session_frame(label, base_seed + i)) for i in range(n)])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def normal_matrix() -> np.ndarray:
    """Feature matrix for a handful of legitimate-miner sessions."""
    return feature_matrix("normal", n=20, base_seed=1000)


@pytest.fixture(scope="session")
def cheater_matrix() -> np.ndarray:
    """Feature matrix for a handful of cheater sessions."""
    return feature_matrix("cheater", n=20, base_seed=5000)


@pytest.fixture(scope="session")
def fitted_model(normal_matrix: np.ndarray):
    """An ``AnomalyModel`` fitted on normal sessions only (one-class)."""
    from src.model import AnomalyModel

    model = AnomalyModel(target_fpr=config.TARGET_FPR, random_state=config.DEFAULT_SEED)
    model.fit(normal_matrix)
    return model


@pytest.fixture()
def model_file(tmp_path, fitted_model):
    """Persist the fitted model to a temp joblib file and return its path."""
    path = tmp_path / "model.joblib"
    fitted_model.save(path)
    return path


@pytest.fixture()
def normal_session_events() -> list[dict]:
    """A normal session as a list of API-shaped event dicts (>= MIN_EVENTS)."""
    return _session_event_dicts("normal", seed=1000)


@pytest.fixture()
def cheater_session_events() -> list[dict]:
    """A cheater session as a list of API-shaped event dicts (>= MIN_EVENTS)."""
    return _session_event_dicts("cheater", seed=5000)


def _session_event_dicts(label: str, seed: int) -> list[dict]:
    """Build API ``events`` payload rows for one session."""
    session = make_session_dict(label, seed)
    return [
        {
            "tick": tick["t"],
            "x": tick["x"],
            "y": tick["y"],
            "z": tick["z"],
            "yaw": tick["yaw"],
            "pitch": tick["pitch"],
            "block_type": tick["block_type"],
        }
        for tick in session["ticks"]
    ]
