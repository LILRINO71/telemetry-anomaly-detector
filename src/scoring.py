"""Reusable, framework-free scoring helpers.

Pure functions over plain ``dict`` events (no FastAPI, no Streamlit), so the
same logic powers the interactive demo, ad-hoc scripts, and tests without
dragging in a web or UI framework. Every function is deterministic and side
-effect free.

An *event* is one per-tick telemetry record::

    {"tick": int|None, "x": float, "y": float, "z": float,
     "yaw": float, "pitch": float, "block_type": str|None}
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import config
from . import features as features_mod

__all__ = [
    "events_to_frame",
    "feature_vector",
    "zscores",
    "top_contributions",
    "score_session",
    "session_to_events",
    "train_demo_model",
]


def events_to_frame(events: list[dict[str, Any]]) -> pd.DataFrame:
    """Turn a list of per-tick event dicts into a DataFrame for the extractor."""
    return pd.DataFrame(list(events))


def feature_vector(events: list[dict[str, Any]]) -> np.ndarray:
    """Extract the ordered ``N_FEATURES`` vector for one session's events."""
    frame = events_to_frame(events)
    out = features_mod.extract_features(frame)
    return np.array([float(out[name]) for name in config.FEATURE_NAMES], dtype=float)


def zscores(model: Any, vector: np.ndarray) -> np.ndarray:
    """Per-feature z-scores using the model's fitted ``StandardScaler``.

    The scaler's standardization *is* the z-score: ``(x - mean_) / scale_``.
    Falls back to zeros if the model exposes no fitted scaler.
    """
    scaler = getattr(model, "scaler_", None)
    if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        mean = np.asarray(scaler.mean_, dtype=float).reshape(-1)
        scale = np.asarray(scaler.scale_, dtype=float).reshape(-1)
        safe_scale = np.where(scale > 0.0, scale, 1.0)
        return (vector - mean) / safe_scale
    return np.zeros_like(vector)


def top_contributions(vector: np.ndarray, zs: np.ndarray, limit: int = 5) -> list[dict[str, Any]]:
    """Rank features by ``|zscore|`` (descending) and return the top ``limit``."""
    order = np.argsort(-np.abs(zs))
    return [
        {
            "feature": config.FEATURE_NAMES[int(i)],
            "value": float(vector[int(i)]),
            "zscore": float(zs[int(i)]),
        }
        for i in order[:limit]
    ]


def score_session(model: Any, events: list[dict[str, Any]], top_k: int = 5) -> dict[str, Any]:
    """Score one session end-to-end and return a fully explained verdict."""
    vector = feature_vector(events)
    score = float(model.score_samples(vector.reshape(1, -1))[0])
    is_anomaly = bool(int(np.asarray(model.predict(vector.reshape(1, -1))).ravel()[0]))
    zs = zscores(model, vector)
    return {
        "n_events": len(events),
        "anomaly_score": score,
        "is_anomaly": is_anomaly,
        "threshold": float(model.threshold_),
        "features": {name: float(vector[i]) for i, name in enumerate(config.FEATURE_NAMES)},
        "zscores": {name: float(zs[i]) for i, name in enumerate(config.FEATURE_NAMES)},
        "top_features": top_contributions(vector, zs, top_k),
    }


def session_to_events(session: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a generated session (nested ``ticks``) into flat event dicts."""
    return [
        {
            "tick": tick.get("t"),
            "x": tick.get("x"),
            "y": tick.get("y"),
            "z": tick.get("z"),
            "yaw": tick.get("yaw"),
            "pitch": tick.get("pitch"),
            "block_type": tick.get("block_type"),
        }
        for tick in session.get("ticks", [])
    ]


def train_demo_model(
    kind: str = "isolation_forest",
    n_normal: int = 500,
    n_cheater: int = 120,
    seed: int = config.DEFAULT_SEED,
    difficulty: float = 0.88,
) -> Any:
    """Train a detector on freshly generated data (for the demo / examples).

    Fits on normal sessions only, exactly like the production trainer, but
    without the train/test split — the demo just needs a usable scorer. Kept here
    (framework-free) so it can be unit-tested and wrapped in Streamlit's cache.
    """
    from .data import build_labeled_dataset
    from .detectors import make_detector
    from .features import extract_feature_frame

    frame, labels = build_labeled_dataset(
        n_normal=n_normal, n_cheater=n_cheater, seed=seed, difficulty=difficulty
    )
    feats = extract_feature_frame(frame)
    labels = labels.reindex(feats.index)
    normal_rows = feats[labels.to_numpy() == 0].loc[:, config.FEATURE_NAMES].to_numpy(dtype=float)
    return make_detector(kind, random_state=seed).fit(normal_rows)
