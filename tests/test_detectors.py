"""Tests for the shared detector interface, factory, and kind-dispatched loading."""

from __future__ import annotations

import numpy as np

from src import config
from src.detectors import (
    DETECTOR_KINDS,
    AutoencoderDetector,
    BaseAnomalyDetector,
    IsolationForestDetector,
    load_detector,
    make_detector,
)
from src.model import AnomalyModel
from tests.conftest import feature_matrix


def test_registry_lists_both_kinds() -> None:
    assert set(DETECTOR_KINDS) == {"isolation_forest", "autoencoder"}


def test_anomaly_model_alias_is_isolation_forest() -> None:
    assert AnomalyModel is IsolationForestDetector


def test_make_detector_builds_requested_kind() -> None:
    assert make_detector("isolation_forest").kind == "isolation_forest"
    assert make_detector("autoencoder").kind == "autoencoder"


def test_make_detector_rejects_unknown_kind() -> None:
    import pytest

    with pytest.raises(ValueError):
        make_detector("random_forest")


def test_both_detectors_share_the_interface() -> None:
    for kind in DETECTOR_KINDS:
        det = make_detector(kind)
        assert isinstance(det, BaseAnomalyDetector)
        for attr in ("fit", "score_samples", "predict", "score_one", "save", "load"):
            assert hasattr(det, attr)


def test_load_detector_dispatches_on_saved_kind(tmp_path) -> None:
    """A saved model reloads as its own class regardless of the caller."""
    normal = feature_matrix("normal", n=40, base_seed=3000)

    iso = IsolationForestDetector(n_estimators=50, random_state=config.DEFAULT_SEED).fit(normal)
    ae = AutoencoderDetector(max_iter=400, random_state=config.DEFAULT_SEED).fit(normal)

    iso_path = tmp_path / "iso.joblib"
    ae_path = tmp_path / "ae.joblib"
    iso.save(iso_path)
    ae.save(ae_path)

    assert isinstance(load_detector(iso_path), IsolationForestDetector)
    assert isinstance(load_detector(ae_path), AutoencoderDetector)
    # The generic loader recovers the concrete kind even via the base class.
    assert BaseAnomalyDetector.load(ae_path).kind == "autoencoder"


def test_score_one_shape(tmp_path) -> None:
    normal = feature_matrix("normal", n=40, base_seed=4000)
    det = make_detector("isolation_forest", random_state=config.DEFAULT_SEED).fit(normal)
    out = det.score_one(np.zeros(config.N_FEATURES))
    assert set(out) == {"anomaly_score", "is_anomaly", "threshold"}
    assert isinstance(out["is_anomaly"], bool)
