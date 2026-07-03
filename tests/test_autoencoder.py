"""Tests for the autoencoder anomaly detector (``src.detectors.AutoencoderDetector``)."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.detectors import AutoencoderDetector
from tests.conftest import feature_matrix


@pytest.fixture(scope="module")
def ae_normal() -> np.ndarray:
    """Feature matrix of legitimate sessions to train the autoencoder on."""
    return feature_matrix("normal", n=60, base_seed=2000)


@pytest.fixture(scope="module")
def ae_cheater() -> np.ndarray:
    """Feature matrix of cheater sessions (held out from training)."""
    return feature_matrix("cheater", n=30, base_seed=9000)


@pytest.fixture(scope="module")
def ae_model(ae_normal: np.ndarray) -> AutoencoderDetector:
    """Autoencoder fitted on normal sessions only (fewer iters for test speed)."""
    model = AutoencoderDetector(max_iter=1500, random_state=config.DEFAULT_SEED)
    model.fit(ae_normal)
    return model


def test_kind_and_threshold(ae_model: AutoencoderDetector) -> None:
    assert ae_model.kind == "autoencoder"
    assert np.isfinite(float(ae_model.threshold_))


def test_scores_finite_one_per_row(ae_model: AutoencoderDetector, ae_normal: np.ndarray) -> None:
    scores = np.asarray(ae_model.score_samples(ae_normal)).ravel()
    assert scores.shape == (ae_normal.shape[0],)
    assert np.isfinite(scores).all()


def test_predict_is_binary(ae_model: AutoencoderDetector, ae_cheater: np.ndarray) -> None:
    labels = np.asarray(ae_model.predict(ae_cheater)).ravel()
    assert set(np.unique(labels)).issubset({0, 1})


def test_reconstruction_error_higher_for_cheaters(
    ae_model: AutoencoderDetector, ae_normal: np.ndarray, ae_cheater: np.ndarray
) -> None:
    """Cheaters fall outside the learned normal manifold -> larger reconstruction error."""
    normal_scores = np.asarray(ae_model.score_samples(ae_normal)).ravel()
    cheater_scores = np.asarray(ae_model.score_samples(ae_cheater)).ravel()
    assert cheater_scores.mean() > normal_scores.mean()


def test_cheaters_flagged_more_often(
    ae_model: AutoencoderDetector, ae_normal: np.ndarray, ae_cheater: np.ndarray
) -> None:
    normal_rate = np.asarray(ae_model.predict(ae_normal)).ravel().mean()
    cheater_rate = np.asarray(ae_model.predict(ae_cheater)).ravel().mean()
    assert cheater_rate > normal_rate


def test_save_load_roundtrip_preserves_scores(
    ae_model: AutoencoderDetector, tmp_path, ae_normal: np.ndarray, ae_cheater: np.ndarray
) -> None:
    path = tmp_path / "ae.joblib"
    ae_model.save(path)
    reloaded = AutoencoderDetector.load(path)
    assert reloaded.kind == "autoencoder"
    stacked = np.vstack([ae_normal, ae_cheater])
    before = np.asarray(ae_model.score_samples(stacked)).ravel()
    after = np.asarray(reloaded.score_samples(stacked)).ravel()
    np.testing.assert_allclose(before, after, rtol=1e-9, atol=1e-9)
    assert float(reloaded.threshold_) == float(ae_model.threshold_)


def test_training_is_reproducible(ae_normal: np.ndarray) -> None:
    a = AutoencoderDetector(max_iter=800, random_state=7).fit(ae_normal)
    b = AutoencoderDetector(max_iter=800, random_state=7).fit(ae_normal)
    np.testing.assert_allclose(
        np.asarray(a.score_samples(ae_normal)).ravel(),
        np.asarray(b.score_samples(ae_normal)).ravel(),
        rtol=1e-9,
        atol=1e-9,
    )
