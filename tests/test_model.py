"""Tests for the anomaly model (``src.model.AnomalyModel``)."""

from __future__ import annotations

import numpy as np

from src import config
from src.model import AnomalyModel


def _new_model() -> AnomalyModel:
    return AnomalyModel(target_fpr=config.TARGET_FPR, random_state=config.DEFAULT_SEED)


def test_score_samples_finite_and_one_per_row(fitted_model, normal_matrix: np.ndarray) -> None:
    """Scoring returns one finite value per input row."""
    scores = np.asarray(fitted_model.score_samples(normal_matrix)).ravel()
    assert scores.shape == (normal_matrix.shape[0],)
    assert np.isfinite(scores).all()


def test_predict_is_binary(fitted_model, cheater_matrix: np.ndarray) -> None:
    """``predict`` returns 0/1 labels, one per row."""
    labels = np.asarray(fitted_model.predict(cheater_matrix)).ravel()
    assert labels.shape == (cheater_matrix.shape[0],)
    assert set(np.unique(labels)).issubset({0, 1})


def test_threshold_is_finite_float(fitted_model) -> None:
    """The calibrated decision threshold is a finite scalar."""
    assert np.isfinite(float(fitted_model.threshold_))


def test_cheaters_score_higher_on_average(
    fitted_model, normal_matrix: np.ndarray, cheater_matrix: np.ndarray
) -> None:
    """Core contract: cheaters are more anomalous than legitimate miners.

    ``score_samples`` is oriented so higher == more anomalous.
    """
    normal_scores = np.asarray(fitted_model.score_samples(normal_matrix)).ravel()
    cheater_scores = np.asarray(fitted_model.score_samples(cheater_matrix)).ravel()
    assert cheater_scores.mean() > normal_scores.mean()


def test_cheaters_flagged_more_often_than_normals(
    fitted_model, normal_matrix: np.ndarray, cheater_matrix: np.ndarray
) -> None:
    """The 0/1 verdict flags cheaters at a higher rate than normals."""
    normal_rate = np.asarray(fitted_model.predict(normal_matrix)).ravel().mean()
    cheater_rate = np.asarray(fitted_model.predict(cheater_matrix)).ravel().mean()
    assert cheater_rate > normal_rate


def test_save_load_roundtrip_preserves_scores(
    fitted_model, model_file, normal_matrix: np.ndarray, cheater_matrix: np.ndarray
) -> None:
    """A reloaded model scores identically to the in-memory original."""
    reloaded = AnomalyModel.load(model_file)
    stacked = np.vstack([normal_matrix, cheater_matrix])
    before = np.asarray(fitted_model.score_samples(stacked)).ravel()
    after = np.asarray(reloaded.score_samples(stacked)).ravel()
    np.testing.assert_allclose(before, after, rtol=1e-9, atol=1e-9)
    assert float(reloaded.threshold_) == float(fitted_model.threshold_)


def test_saved_model_file_exists_and_nonempty(model_file) -> None:
    """Persistence actually writes a joblib artifact to disk."""
    assert model_file.exists()
    assert model_file.stat().st_size > 0


def test_training_is_reproducible_with_fixed_seed(normal_matrix: np.ndarray) -> None:
    """Two models trained on identical data + seed give identical scores."""
    m1 = _new_model()
    m1.fit(normal_matrix)
    m2 = _new_model()
    m2.fit(normal_matrix)
    s1 = np.asarray(m1.score_samples(normal_matrix)).ravel()
    s2 = np.asarray(m2.score_samples(normal_matrix)).ravel()
    np.testing.assert_allclose(s1, s2, rtol=1e-9, atol=1e-9)


def test_scores_separate_populations(
    fitted_model, normal_matrix: np.ndarray, cheater_matrix: np.ndarray
) -> None:
    """Cheater/normal score means differ by a real margin, not just noise."""
    normal_scores = np.asarray(fitted_model.score_samples(normal_matrix)).ravel()
    cheater_scores = np.asarray(fitted_model.score_samples(cheater_matrix)).ravel()
    margin = cheater_scores.mean() - normal_scores.mean()
    spread = normal_scores.std() + config.EPS
    assert margin > 0.5 * spread
