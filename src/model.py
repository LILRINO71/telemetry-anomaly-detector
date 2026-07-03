"""Anomaly model: StandardScaler + IsolationForest fitted on normal sessions only.

The model treats cheating sessions as anomalies. It is fitted exclusively on
*normal* feature vectors (unsupervised, one-class style), then a scalar decision
threshold is calibrated against a held-out set of normal sessions so that the
empirical false-positive rate matches :data:`~src.config.TARGET_FPR`.

Anomaly score convention (used everywhere downstream):
    higher score  ==>  more anomalous / more likely to be cheating.

This is ``-IsolationForest.score_samples``. A session is flagged when its anomaly
score is greater than or equal to the calibrated ``threshold_``.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from . import config


@dataclass(frozen=True)
class ModelMetadata:
    """Provenance and calibration facts persisted alongside the fitted model."""

    model_version: str
    sklearn_version: str
    feature_names: list[str]
    n_features: int
    target_fpr: float
    threshold: float
    n_train_normal: int
    trained_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the metadata."""
        return {
            "model_version": self.model_version,
            "sklearn_version": self.sklearn_version,
            "feature_names": list(self.feature_names),
            "n_features": self.n_features,
            "target_fpr": self.target_fpr,
            "threshold": self.threshold,
            "n_train_normal": self.n_train_normal,
            "trained_at": self.trained_at,
        }


class AnomalyModel:
    """One-class anomaly detector wrapping StandardScaler + IsolationForest.

    Fit on normal sessions only; the decision threshold is calibrated so the
    false-positive rate on the training-normal distribution equals the configured
    :data:`~src.config.TARGET_FPR`.
    """

    def __init__(
        self,
        n_estimators: int = config.ISOFOREST_N_ESTIMATORS,
        max_samples: int | float | str = config.ISOFOREST_MAX_SAMPLES,
        contamination: float | str = config.ISOFOREST_CONTAMINATION,
        target_fpr: float = config.TARGET_FPR,
        random_state: int = config.DEFAULT_SEED,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.target_fpr = target_fpr
        self.random_state = random_state

        self.scaler_: StandardScaler | None = None
        self.forest_: IsolationForest | None = None
        self.threshold_: float | None = None
        self.metadata_: ModelMetadata | None = None

    # ------------------------------------------------------------------
    # Fitting / calibration
    # ------------------------------------------------------------------
    def fit(self, X_normal: ArrayLike) -> AnomalyModel:
        """Fit the scaler and forest on normal sessions and calibrate the threshold.

        Parameters
        ----------
        X_normal:
            Feature matrix of shape ``(n_samples, N_FEATURES)`` containing only
            normal (non-cheating) sessions. Column order must match
            :data:`~src.config.FEATURE_NAMES`.
        """
        X = self._as_matrix(X_normal)

        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)

        # n_jobs=1: the training set is small and inference is single-session /
        # low-latency, so a process pool only adds fork + array-pickling overhead
        # per request. Serial is both faster here and reproducible.
        self.forest_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=1,
        ).fit(Xs)

        scores = self._anomaly_scores_scaled(Xs)
        # Calibrate so ~target_fpr of normal sessions land at/above the threshold.
        self.threshold_ = float(np.quantile(scores, 1.0 - self.target_fpr))

        self.metadata_ = ModelMetadata(
            model_version=config.MODEL_VERSION,
            sklearn_version=_sklearn_version(),
            feature_names=list(config.FEATURE_NAMES),
            n_features=config.N_FEATURES,
            target_fpr=self.target_fpr,
            threshold=self.threshold_,
            n_train_normal=int(X.shape[0]),
            trained_at=_dt.datetime.now(_dt.UTC).isoformat(),
        )
        return self

    # ------------------------------------------------------------------
    # Scoring / prediction
    # ------------------------------------------------------------------
    def score_samples(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return the anomaly score for each row (higher == more anomalous)."""
        self._check_fitted()
        Xs = self.scaler_.transform(self._as_matrix(X))  # type: ignore[union-attr]
        return self._anomaly_scores_scaled(Xs)

    def decision_scores(self, X: ArrayLike) -> NDArray[np.float64]:
        """Alias for :meth:`score_samples` (higher == more anomalous)."""
        return self.score_samples(X)

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        """Return binary labels: ``1`` == anomaly/cheater, ``0`` == normal."""
        self._check_fitted()
        scores = self.score_samples(X)
        return (scores >= self.threshold_).astype(np.int64)  # type: ignore[operator]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path = config.MODEL_PATH) -> Path:
        """Serialise the fitted model + metadata to ``path`` via joblib."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scaler": self.scaler_,
            "forest": self.forest_,
            "threshold": self.threshold_,
            "params": {
                "n_estimators": self.n_estimators,
                "max_samples": self.max_samples,
                "contamination": self.contamination,
                "target_fpr": self.target_fpr,
                "random_state": self.random_state,
            },
            "metadata": self.metadata_.to_dict(),  # type: ignore[union-attr]
        }
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path = config.MODEL_PATH) -> AnomalyModel:
        """Load a model previously written by :meth:`save`."""
        path = Path(path)
        payload: dict[str, Any] = joblib.load(path)
        params = payload["params"]
        model = cls(
            n_estimators=params["n_estimators"],
            max_samples=params["max_samples"],
            contamination=params["contamination"],
            target_fpr=params["target_fpr"],
            random_state=params["random_state"],
        )
        model.scaler_ = payload["scaler"]
        model.forest_ = payload["forest"]
        model.threshold_ = float(payload["threshold"])
        meta = payload["metadata"]
        model.metadata_ = ModelMetadata(
            model_version=meta["model_version"],
            sklearn_version=meta["sklearn_version"],
            feature_names=list(meta["feature_names"]),
            n_features=meta["n_features"],
            target_fpr=meta["target_fpr"],
            threshold=meta["threshold"],
            n_train_normal=meta["n_train_normal"],
            trained_at=meta["trained_at"],
        )
        return model

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _anomaly_scores_scaled(self, Xs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Anomaly scores from already-scaled features (higher == more anomalous)."""
        return -self.forest_.score_samples(Xs).astype(np.float64)  # type: ignore[union-attr]

    def _check_fitted(self) -> None:
        if self.scaler_ is None or self.forest_ is None or self.threshold_ is None:
            raise RuntimeError("AnomalyModel is not fitted; call fit() or load() first.")

    @staticmethod
    def _as_matrix(X: ArrayLike) -> NDArray[np.float64]:
        """Coerce input to a validated 2-D float matrix with N_FEATURES columns."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2-D feature matrix, got ndim={arr.ndim}.")
        if arr.shape[1] != config.N_FEATURES:
            raise ValueError(
                f"Expected {config.N_FEATURES} features "
                f"({config.N_FEATURES} columns), got {arr.shape[1]}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Feature matrix contains NaN or infinite values.")
        return arr


def _sklearn_version() -> str:
    """Return the installed scikit-learn version string."""
    import sklearn

    return sklearn.__version__
