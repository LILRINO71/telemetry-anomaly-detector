"""Anomaly detectors behind one shared interface.

Two unsupervised, one-class detectors — both trained on *legitimate* sessions
only, both exposing an identical surface so the trainer, API, and evaluation
code are model-agnostic:

* :class:`IsolationForestDetector` — a tree ensemble that isolates anomalies in
  few random splits (fast, strong on tabular data).
* :class:`AutoencoderDetector` — a bottleneck neural network (scikit-learn
  ``MLPRegressor``) trained to *reconstruct* normal behavior; a session that
  reconstructs poorly (high error) is anomalous.

Shared conventions for every detector:

* ``anomaly_score`` is oriented so **higher == more anomalous**.
* the input is standardized with a :class:`~sklearn.preprocessing.StandardScaler`
  fitted on the training-normal data;
* the decision ``threshold_`` is calibrated on the training-normal score
  distribution so the empirical false-positive rate matches
  :data:`~src.config.TARGET_FPR`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from . import config

__all__ = [
    "BaseAnomalyDetector",
    "IsolationForestDetector",
    "AutoencoderDetector",
    "make_detector",
    "load_detector",
    "DETECTOR_KINDS",
]

_REGISTRY: dict[str, type[BaseAnomalyDetector]] = {}


def _register(cls: type[BaseAnomalyDetector]) -> type[BaseAnomalyDetector]:
    _REGISTRY[cls.kind] = cls
    return cls


def _sklearn_version() -> str:
    import sklearn

    return sklearn.__version__


class BaseAnomalyDetector:
    """Common one-class anomaly-detector machinery (scale → fit → calibrate).

    Subclasses implement three hooks: :meth:`_build_estimator` (the underlying
    sklearn model), :meth:`_fit_args` (positional args passed to ``estimator.fit``)
    and :meth:`_raw_scores` (map scaled features → anomaly scores, higher worse).
    Everything else — scaling, threshold calibration, prediction, persistence — is
    shared so both detectors stay behaviorally consistent.
    """

    kind = "base"

    def __init__(
        self,
        target_fpr: float = config.TARGET_FPR,
        random_state: int = config.DEFAULT_SEED,
    ) -> None:
        self.target_fpr = float(target_fpr)
        self.random_state = int(random_state)

        self.scaler_: StandardScaler | None = None
        self.estimator_: Any = None
        self.threshold_: float | None = None
        self.metadata_: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    def _build_estimator(self) -> Any:
        raise NotImplementedError

    def _fit_args(self, Xs: NDArray[np.float64]) -> tuple[Any, ...]:
        """Positional args for ``estimator.fit`` (override for autoencoders)."""
        return (Xs,)

    def _raw_scores(self, Xs: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError

    def _extra_params(self) -> dict[str, Any]:
        """Hyper-parameters (beyond target_fpr/random_state) needed to rebuild."""
        return {}

    # ------------------------------------------------------------------
    # Fit / calibrate
    # ------------------------------------------------------------------
    def fit(self, X_normal: ArrayLike) -> BaseAnomalyDetector:
        """Fit on normal sessions only and calibrate the decision threshold."""
        X = self._as_matrix(X_normal)

        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)

        self.estimator_ = self._build_estimator().fit(*self._fit_args(Xs))

        scores = self._raw_scores(Xs)
        # ~target_fpr of normal sessions land at/above the threshold.
        self.threshold_ = float(np.quantile(scores, 1.0 - self.target_fpr))

        self.metadata_ = {
            "kind": self.kind,
            "model_version": config.MODEL_VERSION,
            "sklearn_version": _sklearn_version(),
            "n_features": config.N_FEATURES,
            "feature_names": list(config.FEATURE_NAMES),
            "target_fpr": self.target_fpr,
            "threshold": self.threshold_,
            "n_train_normal": int(X.shape[0]),
        }
        return self

    # ------------------------------------------------------------------
    # Score / predict
    # ------------------------------------------------------------------
    def score_samples(self, X: ArrayLike) -> NDArray[np.float64]:
        """Anomaly score per row (higher == more anomalous)."""
        self._check_fitted()
        Xs = self.scaler_.transform(self._as_matrix(X))  # type: ignore[union-attr]
        return self._raw_scores(Xs)

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        """Binary labels: ``1`` == anomaly/cheater, ``0`` == normal."""
        self._check_fitted()
        return (self.score_samples(X) >= self.threshold_).astype(np.int64)  # type: ignore[operator]

    def score_one(self, vector: ArrayLike) -> dict[str, Any]:
        """Score a single feature vector; returns score/flag/threshold."""
        score = float(self.score_samples(np.asarray(vector, dtype=np.float64).reshape(1, -1))[0])
        return {
            "anomaly_score": score,
            "is_anomaly": bool(score >= self.threshold_),  # type: ignore[operator]
            "threshold": float(self.threshold_),  # type: ignore[arg-type]
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path = config.MODEL_PATH) -> Path:
        """Serialize the fitted detector (kind-tagged) to ``path`` via joblib."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": self.kind,
            "scaler": self.scaler_,
            "estimator": self.estimator_,
            "threshold": self.threshold_,
            "target_fpr": self.target_fpr,
            "random_state": self.random_state,
            "extra_params": self._extra_params(),
            "metadata": self.metadata_,
        }
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path = config.MODEL_PATH) -> BaseAnomalyDetector:
        """Load a detector, dispatching to the right class by its stored ``kind``."""
        payload: dict[str, Any] = joblib.load(Path(path))
        kind = payload.get("kind", IsolationForestDetector.kind)
        target = _REGISTRY.get(
            kind, cls if cls is not BaseAnomalyDetector else IsolationForestDetector
        )
        obj = target(
            target_fpr=payload["target_fpr"],
            random_state=payload["random_state"],
            **payload.get("extra_params", {}),
        )
        obj.scaler_ = payload["scaler"]
        obj.estimator_ = payload["estimator"]
        obj.threshold_ = float(payload["threshold"])
        obj.metadata_ = payload.get("metadata")
        return obj

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self.scaler_ is None or self.estimator_ is None or self.threshold_ is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() or load() first.")

    @staticmethod
    def _as_matrix(X: ArrayLike) -> NDArray[np.float64]:
        """Coerce input to a validated 2-D float matrix with N_FEATURES columns."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2-D feature matrix, got ndim={arr.ndim}.")
        if arr.shape[1] != config.N_FEATURES:
            raise ValueError(f"Expected {config.N_FEATURES} feature columns, got {arr.shape[1]}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Feature matrix contains NaN or infinite values.")
        return arr


@_register
class IsolationForestDetector(BaseAnomalyDetector):
    """One-class detector wrapping :class:`~sklearn.ensemble.IsolationForest`.

    Anomaly score is ``-IsolationForest.score_samples`` (higher == more anomalous).
    """

    kind = "isolation_forest"

    def __init__(
        self,
        n_estimators: int = config.ISOFOREST_N_ESTIMATORS,
        max_samples: int | float | str = config.ISOFOREST_MAX_SAMPLES,
        contamination: float | str = config.ISOFOREST_CONTAMINATION,
        target_fpr: float = config.TARGET_FPR,
        random_state: int = config.DEFAULT_SEED,
    ) -> None:
        super().__init__(target_fpr=target_fpr, random_state=random_state)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination

    def _build_estimator(self) -> IsolationForest:
        # n_jobs=1: small training set + single-session inference, so a process
        # pool only adds overhead and array-pickling churn per request.
        return IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=1,
        )

    def _raw_scores(self, Xs: NDArray[np.float64]) -> NDArray[np.float64]:
        return -self.estimator_.score_samples(Xs).astype(np.float64)

    def _extra_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
        }


@_register
class AutoencoderDetector(BaseAnomalyDetector):
    """One-class detector using a bottleneck MLP autoencoder.

    A :class:`~sklearn.neural_network.MLPRegressor` is trained to reconstruct its
    (standardized) input through a narrow hidden "code" layer. Trained on normal
    play only, it reconstructs legitimate sessions well; cheaters fall outside the
    learned manifold and reconstruct poorly, so the per-row **mean squared
    reconstruction error** is the anomaly score (higher == more anomalous).

    Default architecture for 15 inputs: ``15 → 12 → 6 → 12 → 15`` (the width-6
    layer is the compression bottleneck / latent code).
    """

    kind = "autoencoder"

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (12, 6, 12),
        max_iter: int = 2000,
        alpha: float = 1e-4,
        target_fpr: float = config.TARGET_FPR,
        random_state: int = config.DEFAULT_SEED,
    ) -> None:
        super().__init__(target_fpr=target_fpr, random_state=random_state)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.max_iter = int(max_iter)
        self.alpha = float(alpha)

    def _build_estimator(self) -> MLPRegressor:
        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    def _fit_args(self, Xs: NDArray[np.float64]) -> tuple[Any, ...]:
        # Autoencoder: reconstruct the input, so target == input.
        return (Xs, Xs)

    def _raw_scores(self, Xs: NDArray[np.float64]) -> NDArray[np.float64]:
        reconstructed = np.asarray(self.estimator_.predict(Xs), dtype=np.float64)
        if reconstructed.ndim == 1:
            reconstructed = reconstructed.reshape(-1, 1)
        return np.mean((Xs - reconstructed) ** 2, axis=1)

    def _extra_params(self) -> dict[str, Any]:
        return {
            "hidden_layer_sizes": tuple(self.hidden_layer_sizes),
            "max_iter": self.max_iter,
            "alpha": self.alpha,
        }


DETECTOR_KINDS: tuple[str, ...] = tuple(_REGISTRY.keys())


def make_detector(
    kind: str = IsolationForestDetector.kind,
    *,
    target_fpr: float = config.TARGET_FPR,
    random_state: int = config.DEFAULT_SEED,
    **extra_params: Any,
) -> BaseAnomalyDetector:
    """Factory: build a fresh detector by ``kind`` (see :data:`DETECTOR_KINDS`)."""
    if kind not in _REGISTRY:
        raise ValueError(f"unknown detector kind {kind!r}; choose from {DETECTOR_KINDS}")
    return _REGISTRY[kind](target_fpr=target_fpr, random_state=random_state, **extra_params)


def load_detector(path: str | Path = config.MODEL_PATH) -> BaseAnomalyDetector:
    """Load any saved detector, dispatching on its stored ``kind``."""
    return BaseAnomalyDetector.load(path)
