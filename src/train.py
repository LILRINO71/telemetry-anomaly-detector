"""Train and evaluate the anomaly model, then persist the model and metrics.

Pipeline
--------
1. Build a labelled feature dataset (normal + cheating sessions) via the data
   generator and feature extractor.
2. Split into train/test with stratification on the label.
3. Fit :class:`~src.model.AnomalyModel` on the *normal* training sessions only.
4. Evaluate on the held-out mixed test set: ROC-AUC, average precision,
   precision/recall/F1 (at the calibrated threshold) and a confusion matrix.
5. Write ``models/model.joblib`` and ``models/metrics.json``.

Run as a module::

    python -m src.train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from . import config
from .model import AnomalyModel


def build_dataset(
    n_normal: int,
    n_cheater: int,
    seed: int,
    difficulty: float = 0.88,
) -> tuple[pd.DataFrame, NDArray[np.int64]]:
    """Return a feature ``DataFrame`` and label vector (1 == cheater, 0 == normal).

    Delegates telemetry synthesis and feature extraction to the sibling modules
    (``src.data`` and ``src.features``), which are contract-bound to the same
    :data:`~src.config.FEATURE_NAMES` schema. The returned frame has exactly the
    contractual feature columns in order.
    """
    from . import data as data_mod
    from . import features as feat_mod

    frame, labels = data_mod.build_labeled_dataset(
        n_normal=n_normal,
        n_cheater=n_cheater,
        seed=seed,
        difficulty=difficulty,
    )
    features_df = feat_mod.extract_feature_frame(frame)
    # Align labels to feature rows by session id (robust to any reordering or
    # dropped short sessions) rather than relying on positional order.
    labels = labels.reindex(features_df.index)
    X = features_df.loc[:, config.FEATURE_NAMES].astype(np.float64)
    y = labels.to_numpy(dtype=np.int64).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Feature/label length mismatch: {X.shape[0]} rows vs {y.shape[0]} labels."
        )
    return X, y


def evaluate(
    model: AnomalyModel,
    X_test: pd.DataFrame,
    y_test: NDArray[np.int64],
) -> dict[str, Any]:
    """Compute ranking and threshold-based metrics on the test set."""
    scores = model.score_samples(X_test)
    y_pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "average_precision": float(average_precision_score(y_test, scores)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "threshold": float(model.threshold_),  # type: ignore[arg-type]
        "n_test": int(len(y_test)),
        "n_test_positive": int(np.sum(y_test == 1)),
        "n_test_negative": int(np.sum(y_test == 0)),
    }


def train(
    n_normal: int = 1500,
    n_cheater: int = 400,
    test_size: float = 0.3,
    seed: int = config.DEFAULT_SEED,
    difficulty: float = 0.88,
    model_path: Path = config.MODEL_PATH,
    metrics_path: Path = config.METRICS_PATH,
) -> dict[str, Any]:
    """Run the full fit + evaluate pipeline and persist artifacts.

    Returns the metrics dictionary that is also written to ``metrics_path``.
    """
    X, y = build_dataset(n_normal=n_normal, n_cheater=n_cheater, seed=seed, difficulty=difficulty)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # One-class training: fit on the normal training sessions only.
    X_train_normal = X_train.loc[y_train == 0]
    if X_train_normal.shape[0] < config.N_FEATURES:
        raise ValueError(
            "Too few normal training sessions to fit the model "
            f"({X_train_normal.shape[0]} < {config.N_FEATURES})."
        )

    model = AnomalyModel(target_fpr=config.TARGET_FPR, random_state=seed)
    model.fit(X_train_normal)

    metrics = evaluate(model, X_test, y_test)
    metrics["model_version"] = config.MODEL_VERSION
    metrics["target_fpr"] = config.TARGET_FPR
    metrics["n_train_normal"] = int(X_train_normal.shape[0])
    metrics["seed"] = int(seed)
    metrics["feature_names"] = list(config.FEATURE_NAMES)

    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    model.save(model_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)

    return metrics


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the telemetry anomaly model.")
    parser.add_argument("--n-normal", type=int, default=1500, help="Normal sessions to synthesise.")
    parser.add_argument(
        "--n-cheater", type=int, default=400, help="Cheating sessions to synthesise."
    )
    parser.add_argument("--test-size", type=float, default=0.3, help="Held-out test fraction.")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--difficulty",
        type=float,
        default=0.88,
        help="Cheater/normal overlap in [0, 1]; higher = harder (default: 0.88).",
    )
    parser.add_argument(
        "--model-path", type=Path, default=config.MODEL_PATH, help="Output model path."
    )
    parser.add_argument(
        "--metrics-path", type=Path, default=config.METRICS_PATH, help="Output metrics path."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: train, evaluate, persist, and print a metrics summary."""
    args = _parse_args(argv)
    metrics = train(
        n_normal=args.n_normal,
        n_cheater=args.n_cheater,
        test_size=args.test_size,
        seed=args.seed,
        difficulty=args.difficulty,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
    )
    cm = metrics["confusion_matrix"]
    print(f"Model written to : {args.model_path}")
    print(f"Metrics written to: {args.metrics_path}")
    print(f"  ROC-AUC        : {metrics['roc_auc']:.4f}")
    print(f"  Avg precision  : {metrics['average_precision']:.4f}")
    print(f"  Precision      : {metrics['precision']:.4f}")
    print(f"  Recall         : {metrics['recall']:.4f}")
    print(f"  F1             : {metrics['f1']:.4f}")
    print(f"  Confusion (tn, fp, fn, tp): ({cm['tn']}, {cm['fp']}, {cm['fn']}, {cm['tp']})")


if __name__ == "__main__":
    main()
