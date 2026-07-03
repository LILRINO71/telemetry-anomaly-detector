"""Offline evaluation and report-figure generation for the anomaly detector.

Loads a trained :class:`~src.model.AnomalyModel`, builds (or loads) a labelled
feature matrix, computes per-session anomaly scores, and renders three families
of diagnostic figures into :data:`~src.config.REPORTS_DIR`:

* per-feature class-conditional distributions (cheater vs. legitimate),
* a ROC curve with the calibrated operating point marked,
* a histogram of anomaly scores split by class, with the decision threshold.

Anomaly-score convention matches :mod:`src.model`: higher == more anomalous.
Scores come straight from :meth:`AnomalyModel.score_samples` (already sign-
flipped) and the decision boundary is the model's calibrated ``threshold_``.

Rendering degrades to a guarded no-op when matplotlib is not importable, so the
module can still be imported (and its metric helpers used) in a headless or
minimal environment. A small argparse CLI wires the pieces together.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config, generate_data
from . import features as feature_mod
from .model import AnomalyModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional matplotlib import  --  everything that draws is guarded on this.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - the guarded no-op branch is exercised in tests
    import matplotlib

    matplotlib.use("Agg")  # headless backend; must precede the pyplot import
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:  # noqa: BLE001 - any import/backend failure disables plotting
    plt = None  # type: ignore[assignment]
    _HAS_MPL = False


LABEL_COLUMN = "label"
CHEATER_LABEL = "cheater"
NORMAL_LABEL = "normal"

_LEGIT_COLOR = "#4c72b0"
_CHEAT_COLOR = "#c44e52"
_FIGSIZE = (10.0, 6.0)
_DPI = 120


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def _labels_to_int(values: Sequence[Any]) -> np.ndarray:
    """Map a label column to ``1`` (cheater) / ``0`` (normal) integers.

    Accepts the string labels emitted by the generator (``"cheater"`` /
    ``"normal"``) as well as already-numeric labels.
    """
    out = np.empty(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        if isinstance(v, str):
            out[i] = 1 if v.strip().lower() == CHEATER_LABEL else 0
        else:
            out[i] = int(v != 0)
    return out


def dataset_from_sessions(
    sessions_path: Path | str,
    min_events: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a ``(features, labels)`` pair from a JSONL sessions file.

    Reads the raw per-session records written by :mod:`src.generate_data`,
    flattens each session's ``ticks`` array into a per-tick frame, extracts the
    contractual feature vector via :func:`src.features.features_for_sessions`,
    and derives an integer label (1 == cheater) from each session's ``label``.
    """
    sessions_path = Path(sessions_path)
    if not sessions_path.exists():
        raise FileNotFoundError(f"sessions file not found: {sessions_path}")

    rows: list[dict[str, Any]] = []
    label_by_session: dict[Any, str] = {}
    for session in generate_data.read_jsonl(sessions_path):
        session_id = session.get("session_id")
        label_by_session[session_id] = session.get("label", NORMAL_LABEL)
        for tick in session.get("ticks", []):
            rows.append(
                {
                    "session_id": session_id,
                    "tick": tick.get("t"),
                    "x": tick.get("x"),
                    "y": tick.get("y"),
                    "z": tick.get("z"),
                    "yaw": tick.get("yaw"),
                    "pitch": tick.get("pitch"),
                    "broke_block": tick.get("dig", False),
                    "block_type": tick.get("block_type"),
                }
            )

    if not rows:
        raise ValueError(f"no per-tick rows parsed from {sessions_path}")

    tick_frame = pd.DataFrame(rows)
    feature_frame = feature_mod.features_for_sessions(
        tick_frame, session_col="session_id", min_events=min_events
    )
    labels = _labels_to_int(
        [label_by_session.get(sid, NORMAL_LABEL) for sid in feature_frame.index]
    )
    features = feature_frame.reset_index(drop=True)
    return features, labels


def dataset_from_table(dataset_path: Path | str) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Load a pre-extracted feature table (``.csv`` / ``.parquet`` / ``.pq``).

    The table must contain every column in :data:`~src.config.FEATURE_NAMES`.
    A ``label`` column, if present, is mapped to integer labels; otherwise the
    returned labels are ``None``.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(dataset_path)
    elif suffix == ".csv":
        frame = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"unsupported dataset format: {dataset_path.suffix!r}")

    missing = [c for c in config.FEATURE_NAMES if c not in frame.columns]
    if missing:
        raise ValueError(f"dataset is missing feature columns: {missing}")

    labels: np.ndarray | None = None
    if LABEL_COLUMN in frame.columns:
        labels = _labels_to_int(list(frame[LABEL_COLUMN]))

    features = frame.loc[:, config.FEATURE_NAMES].astype(np.float64).reset_index(drop=True)
    return features, labels


def build_dataset(
    dataset_path: Path | str,
    min_events: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Dispatch to the JSONL or tabular loader based on file extension."""
    dataset_path = Path(dataset_path)
    if dataset_path.suffix.lower() in {".jsonl", ".json"}:
        return dataset_from_sessions(dataset_path, min_events=min_events)
    return dataset_from_table(dataset_path)


# ---------------------------------------------------------------------------
# Model / scoring
# ---------------------------------------------------------------------------
def load_model(model_path: Path | str = config.MODEL_PATH) -> AnomalyModel:
    """Load the trained :class:`~src.model.AnomalyModel` from ``model_path``."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model artifact not found: {model_path}")
    return AnomalyModel.load(model_path)


def load_metrics(metrics_path: Path | str = config.METRICS_PATH) -> dict[str, Any]:
    """Load the training metrics JSON, or an empty dict when it is absent."""
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        return {}
    with open(metrics_path, encoding="utf-8") as fh:
        return json.load(fh)


def anomaly_scores(model: AnomalyModel, features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Return per-row anomaly scores (higher == more anomalous)."""
    X = features.to_numpy() if isinstance(features, pd.DataFrame) else np.asarray(features)
    return np.asarray(model.score_samples(X), dtype=np.float64).ravel()


# ---------------------------------------------------------------------------
# Metric helpers  (import-safe: no matplotlib dependency)
# ---------------------------------------------------------------------------
def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute ranking and threshold-based metrics on ``(labels, scores)``.

    Uses the model's calibrated ``threshold`` for the confusion matrix. ROC-AUC
    and average precision are only defined when both classes are present; they
    are reported as ``nan`` otherwise. The result is JSON-serialisable.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    result: dict[str, Any] = {
        "n_samples": int(labels.size),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "threshold": float(threshold),
    }

    if n_pos > 0 and n_neg > 0:
        result["roc_auc"] = float(roc_auc_score(labels, scores))
        result["average_precision"] = float(average_precision_score(labels, scores))
    else:
        result["roc_auc"] = float("nan")
        result["average_precision"] = float("nan")

    predicted = (scores >= threshold).astype(int)
    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    result["confusion_matrix"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    result["precision"] = float(tp / (tp + fp)) if (tp + fp) else 0.0
    result["recall"] = float(tp / (tp + fn)) if (tp + fn) else 0.0
    denom_f1 = 2 * tp + fp + fn
    result["f1"] = float(2 * tp / denom_f1) if denom_f1 else 0.0
    result["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return result


# ---------------------------------------------------------------------------
# Figures  (guarded on matplotlib availability)
# ---------------------------------------------------------------------------
def _ensure_reports_dir(reports_dir: Path | str) -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def plot_feature_distributions(
    features: pd.DataFrame,
    labels: np.ndarray | None,
    reports_dir: Path | str = config.REPORTS_DIR,
    filename: str = "feature_distributions.png",
) -> Path | None:
    """Grid of per-feature histograms, split by class when labels are given.

    Returns the written path, or ``None`` when matplotlib is unavailable.
    """
    if not _HAS_MPL:
        logger.warning("matplotlib unavailable; skipping feature-distribution figure")
        return None

    reports_dir = _ensure_reports_dir(reports_dir)
    names = [c for c in config.FEATURE_NAMES if c in features.columns]
    n = len(names)
    ncols = 3
    nrows = max(1, int(np.ceil(n / ncols)))
    labels_arr = None if labels is None else np.asarray(labels).astype(int).ravel()

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.0), squeeze=False)
    flat = axes.ravel()

    for idx, name in enumerate(names):
        ax = flat[idx]
        col = features[name].to_numpy(dtype=np.float64)
        col = col[np.isfinite(col)]
        lo = float(np.min(col)) if col.size else 0.0
        hi = float(np.max(col)) if col.size else 1.0
        bins = np.linspace(lo, hi, 41) if hi > lo else 40

        if labels_arr is None or col.size == 0:
            ax.hist(col, bins=bins, color=_LEGIT_COLOR, alpha=0.85)
        else:
            legit = features.loc[labels_arr == 0, name].to_numpy(dtype=np.float64)
            cheat = features.loc[labels_arr == 1, name].to_numpy(dtype=np.float64)
            legit = legit[np.isfinite(legit)]
            cheat = cheat[np.isfinite(cheat)]
            ax.hist(legit, bins=bins, color=_LEGIT_COLOR, alpha=0.6, label="legit")
            ax.hist(cheat, bins=bins, color=_CHEAT_COLOR, alpha=0.6, label="cheater")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)

    for extra in range(n, nrows * ncols):
        flat[extra].axis("off")

    handles, leg_labels = flat[0].get_legend_handles_labels()
    if leg_labels:
        fig.legend(handles, leg_labels, loc="upper right", fontsize=9)
    fig.suptitle("Feature distributions by class", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    out_path = reports_dir / filename
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


def plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    reports_dir: Path | str = config.REPORTS_DIR,
    filename: str = "roc_curve.png",
    threshold: float | None = None,
) -> Path | None:
    """ROC curve annotated with the operating point at the model threshold."""
    if not _HAS_MPL:
        logger.warning("matplotlib unavailable; skipping ROC figure")
        return None

    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if (labels == 1).sum() == 0 or (labels == 0).sum() == 0:
        logger.warning("ROC undefined: both classes required; skipping figure")
        return None

    from sklearn.metrics import roc_auc_score, roc_curve

    reports_dir = _ensure_reports_dir(reports_dir)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = float(roc_auc_score(labels, scores))

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(fpr, tpr, color=_LEGIT_COLOR, lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#888888", lw=1, ls="--", label="chance")

    if threshold is not None and np.isfinite(threshold):
        # roc_curve thresholds are decreasing; find the operating point whose
        # threshold is closest to the model's calibrated boundary.
        op_idx = int(np.argmin(np.abs(thresholds - threshold)))
        ax.scatter(
            [fpr[op_idx]],
            [tpr[op_idx]],
            color=_CHEAT_COLOR,
            zorder=5,
            label=(f"operating point (FPR = {fpr[op_idx]:.3f}, TPR = {tpr[op_idx]:.3f})"),
        )

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Receiver operating characteristic")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out_path = reports_dir / filename
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


def plot_score_histogram(
    scores: np.ndarray,
    labels: np.ndarray | None = None,
    reports_dir: Path | str = config.REPORTS_DIR,
    filename: str = "score_histogram.png",
    threshold: float | None = None,
) -> Path | None:
    """Histogram of anomaly scores, split by class with the decision threshold."""
    if not _HAS_MPL:
        logger.warning("matplotlib unavailable; skipping score-histogram figure")
        return None

    reports_dir = _ensure_reports_dir(reports_dir)
    scores = np.asarray(scores, dtype=np.float64).ravel()
    finite = scores[np.isfinite(scores)]
    lo = float(np.min(finite)) if finite.size else 0.0
    hi = float(np.max(finite)) if finite.size else 1.0
    bins = np.linspace(lo, hi, 61) if hi > lo else 60

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if labels is None:
        ax.hist(scores, bins=bins, color=_LEGIT_COLOR, alpha=0.85)
    else:
        labels_arr = np.asarray(labels).astype(int).ravel()
        ax.hist(
            scores[labels_arr == 0],
            bins=bins,
            color=_LEGIT_COLOR,
            alpha=0.6,
            label="legit",
        )
        ax.hist(
            scores[labels_arr == 1],
            bins=bins,
            color=_CHEAT_COLOR,
            alpha=0.6,
            label="cheater",
        )
    if threshold is not None and np.isfinite(threshold):
        ax.axvline(
            threshold,
            color="#000000",
            lw=1.5,
            ls="--",
            label=f"threshold = {threshold:.4f}",
        )
    if labels is not None or (threshold is not None and np.isfinite(threshold)):
        ax.legend(loc="upper right", fontsize=9)

    ax.set_xlabel("Anomaly score (higher = more anomalous)")
    ax.set_ylabel("Count")
    ax.set_title("Anomaly-score distribution")
    fig.tight_layout()

    out_path = reports_dir / filename
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def evaluate(
    dataset_path: Path | str,
    model_path: Path | str = config.MODEL_PATH,
    reports_dir: Path | str = config.REPORTS_DIR,
    min_events: int | None = None,
) -> dict[str, Any]:
    """Full evaluation: load model + data, score, plot, and return results.

    The returned dict always contains ``metrics`` and ``figures`` keys. Figure
    values are ``None`` when matplotlib is unavailable; ``metrics`` is empty when
    the dataset carries no labels.
    """
    model = load_model(model_path)
    features, labels = build_dataset(dataset_path, min_events=min_events)
    scores = anomaly_scores(model, features)
    threshold = float(model.threshold_) if model.threshold_ is not None else None

    metrics: dict[str, Any] = {}
    if labels is not None and threshold is not None:
        metrics = compute_metrics(labels, scores, threshold)

    figures = {
        "feature_distributions": plot_feature_distributions(features, labels, reports_dir),
        "roc_curve": (
            plot_roc_curve(labels, scores, reports_dir, threshold=threshold)
            if labels is not None
            else None
        ),
        "score_histogram": plot_score_histogram(scores, labels, reports_dir, threshold=threshold),
    }
    return {"metrics": metrics, "figures": figures}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Evaluate the telemetry anomaly detector and render report figures.")
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help=(
            "Path to an evaluation dataset: a raw sessions .jsonl file, or a "
            "pre-extracted feature table (.csv / .parquet)."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=config.MODEL_PATH,
        help="Path to the trained model artifact (joblib).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=config.REPORTS_DIR,
        help="Directory to write report figures into.",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=None,
        help=(
            "Minimum ticks required to score a session when reading a .jsonl "
            "dataset (defaults to config.MIN_EVENTS)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    result = evaluate(
        dataset_path=args.dataset,
        model_path=args.model,
        reports_dir=args.reports_dir,
        min_events=args.min_events,
    )
    metrics = result["metrics"]
    if metrics:
        logger.info("metrics: %s", json.dumps(metrics, indent=2, sort_keys=True))
    for name, path in result["figures"].items():
        logger.info("figure %-22s -> %s", name, path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
