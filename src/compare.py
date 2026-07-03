"""Benchmark the Isolation Forest and autoencoder detectors head-to-head.

Both detectors are trained on the *same* normal training split and evaluated on
the *same* held-out test split, so the comparison is apples-to-apples. Results
are printed as a table, written to ``models/comparison.json``, and rendered as an
overlaid ROC figure at ``reports/model_comparison.png``.

Run as a module::

    python -m src.compare
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from . import config
from .detectors import make_detector
from .train import build_dataset, evaluate

logger = logging.getLogger(__name__)

try:  # pragma: no cover - guarded no-op when matplotlib is unavailable
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:  # noqa: BLE001
    plt = None  # type: ignore[assignment]
    _HAS_MPL = False

MODELS: tuple[str, ...] = ("isolation_forest", "autoencoder")
COMPARISON_PATH = config.MODELS_DIR / "comparison.json"
FIGURE_PATH = config.REPORTS_DIR / "model_comparison.png"

_TABLE_METRICS = [
    ("roc_auc", "ROC-AUC"),
    ("average_precision", "Avg precision"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
]


def run_comparison(
    n_normal: int = 1500,
    n_cheater: int = 400,
    test_size: float = 0.3,
    seed: int = config.DEFAULT_SEED,
    difficulty: float = 0.88,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], tuple[Any, Any]]:
    """Train and evaluate every detector on one shared split.

    Returns ``(results, fitted, (X_test, y_test))`` where ``results`` maps each
    model kind to its metrics dict and ``fitted`` maps it to the trained detector.
    """
    X, y = build_dataset(n_normal=n_normal, n_cheater=n_cheater, seed=seed, difficulty=difficulty)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train_normal = X_train.loc[y_train == 0]

    results: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Any] = {}
    for kind in MODELS:
        model = make_detector(kind, target_fpr=config.TARGET_FPR, random_state=seed)
        model.fit(X_train_normal)
        metrics = evaluate(model, X_test, y_test)
        metrics["model_kind"] = kind
        results[kind] = metrics
        fitted[kind] = model
    return results, fitted, (X_test, y_test)


def plot_roc(
    fitted: dict[str, Any], X_test: Any, y_test: Any, out_path: Path = FIGURE_PATH
) -> Path | None:
    """Render overlaid ROC curves for every fitted detector. Returns the path."""
    if not _HAS_MPL:
        logger.warning("matplotlib unavailable; skipping comparison figure")
        return None

    from sklearn.metrics import roc_auc_score, roc_curve

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    for kind, model in fitted.items():
        scores = model.score_samples(X_test)
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        ax.plot(fpr, tpr, lw=2, label=f"{kind}  (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=1, label="chance")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Detector comparison — ROC (held-out test set)")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


def format_table(results: dict[str, dict[str, Any]]) -> str:
    """Render the metrics side by side as a fixed-width table."""
    kinds = list(results)
    lines = [f"{'metric':<16}" + "".join(f"{k:>20}" for k in kinds)]
    lines.append("-" * (16 + 20 * len(kinds)))
    for key, label in _TABLE_METRICS:
        row = f"{label:<16}" + "".join(f"{results[k][key]:>20.4f}" for k in kinds)
        lines.append(row)
    for k in kinds:
        cm = results[k]["confusion_matrix"]
        results_cm = f"({cm['tn']}, {cm['fp']}, {cm['fn']}, {cm['tp']})"
        results[k]["_cm_str"] = results_cm
    lines.append(f"{'confusion':<16}" + "".join(f"{results[k]['_cm_str']:>20}" for k in kinds))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run the comparison, persist JSON + figure, print a table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Benchmark the anomaly detectors head-to-head.")
    parser.add_argument("--n-normal", type=int, default=1500)
    parser.add_argument("--n-cheater", type=int, default=400)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--difficulty", type=float, default=0.88)
    args = parser.parse_args(argv)

    results, fitted, (X_test, y_test) = run_comparison(
        n_normal=args.n_normal,
        n_cheater=args.n_cheater,
        test_size=args.test_size,
        seed=args.seed,
        difficulty=args.difficulty,
    )

    figure = plot_roc(fitted, X_test, y_test)

    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "difficulty": args.difficulty,
        "n_normal": args.n_normal,
        "n_cheater": args.n_cheater,
        "results": {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in results.items()
        },
    }
    with COMPARISON_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    print(format_table(results))
    print(f"\nComparison JSON : {COMPARISON_PATH}")
    print(f"ROC figure      : {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
