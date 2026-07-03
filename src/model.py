"""Backward-compatible entry point for the anomaly detectors.

The concrete implementations now live in :mod:`src.detectors`, which defines a
shared :class:`~src.detectors.BaseAnomalyDetector` interface plus two models:
the Isolation Forest and the autoencoder.

``AnomalyModel`` remains an alias for
:class:`~src.detectors.IsolationForestDetector` (the default / production model)
so existing imports and saved artifacts keep working unchanged.
"""

from __future__ import annotations

from .detectors import (
    DETECTOR_KINDS,
    AutoencoderDetector,
    BaseAnomalyDetector,
    IsolationForestDetector,
    load_detector,
    make_detector,
)

# Backwards-compatible name used across the trainer, API, tests, and evaluator.
AnomalyModel = IsolationForestDetector

__all__ = [
    "AnomalyModel",
    "BaseAnomalyDetector",
    "IsolationForestDetector",
    "AutoencoderDetector",
    "make_detector",
    "load_detector",
    "DETECTOR_KINDS",
]
