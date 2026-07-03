"""FastAPI service that scores voxel-game telemetry sessions for cheating.

A *session* is a sequence of per-tick player events (position + aim + optional
block-break). The service turns the events into the contractual feature vector
via :mod:`src.features`, scores it with a persisted
:class:`~src.model.AnomalyModel` loaded from :data:`src.config.MODEL_PATH`, and
returns an anomaly verdict plus the features that contributed most to that
verdict (ranked by ``|zscore|``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import Body, Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from src import features as features_mod
from src import model as model_mod
from src.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    FEATURE_NAMES,
    MIN_EVENTS,
    MODEL_PATH,
    MODEL_VERSION,
    N_FEATURES,
)

_TOP_K = 5


# ---------------------------------------------------------------------------
# Request / response schema
# ---------------------------------------------------------------------------
class TelemetryEvent(BaseModel):
    """A single per-tick telemetry sample for one player.

    Field names match the per-tick columns consumed by
    :func:`src.features.extract_features`.
    """

    model_config = ConfigDict(extra="ignore")

    tick: int | None = Field(
        default=None,
        description="Monotonic tick index within the session (used to order).",
    )
    x: float = Field(description="World X position at this tick.")
    y: float = Field(description="World Y (vertical) position at this tick.")
    z: float = Field(description="World Z position at this tick.")
    yaw: float = Field(description="Horizontal aim angle in degrees.")
    pitch: float = Field(description="Vertical aim angle in degrees.")
    block_type: str | None = Field(
        default=None,
        description="Block type broken on this tick, or null if none.",
    )


class ScoreRequest(BaseModel):
    """A single session to score."""

    model_config = ConfigDict(extra="ignore")

    session_id: str | None = Field(default=None, description="Optional caller-supplied identifier.")
    events: list[TelemetryEvent] = Field(
        description="Ordered per-tick telemetry events for the session."
    )


class BatchScoreRequest(BaseModel):
    """A batch of sessions to score in one call."""

    model_config = ConfigDict(extra="ignore")

    sessions: list[ScoreRequest] = Field(min_length=1, description="One or more sessions to score.")


class FeatureContribution(BaseModel):
    """One feature's standardized contribution to the anomaly verdict."""

    feature: str = Field(description="Feature name (from FEATURE_NAMES).")
    value: float = Field(description="Raw feature value for this session.")
    zscore: float = Field(description="Standardized deviation from the norm.")


class ScoreResponse(BaseModel):
    """Anomaly verdict for a single session."""

    session_id: str | None = Field(default=None)
    anomaly_score: float = Field(description="Anomaly score; higher == more anomalous.")
    is_anomaly: bool = Field(description="Whether the score crosses the threshold.")
    n_events: int = Field(description="Number of events scored.")
    top_features: list[FeatureContribution] = Field(
        description="Top contributing features ranked by |zscore|, descending."
    )
    model_version: str = Field(default=MODEL_VERSION)


class BatchScoreResponse(BaseModel):
    """Anomaly verdicts for a batch of sessions."""

    results: list[ScoreResponse]


class HealthResponse(BaseModel):
    """Liveness / readiness signal."""

    status: str
    model_loaded: bool
    model_version: str = Field(default=MODEL_VERSION)
    n_features: int = Field(default=N_FEATURES)


class RootResponse(BaseModel):
    """Service metadata returned from the root endpoint."""

    name: str = Field(default=API_TITLE)
    description: str = Field(default=API_DESCRIPTION)
    version: str = Field(default=API_VERSION)
    model_loaded: bool
    endpoints: list[str]


# ---------------------------------------------------------------------------
# Lifespan: load the model into app.state once, at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the persisted model into ``app.state`` for the app's lifetime."""
    app.state.model = None
    if MODEL_PATH.exists():
        app.state.model = model_mod.AnomalyModel.load(MODEL_PATH)
    yield
    app.state.model = None


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
def get_model(request: Request) -> Any:
    """Return the loaded model or raise 503 if it is unavailable."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded. Expected a trained model at {MODEL_PATH}.",
        )
    return model


ModelDep = Annotated[Any, Depends(get_model)]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def _events_to_frame(events: Sequence[TelemetryEvent]) -> pd.DataFrame:
    """Turn validated events into a per-tick DataFrame for the extractor."""
    return pd.DataFrame([event.model_dump() for event in events])


def _coerce_vector(out: Any) -> np.ndarray:
    """Normalise a feature-extractor result into an ordered float ndarray.

    Accepts the contractual ``dict[str, float]`` keyed by
    :data:`~src.config.FEATURE_NAMES`, as well as a ``pandas.Series`` or a bare
    array, so the API stays robust to the extractor's exact return flavour.
    """
    if isinstance(out, dict):
        return np.array([float(out[name]) for name in FEATURE_NAMES], dtype=float)
    if isinstance(out, pd.Series):
        if set(FEATURE_NAMES).issubset(set(out.index)):
            return out.reindex(FEATURE_NAMES).to_numpy(dtype=float)
        return out.to_numpy(dtype=float)
    return np.asarray(out, dtype=float).ravel()


def _extract_feature_vector(events: Sequence[TelemetryEvent]) -> np.ndarray:
    """Extract the contractual feature vector for one session."""
    frame = _events_to_frame(events)
    vector = _coerce_vector(features_mod.extract_features(frame))
    if vector.shape[0] != N_FEATURES:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Feature extractor returned {vector.shape[0]} features, expected {N_FEATURES}."
            ),
        )
    return vector


def _zscores(model: Any, vector: np.ndarray) -> np.ndarray:
    """Return per-feature z-scores using the model's fitted StandardScaler.

    The scaler's standardization *is* the z-score: ``(x - mean_) / scale_``.
    """
    scaler = getattr(model, "scaler_", None)
    if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        mean = np.asarray(scaler.mean_, dtype=float).reshape(-1)
        scale = np.asarray(scaler.scale_, dtype=float).reshape(-1)
        safe_scale = np.where(scale > 0.0, scale, 1.0)
        return (vector - mean) / safe_scale
    return np.zeros_like(vector)


def _top_contributions(
    vector: np.ndarray, zscores: np.ndarray, limit: int = _TOP_K
) -> list[FeatureContribution]:
    """Rank features by ``|zscore|`` (descending) and take the top ``limit``."""
    order = np.argsort(-np.abs(zscores))
    return [
        FeatureContribution(
            feature=FEATURE_NAMES[int(idx)],
            value=float(vector[int(idx)]),
            zscore=float(zscores[int(idx)]),
        )
        for idx in order[:limit]
    ]


def _scalar_score(model: Any, vector: np.ndarray) -> float:
    """Return a single anomaly score (higher == more anomalous)."""
    scores = np.asarray(model.score_samples(vector), dtype=float).reshape(-1)
    return float(scores[0])


def _scalar_predict(model: Any, vector: np.ndarray) -> bool:
    """Return whether the session is flagged as an anomaly."""
    labels = np.asarray(model.predict(vector)).reshape(-1)
    return bool(int(labels[0]))


def _score_session(model: Any, session: ScoreRequest) -> ScoreResponse:
    """Validate event count, extract features, score, and rank contributions."""
    n_events = len(session.events)
    if n_events < MIN_EVENTS:
        # 422 Unprocessable Content: the request is well-formed but the session
        # is too short to yield a meaningful feature vector.
        raise HTTPException(
            status_code=422,
            detail=(f"Session has {n_events} events; at least {MIN_EVENTS} are required to score."),
        )

    vector = _extract_feature_vector(session.events)
    anomaly_score = _scalar_score(model, vector)
    is_anomaly = _scalar_predict(model, vector)
    zscores = _zscores(model, vector)

    return ScoreResponse(
        session_id=session.session_id,
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly,
        n_events=n_events,
        top_features=_top_contributions(vector, zscores),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root(request: Request) -> RootResponse:
    """Return service metadata and readiness."""
    return RootResponse(
        model_loaded=getattr(request.app.state, "model", None) is not None,
        endpoints=["/", "/health", "/score", "/score/batch"],
    )


@app.get("/health")
def health(request: Request) -> HealthResponse:
    """Report liveness and whether the model is loaded and ready."""
    loaded = getattr(request.app.state, "model", None) is not None
    return HealthResponse(status="ok", model_loaded=loaded)


@app.post("/score")
def score(model: ModelDep, payload: Annotated[ScoreRequest, Body()]) -> ScoreResponse:
    """Score a single session and return its anomaly verdict."""
    return _score_session(model, payload)


@app.post("/score/batch")
def score_batch(
    model: ModelDep, payload: Annotated[BatchScoreRequest, Body()]
) -> BatchScoreResponse:
    """Score a batch of sessions and return one verdict per session."""
    results = [_score_session(model, session) for session in payload.sessions]
    return BatchScoreResponse(results=results)
