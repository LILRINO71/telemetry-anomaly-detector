"""Tests for the FastAPI service (``api.main``): schema, scoring, validation.

The app loads its model from ``config.MODEL_PATH`` at startup (lifespan). The
tests never rely on a pre-trained artifact on disk: they build a tiny fixture
model from synthetic data and inject it via ``app.dependency_overrides`` on the
``get_model`` dependency -- the documented FastAPI test-seam. The 503 "model not
loaded" path is exercised separately, with no override and no model file.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import main as api_main
from src import config
from src.model import AnomalyModel
from tests.conftest import feature_matrix, make_session_dict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def api_model() -> AnomalyModel:
    """A tiny model fitted on normal synthetic sessions, for the API to use."""
    matrix = feature_matrix("normal", n=20, base_seed=1000)
    model = AnomalyModel(target_fpr=config.TARGET_FPR, random_state=config.DEFAULT_SEED)
    model.fit(matrix)
    return model


@pytest.fixture()
def client(api_model: AnomalyModel) -> TestClient:
    """TestClient wired to the fixture model.

    Overrides ``get_model`` (used by the scoring routes) *and* injects the model
    into ``app.state`` (read directly by ``/health`` and ``/``), so the client is
    self-contained and does not depend on a trained ``model.joblib`` existing on
    disk.
    """
    api_main.app.dependency_overrides[api_main.get_model] = lambda: api_model
    with TestClient(api_main.app) as c:
        c.app.state.model = api_model
        yield c
    api_main.app.dependency_overrides.clear()
    api_main.app.state.model = None


def _events(label: str, seed: int) -> list[dict]:
    session = make_session_dict(label, seed)
    return [
        {
            "tick": t["t"],
            "x": t["x"],
            "y": t["y"],
            "z": t["z"],
            "yaw": t["yaw"],
            "pitch": t["pitch"],
            "block_type": t["block_type"],
        }
        for t in session["ticks"]
    ]


# ---------------------------------------------------------------------------
# Schema / metadata
# ---------------------------------------------------------------------------
def test_openapi_schema_served(client: TestClient) -> None:
    """The service publishes an OpenAPI schema with its configured metadata."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    info = resp.json()["info"]
    assert info["title"] == config.API_TITLE
    assert info["version"] == config.API_VERSION


def test_root_lists_endpoints(client: TestClient) -> None:
    """The root endpoint advertises the scoring routes and readiness."""
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == config.API_VERSION
    assert "/score" in body["endpoints"]


def test_health_ok(client: TestClient) -> None:
    """Health reports ok and that a model is loaded (via the override)."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_features"] == config.N_FEATURES


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def test_score_returns_valid_response(client: TestClient) -> None:
    """A valid session yields a well-formed anomaly verdict."""
    resp = client.post("/score", json={"session_id": "s1", "events": _events("normal", 1000)})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["session_id"] == "s1"
    assert isinstance(body["anomaly_score"], (int, float))
    assert isinstance(body["is_anomaly"], bool)
    assert body["n_events"] >= config.MIN_EVENTS
    assert body["model_version"] == config.MODEL_VERSION
    assert 1 <= len(body["top_features"]) <= config.N_FEATURES
    for contrib in body["top_features"]:
        assert contrib["feature"] in config.FEATURE_NAMES


def test_top_features_sorted_by_abs_zscore(client: TestClient) -> None:
    """``top_features`` is ranked by |zscore| descending."""
    resp = client.post("/score", json={"events": _events("cheater", 5000)})
    assert resp.status_code == 200, resp.text
    zs = [abs(c["zscore"]) for c in resp.json()["top_features"]]
    assert zs == sorted(zs, reverse=True)


def test_api_flags_cheater_higher_than_normal(client: TestClient) -> None:
    """A cheater session scores higher than a normal one through the API."""
    normal = client.post("/score", json={"events": _events("normal", 1000)}).json()
    cheater = client.post("/score", json={"events": _events("cheater", 5000)}).json()
    assert cheater["anomaly_score"] > normal["anomaly_score"]


def test_batch_scoring(client: TestClient) -> None:
    """The batch endpoint returns one verdict per submitted session."""
    payload = {
        "sessions": [
            {"session_id": "a", "events": _events("normal", 1000)},
            {"session_id": "b", "events": _events("cheater", 5000)},
        ]
    }
    resp = client.post("/score/batch", json=payload)
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert [r["session_id"] for r in results] == ["a", "b"]
    assert results[1]["anomaly_score"] > results[0]["anomaly_score"]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
def test_score_rejects_missing_events(client: TestClient) -> None:
    """A body missing the required ``events`` field is a 422."""
    resp = client.post("/score", json={"session_id": "x"})
    assert resp.status_code == 422


def test_score_rejects_malformed_event(client: TestClient) -> None:
    """An event missing required numeric fields is a 422 (pydantic validation)."""
    resp = client.post(
        "/score",
        json={"events": [{"x": 1.0, "y": 2.0}]},  # missing z/yaw/pitch
    )
    assert resp.status_code == 422


def test_score_rejects_non_numeric_field(client: TestClient) -> None:
    """A non-numeric coordinate is rejected by request validation."""
    events = _events("normal", 1000)
    events[0]["x"] = "not-a-number"
    resp = client.post("/score", json={"events": events})
    assert resp.status_code == 422


def test_score_rejects_too_short_session(client: TestClient) -> None:
    """A well-formed but too-short session is a 422 (below MIN_EVENTS)."""
    events = _events("normal", 1000)[: config.MIN_EVENTS - 1]
    resp = client.post("/score", json={"events": events})
    assert resp.status_code == 422
    assert str(config.MIN_EVENTS) in resp.text


def test_batch_rejects_empty_sessions(client: TestClient) -> None:
    """The batch endpoint requires at least one session (min_length=1)."""
    resp = client.post("/score/batch", json={"sessions": []})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Model-unavailable path (no override, no model file)
# ---------------------------------------------------------------------------
def test_score_returns_503_when_model_missing(tmp_path, monkeypatch) -> None:
    """With no loaded model, scoring returns 503 rather than crashing."""
    # Point the app at a non-existent model path and (re)start the lifespan so
    # ``app.state.model`` stays None; no dependency override is installed.
    monkeypatch.setattr(api_main, "MODEL_PATH", tmp_path / "missing.joblib")
    api_main.app.dependency_overrides.clear()
    with TestClient(api_main.app) as c:
        health = c.get("/health").json()
        assert health["model_loaded"] is False
        resp = c.post("/score", json={"events": _events("normal", 1000)})
        assert resp.status_code == 503
