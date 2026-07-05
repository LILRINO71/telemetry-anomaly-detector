"""Interactive Streamlit demo for the telemetry anomaly detector.

Run locally from the repo root::

    pip install -r requirements-demo.txt
    streamlit run streamlit_app.py

Generate (or paste) a player session, pick a detector, and see the anomaly score,
the flag decision against the calibrated threshold, the features that drove the
verdict, and a visualization of the player's path and aim.

Deployable as-is to Streamlit Community Cloud (point it at this file); the model
is trained on first load and cached, so no committed model artifact is required.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src import config, scoring
from src import generate_data as gen
from src.detectors import DETECTOR_KINDS

_LEGIT = "#4c72b0"
_CHEAT = "#c44e52"


@st.cache_resource(show_spinner="Training detector on synthetic play…")
def load_model(kind: str):
    """Train (once, then cache) a detector on freshly generated normal play."""
    return scoring.train_demo_model(kind=kind)


def make_events(source: str, label: str, difficulty: float, seed: int, pasted: str):
    """Return (events, error). Exactly one of the two is meaningful."""
    if source == "Generate synthetic":
        session = gen.generate_session(
            f"{label}-{seed}", label, difficulty, np.random.default_rng(seed), seed
        )
        return scoring.session_to_events(session), None
    try:
        parsed = json.loads(pasted)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON: {exc}"
    events = parsed.get("events", parsed) if isinstance(parsed, dict) else parsed
    if not isinstance(events, list) or not events:
        return None, "Expected a non-empty list of events (or an object with an 'events' key)."
    return events, None


def plot_path(events: list[dict]):
    """Top-down (x, z) path colored by time, plus a depth-over-time panel."""
    x = np.array([e.get("x", 0.0) for e in events], dtype=float)
    y = np.array([e.get("y", 0.0) for e in events], dtype=float)
    z = np.array([e.get("z", 0.0) for e in events], dtype=float)
    t = np.arange(len(events))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(x, z, color="#cccccc", lw=0.8, zorder=1)
    sc = ax1.scatter(x, z, c=t, cmap="viridis", s=10, zorder=2)
    ax1.scatter([x[0]], [z[0]], color="green", s=60, marker="o", label="start", zorder=3)
    ax1.scatter([x[-1]], [z[-1]], color="red", s=60, marker="X", label="end", zorder=3)
    ax1.set_title("Path (top-down)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("z")
    ax1.legend(fontsize=8, loc="best")
    ax1.set_aspect("equal", adjustable="datalim")
    fig.colorbar(sc, ax=ax1, label="tick")

    ax2.plot(t, y, color="#8172b3", lw=1.2)
    ax2.set_title("Depth over time (y)")
    ax2.set_xlabel("tick")
    ax2.set_ylabel("y (height)")
    fig.tight_layout()
    return fig


def plot_aim(events: list[dict]):
    """Yaw and pitch over time — smooth panning vs robotic snaps."""
    t = np.arange(len(events))
    yaw = np.array([e.get("yaw", 0.0) for e in events], dtype=float)
    pitch = np.array([e.get("pitch", 0.0) for e in events], dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 4.4), sharex=True)
    ax1.plot(t, yaw, color=_LEGIT, lw=0.9)
    ax1.set_ylabel("yaw (°)")
    ax1.set_title("Aim over time (look for instant snaps vs smooth panning)")
    ax2.plot(t, pitch, color=_CHEAT, lw=0.9)
    ax2.set_ylabel("pitch (°)")
    ax2.set_xlabel("tick")
    fig.tight_layout()
    return fig


def plot_top_features(top_features: list[dict]):
    """Horizontal bar chart of the top contributing features by signed z-score."""
    names = [c["feature"] for c in top_features][::-1]
    zs = [c["zscore"] for c in top_features][::-1]
    colors = [_CHEAT if z > 0 else _LEGIT for z in zs]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(names, zs, color=colors)
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_xlabel("z-score vs. legitimate play  (→ more anomalous)")
    ax.set_title("Top contributing features")
    fig.tight_layout()
    return fig


def render_about() -> None:
    """Explain the models and show the committed comparison, if present."""
    st.subheader("About the models")
    st.markdown(
        "Two unsupervised detectors share one interface. Both train on **legitimate "
        "play only** and flag deviations, so they generalize to cheats never seen.\n\n"
        "- **Isolation Forest** — isolates anomalies in a few random tree splits.\n"
        "- **Autoencoder** — a bottleneck neural net; high reconstruction error = anomalous."
    )
    comparison = config.MODELS_DIR / "comparison.json"
    if comparison.exists():
        data = json.loads(comparison.read_text(encoding="utf-8"))
        rows = {
            k: {m: round(v[m], 3) for m in ("roc_auc", "precision", "recall", "f1")}
            for k, v in data.get("results", {}).items()
        }
        if rows:
            st.dataframe(rows)
    figure = config.REPORTS_DIR / "model_comparison.png"
    if figure.exists():
        st.image(str(figure), caption="Held-out ROC comparison")


def main() -> None:
    st.set_page_config(page_title="Telemetry Anomaly Detector", page_icon="🛡️", layout="wide")
    st.title("🛡️ Telemetry Anomaly Detector")
    st.caption(
        "Unsupervised anti-cheat for 3D voxel-game telemetry — catches X-ray & ESP "
        "behavior from per-tick player logs."
    )

    with st.sidebar:
        st.header("⚙️ Configuration")
        kind = st.selectbox("Detector", list(DETECTOR_KINDS), index=0)
        st.divider()
        source = st.radio("Session source", ["Generate synthetic", "Paste JSON"])
        label, difficulty, seed, pasted = "normal", 0.3, 42, ""
        if source == "Generate synthetic":
            label = st.selectbox("Behavior", ["normal", "cheater"])
            difficulty = st.slider(
                "Cheater mimicry (difficulty)",
                0.0,
                1.0,
                0.3,
                0.05,
                help="Higher = the cheater throttles harder to look human.",
            )
            seed = int(st.number_input("Seed", min_value=0, max_value=999_999, value=42))
        else:
            pasted = st.text_area(
                "Session JSON",
                height=200,
                placeholder='{"events": [{"tick": 0, "x": 0, "y": 64, "z": 0, '
                '"yaw": 90, "pitch": 0, "block_type": null}, ...]}',
            )
        run = st.button("Score session", type="primary", use_container_width=True)

    events, error = make_events(source, label, difficulty, seed, pasted)
    if source == "Paste JSON" and not pasted:
        st.info("Paste a session on the left, or switch to **Generate synthetic**.")
        render_about()
        return
    if error:
        st.error(error)
        return
    if len(events) < config.MIN_EVENTS:
        st.warning(
            f"Session has {len(events)} events; at least {config.MIN_EVENTS} are needed to score."
        )
        return

    if not run and source == "Paste JSON":
        st.info("Press **Score session** to evaluate.")
        return

    model = load_model(kind)
    result = scoring.score_session(model, events)

    st.subheader("Verdict")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Anomaly score", f"{result['anomaly_score']:.3f}")
    c2.metric("Threshold", f"{result['threshold']:.3f}")
    c3.metric("Events", result["n_events"])
    if result["is_anomaly"]:
        c4.metric("Verdict", "🚨 FLAGGED")
        st.error(
            "This session is flagged as **likely cheating** (score exceeds the calibrated threshold)."
        )
    else:
        c4.metric("Verdict", "✅ CLEAN")
        st.success("This session looks like **legitimate play** (score below the threshold).")

    left, right = st.columns([1, 1])
    with left:
        st.pyplot(plot_top_features(result["top_features"]))
    with right:
        st.markdown("**All 15 features**")
        st.dataframe(
            {
                "value": {k: round(v, 4) for k, v in result["features"].items()},
                "z-score": {k: round(v, 3) for k, v in result["zscores"].items()},
            }
        )

    st.subheader("Session behavior")
    st.pyplot(plot_path(events))
    st.pyplot(plot_aim(events))

    with st.expander("ℹ️ About the models"):
        render_about()


if __name__ == "__main__":
    main()
