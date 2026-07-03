"""Central configuration and constants for the telemetry anomaly detector.

Everything that must be shared across the data generator, feature extractor,
model, and API lives here so the pipeline stays internally consistent.

The FEATURE_NAMES list is *contractual*: its order defines the meaning of every
column in the feature matrix and must never be reordered without retraining.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
TICKS_PER_SECOND = 20
SECONDS_PER_TICK = 1.0 / TICKS_PER_SECOND
DEFAULT_SEED = 1337

# Block taxonomy. "Valuable" ores are the ones an X-ray cheater beelines toward;
# coal/stone/dirt/gravel are the common blocks a legitimate miner churns through.
BLOCK_TYPES = [
    "stone",
    "dirt",
    "gravel",
    "coal_ore",
    "iron_ore",
    "gold_ore",
    "redstone_ore",
    "diamond_ore",
]
VALUABLE_ORES = frozenset({"iron_ore", "gold_ore", "redstone_ore", "diamond_ore"})
HIGH_VALUE_ORES = frozenset({"gold_ore", "diamond_ore"})

# ---------------------------------------------------------------------------
# Feature-engineering constants
# ---------------------------------------------------------------------------
EPS = 1e-9
MIN_EVENTS = 60  # minimum ticks (>=3s at 20 tps) required to score a session
SNAP_DEG_THRESHOLD = 30.0  # |delta yaw/pitch| per tick above this counts as a robotic "snap"
LOW_AIM_SPEED_DEG = 2.0  # angular speed (deg/tick) below this = "still" aim (jitter window)
HIGH_AIM_SPEED_DEG = 25.0  # angular speed (deg/tick) above this = "fast" (bimodality)
ANGLE_GRID_DEG = 15.0  # aimbots snap to a coarse grid; we measure residual to nearest multiple
DIRECTION_BINS = 8  # azimuth bins used for dig-direction Shannon entropy

# ---------------------------------------------------------------------------
# Feature vector  --  ORDER IS CONTRACTUAL
# ---------------------------------------------------------------------------
# index : name                    cheater tendency (relative to a normal miner)
#   0   : path_efficiency         HIGH   (straight beeline vs. wandering)
#   1   : mean_step_speed         ~      (weakly informative on its own)
#   2   : speed_cv                LOW    (constant-velocity tunnelling)
#   3   : vertical_travel_ratio   HIGH   (dives straight down to deep ore)
#   4   : heading_change_mean     LOW    (few direction changes)
#   5   : blocks_broken           ~      (context feature)
#   6   : valuable_ore_ratio      HIGH   (mines mostly the good stuff)
#   7   : non_ore_dig_ratio       LOW    (little wasted digging)
#   8   : ore_discovery_rate      HIGH   (finds valuable ore per block travelled)
#   9   : dig_direction_entropy   LOW    (mining concentrated along one axis)
#   10  : yaw_snap_rate           HIGH   (instantaneous horizontal snaps)
#   11  : pitch_snap_rate         HIGH   (instantaneous vertical snaps)
#   12  : aim_still_jitter        LOW    (no human micro-tremor when holding aim)
#   13  : aim_speed_bimodality    HIGH   (aim speed is either ~0 or huge, nothing between)
#   14  : angle_grid_residual     LOW    (angles land on a quantized grid)
FEATURE_NAMES = [
    "path_efficiency",
    "mean_step_speed",
    "speed_cv",
    "vertical_travel_ratio",
    "heading_change_mean",
    "blocks_broken",
    "valuable_ore_ratio",
    "non_ore_dig_ratio",
    "ore_discovery_rate",
    "dig_direction_entropy",
    "yaw_snap_rate",
    "pitch_snap_rate",
    "aim_still_jitter",
    "aim_speed_bimodality",
    "angle_grid_residual",
]
N_FEATURES = len(FEATURE_NAMES)

# ---------------------------------------------------------------------------
# Model / scoring
# ---------------------------------------------------------------------------
ISOFOREST_N_ESTIMATORS = 300
ISOFOREST_MAX_SAMPLES = "auto"
ISOFOREST_CONTAMINATION = "auto"
TARGET_FPR = 0.02  # calibrate the decision threshold to ~2% false-positive rate
MODEL_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_TITLE = "Telemetry Anomaly Detector"
API_DESCRIPTION = (
    "Real-time anomaly detection for 3D voxel game telemetry. "
    "Flags X-ray / ESP-style cheating from per-tick player logs."
)
API_VERSION = MODEL_VERSION
