"""Synthetic telemetry generator for the voxel-game anomaly detector.

Produces per-tick player session logs for two populations:

* ``normal``  -- a legitimate miner: wanders, tunnels along the local terrain,
  breaks mostly worthless blocks, and moves the camera with human micro-tremor.
* ``cheater`` -- an X-ray / aimbot user: beelines toward valuable ore, digs
  almost nothing worthless, snaps the camera onto a quantized angle grid, and
  holds aim perfectly still between snaps.

The two populations are engineered to diverge along the three behavioural axes
named in :mod:`src.config` -- pathing, mining, and aim -- while a ``difficulty``
knob in ``[0, 1]`` linearly blends each cheater trait back toward the normal
distribution so the classification problem can be made arbitrarily hard.

Output is JSONL: one line per session. Each line is a session object holding
metadata plus a ``ticks`` array of per-tick records. The record schema is the
raw contract consumed by the feature extractor::

    session = {
        "session_id": str,
        "label": "normal" | "cheater",
        "difficulty": float,
        "seed": int,
        "ticks": [
            {
                "t": int,                 # tick index, 0-based
                "x": float, "y": float, "z": float,   # player position (blocks)
                "yaw": float,             # heading, degrees in [0, 360)
                "pitch": float,           # elevation, degrees in [-90, 90]
                "dig": bool,              # broke a block this tick
                "block_type": str | None  # block broken, else None
            },
            ...
        ]
    }
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src import config

# ---------------------------------------------------------------------------
# Population defaults
# ---------------------------------------------------------------------------
_MIN_TICKS = config.MIN_EVENTS  # never emit an unscoreable session
_DEFAULT_MEAN_TICKS = 900  # ~45 s at 20 tps
_TICKS_JITTER = 300  # +/- spread on session length

# Block-break probability per tick while actively mining, by population.
_NORMAL_DIG_PROB = 0.55
_CHEATER_DIG_PROB = 0.45

# Per-population sampling weights over ``config.BLOCK_TYPES``. Normal players
# churn through worthless stone/dirt; cheaters skip straight to valuable ore.
_NORMAL_BLOCK_WEIGHTS = np.array(
    [0.52, 0.18, 0.10, 0.12, 0.05, 0.018, 0.010, 0.002], dtype=np.float64
)
_CHEATER_BLOCK_WEIGHTS = np.array(
    [0.10, 0.03, 0.02, 0.10, 0.30, 0.18, 0.15, 0.12], dtype=np.float64
)


def _normalise(weights: np.ndarray) -> np.ndarray:
    """Return a copy of ``weights`` scaled to sum to 1."""
    total = float(weights.sum())
    return weights / total if total > 0 else weights


_NORMAL_BLOCK_WEIGHTS = _normalise(_NORMAL_BLOCK_WEIGHTS)
_CHEATER_BLOCK_WEIGHTS = _normalise(_CHEATER_BLOCK_WEIGHTS)


def _blend(cheater_value: float, normal_value: float, difficulty: float) -> float:
    """Linearly interpolate a cheater trait toward its normal counterpart.

    ``difficulty == 0`` yields the pure cheater value (easy to detect);
    ``difficulty == 1`` collapses the trait onto the normal value (impossible).
    """
    d = float(np.clip(difficulty, 0.0, 1.0))
    return (1.0 - d) * cheater_value + d * normal_value


def _blend_weights(cheater_w: np.ndarray, normal_w: np.ndarray, difficulty: float) -> np.ndarray:
    """Blend two categorical weight vectors and renormalise."""
    d = float(np.clip(difficulty, 0.0, 1.0))
    return _normalise((1.0 - d) * cheater_w + d * normal_w)


@dataclass(frozen=True)
class SessionParams:
    """Resolved per-session behavioural parameters after difficulty blending."""

    label: str
    difficulty: float
    n_ticks: int
    step_speed: float  # blocks / tick along the movement path
    speed_jitter: float  # relative sigma of per-tick speed noise
    wander_sigma: float  # heading random-walk sigma (deg / tick)
    vertical_bias: float  # fraction of motion directed straight down
    dig_prob: float  # P(break a block | mining tick)
    block_weights: np.ndarray  # categorical over config.BLOCK_TYPES
    snap_prob: float  # P(camera snap this tick)
    snap_to_grid: float  # blend toward config.ANGLE_GRID_DEG quantization
    aim_tremor: float  # human micro-tremor sigma while holding aim (deg)
    fast_move_frac: float  # fraction of camera motion that is a fast sweep


# ---------------------------------------------------------------------------
# Parameter resolution
# ---------------------------------------------------------------------------
def _resolve_params(label: str, difficulty: float, rng: np.random.Generator) -> SessionParams:
    """Draw the behavioural parameters for one session of the given ``label``."""
    n_ticks = int(rng.normal(_DEFAULT_MEAN_TICKS, _TICKS_JITTER))
    n_ticks = max(_MIN_TICKS, n_ticks)

    if label == "normal":
        return SessionParams(
            label=label,
            difficulty=difficulty,
            n_ticks=n_ticks,
            step_speed=float(rng.uniform(0.04, 0.10)),
            speed_jitter=0.45,
            wander_sigma=18.0,
            vertical_bias=0.12,
            dig_prob=_NORMAL_DIG_PROB,
            block_weights=_NORMAL_BLOCK_WEIGHTS.copy(),
            snap_prob=0.02,
            snap_to_grid=0.0,
            aim_tremor=1.2,
            fast_move_frac=0.15,
        )

    if label != "cheater":
        raise ValueError(f"unknown label: {label!r}")

    # Cheater traits, each blended from its distinctive value toward the normal
    # one by ``difficulty``.
    return SessionParams(
        label=label,
        difficulty=difficulty,
        n_ticks=n_ticks,
        step_speed=float(rng.uniform(0.08, 0.14)),
        speed_jitter=_blend(0.06, 0.45, difficulty),  # constant velocity
        wander_sigma=_blend(3.0, 18.0, difficulty),  # few heading changes
        vertical_bias=_blend(0.55, 0.12, difficulty),  # dives to deep ore
        dig_prob=_CHEATER_DIG_PROB,
        block_weights=_blend_weights(_CHEATER_BLOCK_WEIGHTS, _NORMAL_BLOCK_WEIGHTS, difficulty),
        snap_prob=_blend(0.22, 0.02, difficulty),  # frequent snaps
        snap_to_grid=_blend(0.92, 0.0, difficulty),  # quantized angles
        aim_tremor=_blend(0.05, 1.2, difficulty),  # no micro-tremor
        fast_move_frac=_blend(0.85, 0.15, difficulty),  # bimodal aim speed
    )


# ---------------------------------------------------------------------------
# Tick simulation
# ---------------------------------------------------------------------------
def _simulate_ticks(params: SessionParams, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Roll out one session's per-tick position, orientation, and dig events."""
    n = params.n_ticks

    # --- Movement / pathing --------------------------------------------------
    # A heading random walk drives horizontal motion; ``vertical_bias`` steals a
    # fraction of each step for a straight-down dive (the X-ray beeline).
    heading = float(rng.uniform(0.0, 360.0))
    x = float(rng.uniform(-64.0, 64.0))
    y = float(rng.uniform(24.0, 64.0))  # start mid-depth
    z = float(rng.uniform(-64.0, 64.0))

    # --- Aim / camera --------------------------------------------------------
    yaw = float(rng.uniform(0.0, 360.0))
    pitch = float(rng.uniform(-30.0, 30.0))

    ticks: list[dict[str, Any]] = []
    for t in range(n):
        # Heading random walk (small for cheaters -> low heading_change_mean).
        heading = (heading + rng.normal(0.0, params.wander_sigma)) % 360.0
        speed = params.step_speed * (1.0 + rng.normal(0.0, params.speed_jitter))
        speed = max(0.0, speed)

        horiz = speed * (1.0 - params.vertical_bias)
        vert = speed * params.vertical_bias
        rad = math.radians(heading)
        x += horiz * math.cos(rad)
        z += horiz * math.sin(rad)
        # Net downward dive with mild noise; keep inside a sane world height.
        y -= vert + abs(rng.normal(0.0, 0.01))
        y = float(np.clip(y, -60.0, 120.0))

        # Camera update: with probability ``snap_prob`` perform a large jump,
        # optionally quantized to the angle grid; otherwise hold aim with only
        # micro-tremor (humans) or near-perfect stillness (cheaters).
        if rng.random() < params.snap_prob:
            if rng.random() < params.fast_move_frac:
                d_yaw = float(rng.uniform(-120.0, 120.0))
                d_pitch = float(rng.uniform(-60.0, 60.0))
            else:
                d_yaw = float(rng.normal(0.0, 8.0))
                d_pitch = float(rng.normal(0.0, 4.0))
            yaw = (yaw + d_yaw) % 360.0
            pitch = float(np.clip(pitch + d_pitch, -90.0, 90.0))
            # Snap onto the coarse grid, blended by ``snap_to_grid``.
            if params.snap_to_grid > 0.0:
                g = config.ANGLE_GRID_DEG
                gy = round(yaw / g) * g
                gp = round(pitch / g) * g
                yaw = (1.0 - params.snap_to_grid) * yaw + params.snap_to_grid * gy
                pitch = (1.0 - params.snap_to_grid) * pitch + params.snap_to_grid * gp
                yaw %= 360.0
                pitch = float(np.clip(pitch, -90.0, 90.0))
        else:
            yaw = (yaw + rng.normal(0.0, params.aim_tremor)) % 360.0
            pitch = float(np.clip(pitch + rng.normal(0.0, params.aim_tremor), -90.0, 90.0))

        # Dig event.
        dig = bool(rng.random() < params.dig_prob)
        block_type: str | None = None
        if dig:
            idx = int(rng.choice(len(config.BLOCK_TYPES), p=params.block_weights))
            block_type = config.BLOCK_TYPES[idx]

        ticks.append(
            {
                "t": t,
                "x": round(x, 4),
                "y": round(y, 4),
                "z": round(z, 4),
                "yaw": round(yaw % 360.0, 4),
                "pitch": round(pitch, 4),
                "dig": dig,
                "block_type": block_type,
            }
        )

    return ticks


def generate_session(
    session_id: str,
    label: str,
    difficulty: float,
    rng: np.random.Generator,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate one labelled session dict with its per-tick ``ticks`` array.

    Parameters
    ----------
    session_id:
        Stable identifier written into the record.
    label:
        Either ``"normal"`` or ``"cheater"``.
    difficulty:
        Overlap knob in ``[0, 1]``; higher blends cheater traits toward normal.
    rng:
        Seeded numpy ``Generator`` supplying all randomness.
    seed:
        Dataset seed recorded on the session for provenance; ``None`` if the
        session is generated ad hoc without a known seed.
    """
    params = _resolve_params(label, difficulty, rng)
    ticks = _simulate_ticks(params, rng)
    return {
        "session_id": session_id,
        "label": label,
        "difficulty": round(float(difficulty), 4),
        "seed": int(seed) if seed is not None else None,
        "ticks": ticks,
    }


def generate_dataset(
    n_normal: int,
    n_cheater: int,
    difficulty: float,
    seed: int,
) -> Iterator[dict[str, Any]]:
    """Yield ``n_normal`` normal then ``n_cheater`` cheater sessions.

    A single seeded :class:`numpy.random.Generator` drives the whole dataset so
    output is fully reproducible from ``seed`` alone. Sessions are interleaved
    deterministically by construction order (all normals, then all cheaters).
    """
    rng = np.random.default_rng(seed)
    for i in range(n_normal):
        yield generate_session(f"normal-{i:06d}", "normal", difficulty, rng, seed)
    for i in range(n_cheater):
        yield generate_session(f"cheater-{i:06d}", "cheater", difficulty, rng, seed)


# ---------------------------------------------------------------------------
# JSONL IO
# ---------------------------------------------------------------------------
def write_jsonl(sessions: Iterator[dict[str, Any]], path: Path) -> int:
    """Write ``sessions`` to ``path`` as UTF-8 JSONL. Return the count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for session in sessions:
            fh.write(json.dumps(session, ensure_ascii=False))
            fh.write("\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield session dicts from a UTF-8 JSONL file, skipping blank lines."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic voxel-game telemetry sessions as JSONL."
    )
    parser.add_argument(
        "--n-normal",
        type=int,
        default=800,
        help="number of legitimate sessions to generate (default: 800)",
    )
    parser.add_argument(
        "--n-cheater",
        type=int,
        default=200,
        help="number of cheater sessions to generate (default: 200)",
    )
    parser.add_argument(
        "--difficulty",
        type=float,
        default=0.88,
        help="overlap knob in [0, 1]; higher = harder (default: 0.88)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.DEFAULT_SEED,
        help=f"random seed (default: {config.DEFAULT_SEED})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=config.RAW_DATA_DIR / "sessions.jsonl",
        help="output JSONL path (default: data/raw/sessions.jsonl)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse args, generate the dataset, write JSONL."""
    args = _build_parser().parse_args(argv)
    if not 0.0 <= args.difficulty <= 1.0:
        raise SystemExit("--difficulty must be in [0, 1]")
    if args.n_normal < 0 or args.n_cheater < 0:
        raise SystemExit("--n-normal and --n-cheater must be non-negative")

    sessions = generate_dataset(
        n_normal=args.n_normal,
        n_cheater=args.n_cheater,
        difficulty=args.difficulty,
        seed=args.seed,
    )
    total = write_jsonl(sessions, args.out)
    print(
        f"wrote {total} sessions "
        f"({args.n_normal} normal, {args.n_cheater} cheater) "
        f"to {args.out} [seed={args.seed}, difficulty={args.difficulty}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
