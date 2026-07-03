"""Feature extraction: turn per-tick telemetry into a fixed 15-column vector.

Each session is a stream of per-tick rows describing a player's movement, aim,
and block-breaking activity. :func:`extract_features` collapses one such stream
into the feature vector defined by :data:`config.FEATURE_NAMES` (order is
contractual). :func:`features_for_sessions` maps a multi-session frame to a
tidy feature matrix, and :func:`feature_series` returns a single session's
vector as a labelled ``pandas.Series``.

Expected per-tick columns (one row per game tick, ordered within a session):

======================  =========================================================
column                  meaning
======================  =========================================================
``session_id``          session grouping key
``tick``                monotonically increasing tick index within the session
``x``, ``y``, ``z``     player position (blocks); ``y`` is the vertical axis
``yaw``                 horizontal aim angle in degrees (wraps at 360)
``pitch``               vertical aim angle in degrees
``block_type``          block broken this tick, else null / empty (see below)
``broke_block``         optional bool flag; if absent it is derived from
                        ``block_type`` being a non-null, non-empty value
======================  =========================================================

Every computation is numerically defended: divisions are guarded with
:data:`config.EPS`, circular yaw deltas are wrapped into ``[-180, 180]``, and
the returned vector is scrubbed so it never contains ``NaN`` or ``inf``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from . import config

__all__ = [
    "wrap_deg",
    "extract_features",
    "feature_series",
    "features_for_sessions",
    "extract_feature_frame",
]


def wrap_deg(delta: np.ndarray) -> np.ndarray:
    """Wrap angular differences (degrees) into the half-open range ``[-180, 180)``.

    A raw yaw difference such as ``350 - 10 = 340`` really represents a ``-20``
    degree turn; this collapses it to the shortest signed rotation so circular
    quantities are handled correctly.

    Parameters
    ----------
    delta:
        Array of angle differences in degrees.

    Returns
    -------
    numpy.ndarray
        The wrapped differences, ``float64``.
    """
    delta = np.asarray(delta, dtype=np.float64)
    return (delta + 180.0) % 360.0 - 180.0


def _safe_div(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray:
    """Elementwise division with the denominator floored away from zero by ``EPS``."""
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    return num / (np.abs(den) + config.EPS)


def _finite(value: float) -> float:
    """Coerce a scalar to a finite ``float`` (``NaN`` / ``inf`` collapse to ``0.0``)."""
    out = float(value)
    if not np.isfinite(out):
        return 0.0
    return out


def _column(df: pd.DataFrame, name: str) -> np.ndarray:
    """Return ``df[name]`` as a contiguous ``float64`` array (zeros if missing)."""
    if name not in df.columns:
        return np.zeros(len(df), dtype=np.float64)
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)


def _break_mask(df: pd.DataFrame) -> np.ndarray:
    """Boolean per-tick mask of ticks on which a block was broken."""
    if "broke_block" in df.columns:
        return df["broke_block"].fillna(False).to_numpy(dtype=bool)
    if "block_type" in df.columns:
        col = df["block_type"]
        present = col.notna().to_numpy(dtype=bool)
        # Treat empty / whitespace strings as "no break".
        stringy = col.astype("string").str.strip()
        non_empty = (stringy != "").fillna(False).to_numpy(dtype=bool)
        return np.logical_and(present, non_empty)
    return np.zeros(len(df), dtype=bool)


def _broken_block_types(df: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """Array of block-type strings for the ticks flagged in ``mask``."""
    if "block_type" not in df.columns:
        return np.empty(0, dtype=object)
    types = df["block_type"].astype("string").str.strip().to_numpy(dtype=object)
    return types[mask]


def _movement_features(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict[str, float]:
    """Path-shape and speed features from the position trace.

    Covers ``path_efficiency``, ``mean_step_speed``, ``speed_cv``,
    ``vertical_travel_ratio`` and ``heading_change_mean``.
    """
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)

    step_len = np.sqrt(dx * dx + dy * dy + dz * dz)
    total_path = float(step_len.sum())

    # Straight-line displacement between the first and last position.
    net = np.array([x[-1] - x[0], y[-1] - y[0], z[-1] - z[0]], dtype=np.float64)
    net_disp = float(np.sqrt(np.sum(net * net)))

    # path_efficiency: beeline / actual, clipped into [0, 1].
    path_efficiency = float(np.clip(_safe_div(net_disp, total_path), 0.0, 1.0))

    # mean_step_speed: blocks travelled per tick.
    mean_step_speed = float(step_len.mean()) if step_len.size else 0.0

    # speed_cv: coefficient of variation of step speed (std / mean).
    if step_len.size:
        speed_std = float(step_len.std())
        speed_cv = float(_safe_div(speed_std, mean_step_speed))
    else:
        speed_cv = 0.0

    # vertical_travel_ratio: share of path length spent moving vertically.
    vertical_travel_ratio = float(np.clip(_safe_div(np.abs(dy).sum(), total_path), 0.0, 1.0))

    # heading_change_mean: mean absolute change in horizontal travel azimuth
    # between consecutive moving steps (radians), wrapped circularly.
    horiz_len = np.sqrt(dx * dx + dz * dz)
    moving = horiz_len > config.EPS
    if np.count_nonzero(moving) >= 2:
        azimuth = np.arctan2(dz[moving], dx[moving])
        az_diff = np.diff(azimuth)
        # Wrap radian differences into [-pi, pi).
        az_diff = (az_diff + np.pi) % (2.0 * np.pi) - np.pi
        heading_change_mean = float(np.abs(az_diff).mean())
    else:
        heading_change_mean = 0.0

    return {
        "path_efficiency": _finite(path_efficiency),
        "mean_step_speed": _finite(mean_step_speed),
        "speed_cv": _finite(speed_cv),
        "vertical_travel_ratio": _finite(vertical_travel_ratio),
        "heading_change_mean": _finite(heading_change_mean),
        "_total_path": _finite(total_path),
    }


def _digging_features(
    df: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    total_path: float,
) -> dict[str, float]:
    """Block-breaking features.

    Covers ``blocks_broken``, ``valuable_ore_ratio``, ``non_ore_dig_ratio``,
    ``ore_discovery_rate`` and ``dig_direction_entropy``.
    """
    mask = _break_mask(df)
    n_broken = int(np.count_nonzero(mask))
    types = _broken_block_types(df, mask)

    n_valuable = int(sum(1 for t in types if t in config.VALUABLE_ORES))
    n_non_ore = n_broken - int(sum(1 for t in types if t in config.VALUABLE_ORES))
    # "non-ore" = anything not a valuable ore (stone / dirt / gravel / coal).

    blocks_broken = float(n_broken)
    valuable_ore_ratio = float(_safe_div(n_valuable, n_broken)) if n_broken else 0.0
    non_ore_dig_ratio = float(_safe_div(n_non_ore, n_broken)) if n_broken else 0.0

    # ore_discovery_rate: valuable ores found per block travelled.
    ore_discovery_rate = float(_safe_div(n_valuable, total_path))

    # dig_direction_entropy: Shannon entropy (bits) of the horizontal azimuth of
    # the player's aim at break ticks, binned into DIRECTION_BINS. A cheater who
    # mines along one axis concentrates mass in a few bins -> low entropy.
    dig_direction_entropy = _dig_direction_entropy(df, x, z, mask)

    return {
        "blocks_broken": _finite(blocks_broken),
        "valuable_ore_ratio": _finite(valuable_ore_ratio),
        "non_ore_dig_ratio": _finite(non_ore_dig_ratio),
        "ore_discovery_rate": _finite(ore_discovery_rate),
        "dig_direction_entropy": _finite(dig_direction_entropy),
    }


def _dig_direction_entropy(
    df: pd.DataFrame,
    x: np.ndarray,
    z: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Shannon entropy (bits) of dig direction over ``DIRECTION_BINS`` azimuth bins."""
    n_bins = int(config.DIRECTION_BINS)
    if n_bins <= 1:
        return 0.0

    # Prefer yaw at the break tick (the direction the player faces while digging);
    # fall back to the horizontal movement heading into that tick.
    if "yaw" in df.columns:
        yaw = _column(df, "yaw")[mask]
        angles = np.deg2rad(yaw)
    else:
        idx = np.flatnonzero(mask)
        idx = idx[idx > 0]
        if idx.size == 0:
            return 0.0
        angles = np.arctan2(z[idx] - z[idx - 1], x[idx] - x[idx - 1])

    if angles.size == 0:
        return 0.0

    # Bin azimuths into [0, 2*pi) and count.
    wrapped = np.mod(angles, 2.0 * np.pi)
    bin_idx = np.floor(wrapped / (2.0 * np.pi / n_bins)).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)

    total = counts.sum()
    if total <= 0.0:
        return 0.0
    probs = counts / total
    nz = probs > 0.0
    entropy = float(-np.sum(probs[nz] * np.log2(probs[nz])))
    return entropy


def _aim_features(df: pd.DataFrame) -> dict[str, float]:
    """Aim-dynamics features derived from the yaw / pitch traces.

    Covers ``yaw_snap_rate``, ``pitch_snap_rate``, ``aim_still_jitter``,
    ``aim_speed_bimodality`` and ``angle_grid_residual``.
    """
    yaw = _column(df, "yaw")
    pitch = _column(df, "pitch")
    n = yaw.size

    if n < 2:
        return {
            "yaw_snap_rate": 0.0,
            "pitch_snap_rate": 0.0,
            "aim_still_jitter": 0.0,
            "aim_speed_bimodality": 0.0,
            "angle_grid_residual": 0.0,
        }

    d_yaw = wrap_deg(np.diff(yaw))
    d_pitch = wrap_deg(np.diff(pitch))
    n_deltas = d_yaw.size

    # Snap rates: fraction of ticks whose per-tick angular change exceeds the
    # robotic snap threshold.
    yaw_snap_rate = float(np.count_nonzero(np.abs(d_yaw) > config.SNAP_DEG_THRESHOLD) / n_deltas)
    pitch_snap_rate = float(
        np.count_nonzero(np.abs(d_pitch) > config.SNAP_DEG_THRESHOLD) / n_deltas
    )

    # Combined angular speed per tick (degrees/tick).
    aim_speed = np.sqrt(d_yaw * d_yaw + d_pitch * d_pitch)

    # aim_still_jitter: std of aim speed while the player is holding "still"
    # (angular speed below LOW_AIM_SPEED_DEG). Humans micro-tremor; a bot is dead
    # flat, so a low value is suspicious.
    still = aim_speed < config.LOW_AIM_SPEED_DEG
    if np.count_nonzero(still) >= 2:
        aim_still_jitter = float(aim_speed[still].std())
    else:
        aim_still_jitter = 0.0

    # aim_speed_bimodality: fraction of ticks that are either "still" or "fast",
    # i.e. mass avoiding the human mid-band between the two thresholds.
    fast = aim_speed > config.HIGH_AIM_SPEED_DEG
    aim_speed_bimodality = float(np.count_nonzero(still | fast) / n_deltas)

    # angle_grid_residual: mean absolute residual (degrees) of yaw & pitch to the
    # nearest multiple of ANGLE_GRID_DEG. Aimbots snap to a coarse grid, driving
    # this toward zero.
    angle_grid_residual = _angle_grid_residual(yaw, pitch)

    return {
        "yaw_snap_rate": _finite(yaw_snap_rate),
        "pitch_snap_rate": _finite(pitch_snap_rate),
        "aim_still_jitter": _finite(aim_still_jitter),
        "aim_speed_bimodality": _finite(aim_speed_bimodality),
        "angle_grid_residual": _finite(angle_grid_residual),
    }


def _angle_grid_residual(yaw: np.ndarray, pitch: np.ndarray) -> float:
    """Mean absolute distance of yaw & pitch to the nearest ``ANGLE_GRID_DEG`` multiple."""
    grid = float(config.ANGLE_GRID_DEG)
    if grid <= 0.0:
        return 0.0
    both = np.concatenate([yaw, pitch])
    # Residual to nearest grid line, folded into [0, grid/2].
    residual = np.abs(both - np.round(both / grid) * grid)
    return float(residual.mean()) if residual.size else 0.0


def extract_features(session: pd.DataFrame) -> dict[str, float]:
    """Compute the 15-feature vector for a single session's per-tick frame.

    Parameters
    ----------
    session:
        Per-tick rows for exactly one session, ordered by ``tick`` if that
        column is present (otherwise assumed already ordered).

    Returns
    -------
    dict[str, float]
        Mapping from every name in :data:`config.FEATURE_NAMES` to a finite
        float. Sessions with fewer than :data:`config.MIN_EVENTS` ticks still
        return a complete, well-defined vector (degenerate but never ``NaN``).
    """
    df = session
    if "tick" in df.columns:
        df = df.sort_values("tick", kind="stable")

    x = _column(df, "x")
    y = _column(df, "y")
    z = _column(df, "z")

    if len(df) < 2:
        # Not enough motion to define diffs; emit an all-zero vector.
        return {name: 0.0 for name in config.FEATURE_NAMES}

    movement = _movement_features(x, y, z)
    total_path = movement.pop("_total_path")
    digging = _digging_features(df, x, y, z, total_path)
    aim = _aim_features(df)

    combined: dict[str, float] = {**movement, **digging, **aim}
    # Re-emit in contractual order, defending once more against NaN/inf.
    return {name: _finite(combined.get(name, 0.0)) for name in config.FEATURE_NAMES}


def feature_series(session: pd.DataFrame) -> pd.Series:
    """Return one session's feature vector as a ``pandas.Series`` indexed by name.

    The index is exactly :data:`config.FEATURE_NAMES`, in order.
    """
    features = extract_features(session)
    return pd.Series(
        [features[name] for name in config.FEATURE_NAMES],
        index=list(config.FEATURE_NAMES),
        dtype=np.float64,
        name=_session_label(session),
    )


def _session_label(session: pd.DataFrame) -> object:
    """Best-effort session identifier for labelling a feature series."""
    if "session_id" in session.columns and len(session):
        return session["session_id"].iloc[0]
    return None


def features_for_sessions(
    df: pd.DataFrame,
    session_col: str = "session_id",
    min_events: int | None = None,
) -> pd.DataFrame:
    """Extract features for every session in a multi-session telemetry frame.

    Parameters
    ----------
    df:
        Per-tick rows spanning one or more sessions.
    session_col:
        Column that groups rows into sessions. Defaults to ``"session_id"``.
    min_events:
        Sessions with fewer than this many ticks are skipped. Defaults to
        :data:`config.MIN_EVENTS`; pass ``0`` to keep every session.

    Returns
    -------
    pandas.DataFrame
        One row per qualifying session, indexed by the session id, with columns
        equal to :data:`config.FEATURE_NAMES` in order. Guaranteed finite.
    """
    threshold = config.MIN_EVENTS if min_events is None else int(min_events)

    if session_col not in df.columns:
        raise KeyError(
            f"session column {session_col!r} not found in telemetry frame; "
            f"available columns: {list(df.columns)}"
        )

    ids: list[object] = []
    rows: list[list[float]] = []
    for session_id, group in df.groupby(session_col, sort=False):
        if len(group) < threshold:
            continue
        features = extract_features(group)
        ids.append(session_id)
        rows.append([features[name] for name in config.FEATURE_NAMES])

    matrix = pd.DataFrame(
        rows,
        columns=list(config.FEATURE_NAMES),
        index=pd.Index(ids, name=session_col),
        dtype=np.float64,
    )
    # Final defensive scrub across the whole matrix.
    return matrix.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def extract_feature_frame(
    df: pd.DataFrame,
    session_col: str = "session_id",
    min_events: int = 0,
) -> pd.DataFrame:
    """Per-session feature matrix indexed by session id (train/eval entry point).

    Thin wrapper over :func:`features_for_sessions`. ``min_events`` defaults to
    ``0`` here so no generated session is silently dropped and every feature row
    stays aligned with its label.
    """
    return features_for_sessions(df, session_col=session_col, min_events=min_events)


def feature_matrix(sessions: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Build a feature matrix from an iterable of per-session frames.

    Convenience wrapper for callers holding already-split sessions rather than a
    single concatenated frame. Rows are indexed by each session's id when
    available, else by position.
    """
    ids: list[object] = []
    rows: list[list[float]] = []
    for pos, session in enumerate(sessions):
        features = extract_features(session)
        label = _session_label(session)
        ids.append(pos if label is None else label)
        rows.append([features[name] for name in config.FEATURE_NAMES])

    matrix = pd.DataFrame(
        rows,
        columns=list(config.FEATURE_NAMES),
        index=pd.Index(ids, name="session_id"),
        dtype=np.float64,
    )
    return matrix.replace([np.inf, -np.inf], 0.0).fillna(0.0)
