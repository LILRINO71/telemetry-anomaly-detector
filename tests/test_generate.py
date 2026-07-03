"""Tests for the synthetic telemetry generator (``src.generate_data``)."""

from __future__ import annotations

import numpy as np
import pytest

from src import config, generate_data
from tests.conftest import make_session_dict

_TICK_KEYS = {"t", "x", "y", "z", "yaw", "pitch", "dig", "block_type"}


def test_session_schema_and_length() -> None:
    """A session carries its metadata and enough ticks to be scorable."""
    session = make_session_dict("normal", seed=7)
    assert session["label"] == "normal"
    assert session["session_id"]
    ticks = session["ticks"]
    assert len(ticks) >= config.MIN_EVENTS
    assert _TICK_KEYS.issubset(ticks[0].keys())


def test_tick_indices_are_monotonic() -> None:
    """The per-tick ``t`` index runs 0..n-1 in order."""
    ticks = make_session_dict("cheater", seed=8)["ticks"]
    ts = [tick["t"] for tick in ticks]
    assert ts == list(range(len(ts)))


def test_generation_is_deterministic_for_fixed_seed() -> None:
    """Same seed + label + difficulty reproduces the session exactly."""
    a = make_session_dict("normal", seed=42)
    b = make_session_dict("normal", seed=42)
    assert a == b


def test_different_seeds_differ() -> None:
    """Distinct seeds produce distinct trajectories."""
    a = make_session_dict("normal", seed=1)
    b = make_session_dict("normal", seed=2)
    assert a["ticks"] != b["ticks"]


def test_positions_and_angles_are_finite_and_bounded() -> None:
    """Coordinates are finite; yaw in [0,360), pitch in [-90,90]."""
    ticks = make_session_dict("cheater", seed=3)["ticks"]
    for tick in ticks:
        for axis in ("x", "y", "z"):
            assert np.isfinite(tick[axis]), f"non-finite {axis}"
        assert 0.0 <= tick["yaw"] < 360.0, tick["yaw"]
        assert -90.0 <= tick["pitch"] <= 90.0, tick["pitch"]


def test_no_nan_in_session() -> None:
    """The generator never emits NaN in numeric tick fields."""
    for label in ("normal", "cheater"):
        for tick in make_session_dict(label, seed=11)["ticks"]:
            for key in ("x", "y", "z", "yaw", "pitch"):
                assert not np.isnan(tick[key]), f"NaN in {label}.{key}"


def test_broken_blocks_are_known_types() -> None:
    """Every broken block references the config taxonomy; dig implies a type."""
    allowed = set(config.BLOCK_TYPES)
    seen: set[str] = set()
    for tick in make_session_dict("cheater", seed=9)["ticks"]:
        if tick["dig"]:
            assert tick["block_type"] in allowed, tick["block_type"]
            seen.add(tick["block_type"])
        else:
            assert tick["block_type"] is None
    assert seen, "cheater session broke no blocks at difficulty 0"
    assert seen.issubset(allowed)


def test_generate_dataset_counts_and_order() -> None:
    """``generate_dataset`` yields n_normal normals then n_cheater cheaters."""
    sessions = list(
        generate_data.generate_dataset(
            n_normal=3, n_cheater=2, difficulty=0.0, seed=config.DEFAULT_SEED
        )
    )
    assert len(sessions) == 5
    labels = [s["label"] for s in sessions]
    assert labels == ["normal", "normal", "normal", "cheater", "cheater"]


def test_generate_dataset_is_reproducible() -> None:
    """The whole dataset is reproducible from the seed alone."""
    kwargs = dict(n_normal=2, n_cheater=2, difficulty=0.2, seed=99)
    a = list(generate_data.generate_dataset(**kwargs))
    b = list(generate_data.generate_dataset(**kwargs))
    assert a == b


def _valuable_fraction(session: dict) -> float | None:
    broken = [t["block_type"] for t in session["ticks"] if t["dig"] and t["block_type"]]
    if not broken:
        return None
    return float(np.mean([b in config.VALUABLE_ORES for b in broken]))


def test_cheaters_mine_more_valuable_ore_on_average() -> None:
    """Cheater sessions dig a higher fraction of valuable ore than miners.

    This is the ground-truth behavioural signal the detector learns.
    """
    cheater = [
        f
        for s in range(600, 606)
        if (f := _valuable_fraction(make_session_dict("cheater", s))) is not None
    ]
    normal = [
        f
        for s in range(200, 206)
        if (f := _valuable_fraction(make_session_dict("normal", s))) is not None
    ]
    assert cheater and normal
    assert np.mean(cheater) > np.mean(normal)


def test_difficulty_one_collapses_cheater_toward_normal() -> None:
    """At difficulty 1.0 the cheater ore mix approaches the normal population."""
    hard = np.mean(
        [
            f
            for s in range(700, 706)
            if (f := _valuable_fraction(make_session_dict("cheater", s, difficulty=1.0)))
            is not None
        ]
    )
    easy = np.mean(
        [
            f
            for s in range(700, 706)
            if (f := _valuable_fraction(make_session_dict("cheater", s, difficulty=0.0)))
            is not None
        ]
    )
    assert hard < easy


@pytest.mark.parametrize("difficulty", [-0.5, 1.5])
def test_difficulty_is_clipped(difficulty: float) -> None:
    """Out-of-range difficulty is clipped rather than crashing."""
    session = make_session_dict("cheater", seed=5, difficulty=difficulty)
    assert len(session["ticks"]) >= config.MIN_EVENTS
