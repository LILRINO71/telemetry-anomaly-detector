"""Bridge between raw sessions and the feature extractor.

The generator (:mod:`src.generate_data`) emits *nested* session records -- one
JSON object per session with a ``ticks`` array. The feature extractor
(:mod:`src.features`) consumes a *flat* per-tick :class:`pandas.DataFrame`
grouped by ``session_id``. This module is the single canonical bridge between
those two representations and is shared by the training and evaluation
pipelines.

A per-tick row carries the columns the extractor reads plus the grouping /
labelling metadata::

    session_id  label  tick  x  y  z  yaw  pitch  broke_block  block_type
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from . import config, generate_data

CHEATER_LABEL = "cheater"

TICK_COLUMNS = [
    "session_id",
    "label",
    "tick",
    "x",
    "y",
    "z",
    "yaw",
    "pitch",
    "broke_block",
    "block_type",
]


def sessions_to_tick_frame(sessions: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Flatten nested session records into one per-tick :class:`pandas.DataFrame`.

    Parameters
    ----------
    sessions:
        Iterable of session dicts as produced by
        :func:`src.generate_data.generate_dataset` or read back via
        :func:`src.generate_data.read_jsonl`.

    Returns
    -------
    pandas.DataFrame
        One row per tick with :data:`TICK_COLUMNS`. Empty (but correctly typed)
        when no ticks are present.
    """
    rows: list[dict[str, Any]] = []
    for session in sessions:
        session_id = session.get("session_id")
        label = session.get("label", "normal")
        for tick in session.get("ticks", []):
            rows.append(
                {
                    "session_id": session_id,
                    "label": label,
                    "tick": tick.get("t"),
                    "x": tick.get("x"),
                    "y": tick.get("y"),
                    "z": tick.get("z"),
                    "yaw": tick.get("yaw"),
                    "pitch": tick.get("pitch"),
                    "broke_block": bool(tick.get("dig", False)),
                    "block_type": tick.get("block_type"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=TICK_COLUMNS)
    return pd.DataFrame(rows, columns=TICK_COLUMNS)


def labels_from_frame(tick_frame: pd.DataFrame) -> pd.Series:
    """Return one integer label per session (``1`` == cheater), indexed by id.

    The label is taken from each session's first tick row. Order follows first
    appearance so it matches :func:`src.features.features_for_sessions`.
    """
    first = tick_frame.groupby("session_id", sort=False)["label"].first()
    return (first.astype("string").str.lower() == CHEATER_LABEL).astype("int64")


def build_labeled_dataset(
    n_normal: int,
    n_cheater: int,
    seed: int = config.DEFAULT_SEED,
    difficulty: float = 0.88,
) -> tuple[pd.DataFrame, pd.Series]:
    """Synthesize a labelled dataset in memory.

    Returns a ``(tick_frame, labels)`` pair where ``labels`` is a
    :class:`pandas.Series` of ints (1 == cheater) indexed by ``session_id`` and
    aligned to the sessions present in ``tick_frame``.
    """
    sessions = generate_data.generate_dataset(
        n_normal=n_normal,
        n_cheater=n_cheater,
        difficulty=difficulty,
        seed=seed,
    )
    frame = sessions_to_tick_frame(sessions)
    labels = labels_from_frame(frame)
    return frame, labels


def load_labeled_dataset(path: Path | str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a JSONL sessions file into a ``(tick_frame, labels)`` pair."""
    sessions = generate_data.read_jsonl(Path(path))
    frame = sessions_to_tick_frame(sessions)
    labels = labels_from_frame(frame)
    return frame, labels
