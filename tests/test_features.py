"""Tests for the feature extractor (``src.features``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from tests.conftest import extract_vector, make_session_frame


def test_feature_vector_has_contractual_length() -> None:
    """Every session maps to exactly N_FEATURES values."""
    vec = extract_vector(make_session_frame("normal", seed=21))
    assert vec.shape == (config.N_FEATURES,)
    assert len(config.FEATURE_NAMES) == config.N_FEATURES


def test_extract_features_dict_is_keyed_by_feature_names() -> None:
    """When the extractor returns a mapping it uses exactly FEATURE_NAMES."""
    from src import features

    out = features.extract_features(make_session_frame("cheater", seed=21))
    if isinstance(out, dict):
        assert set(out.keys()) == set(config.FEATURE_NAMES)
    elif isinstance(out, pd.Series):
        assert set(config.FEATURE_NAMES).issubset(set(out.index))
    else:
        assert np.asarray(out).ravel().shape == (config.N_FEATURES,)


def test_feature_vector_is_finite() -> None:
    """No NaN or inf in the feature vector, for either population."""
    for label in ("normal", "cheater"):
        vec = extract_vector(make_session_frame(label, seed=22))
        assert np.isfinite(vec).all(), f"non-finite feature in {label} vector"


def test_feature_extraction_is_deterministic() -> None:
    """Same session frame -> identical feature vector across calls."""
    frame = make_session_frame("cheater", seed=23)
    np.testing.assert_array_equal(extract_vector(frame), extract_vector(frame))


def test_feature_vector_dtype_is_float() -> None:
    """Feature vector is real-valued floating point."""
    vec = extract_vector(make_session_frame("normal", seed=24))
    assert np.issubdtype(vec.dtype, np.floating)


def test_cheater_and_normal_features_differ() -> None:
    """Cheater and legitimate sessions produce distinguishable vectors."""
    normal = extract_vector(make_session_frame("normal", seed=25))
    cheat = extract_vector(make_session_frame("cheater", seed=25))
    assert not np.allclose(normal, cheat)


@pytest.mark.parametrize("label", ["normal", "cheater"])
def test_ratio_features_within_unit_interval(label: str) -> None:
    """Features named as ratios stay within [0, 1] (± float tolerance)."""
    vec = extract_vector(make_session_frame(label, seed=26))
    for name in ("valuable_ore_ratio", "non_ore_dig_ratio", "vertical_travel_ratio"):
        idx = config.FEATURE_NAMES.index(name)
        val = vec[idx]
        assert -1e-6 <= val <= 1.0 + 1e-6, f"{name}={val} outside [0,1]"


def _mean_feature(label: str, name: str, seeds: range) -> float:
    idx = config.FEATURE_NAMES.index(name)
    return float(np.mean([extract_vector(make_session_frame(label, s))[idx] for s in seeds]))


def test_cheaters_have_higher_valuable_ore_ratio_on_average() -> None:
    """The ``valuable_ore_ratio`` feature separates cheaters from miners."""
    seeds = range(300, 312)
    assert _mean_feature("cheater", "valuable_ore_ratio", seeds) > _mean_feature(
        "normal", "valuable_ore_ratio", seeds
    )


def test_cheaters_have_higher_path_efficiency_on_average() -> None:
    """Cheaters beeline, so mean ``path_efficiency`` exceeds miners'."""
    seeds = range(400, 412)
    assert _mean_feature("cheater", "path_efficiency", seeds) > _mean_feature(
        "normal", "path_efficiency", seeds
    )


def test_cheaters_have_lower_heading_change_on_average() -> None:
    """Cheaters change heading less often (straight tunnels)."""
    seeds = range(450, 462)
    assert _mean_feature("cheater", "heading_change_mean", seeds) < _mean_feature(
        "normal", "heading_change_mean", seeds
    )
