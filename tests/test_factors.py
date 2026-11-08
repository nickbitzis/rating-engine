import numpy as np
import pytest

from ratingengine.factors import (
    FINE_AGE_LEVELS,
    TRUTH_BY_NAME,
    BandingError,
    categorical_levels,
    classify,
    classify_one,
    merge_sparse_levels,
    numeric_levels,
)

AGE = TRUTH_BY_NAME["age_band"].levels


def test_numeric_band_edges():
    codes = classify(np.array([16, 20, 21, 24, 25, 39, 40, 70, 95]), AGE)
    assert [AGE[c]["label"] for c in codes] == [
        "16-20",
        "16-20",
        "21-24",
        "21-24",
        "25-29",
        "30-39",
        "40-49",
        "70+",
        "70+",
    ]


def test_open_ended_last_band():
    lv = numeric_levels([(0, 2, "0-2"), (3, None, "3+")])
    assert classify_one(250, lv) == 1


def test_below_first_band_raises():
    with pytest.raises(BandingError):
        classify(np.array([15]), AGE)


def test_above_closed_last_band_raises():
    with pytest.raises(BandingError):
        classify(np.array([96]), AGE)


def test_categorical_numbers_and_strings():
    cov = TRUTH_BY_NAME["coverage"].levels
    assert list(classify(np.array(["full", "liability", "standard"]), cov)) == [2, 0, 1]
    terr = categorical_levels([([1, 2], "1-2"), ([3], "3")])
    assert list(classify(np.array([3, 1, 2]), terr)) == [1, 0, 0]


def test_unknown_category_raises():
    with pytest.raises(BandingError):
        classify(np.array(["gold"]), TRUTH_BY_NAME["coverage"].levels)


def test_bool_maps_to_integer_values():
    mp = TRUTH_BY_NAME["multi_policy"].levels
    assert classify_one(True, mp) == 1 and classify_one(False, mp) == 0


def test_fine_age_levels_nest_inside_coarse_bands():
    for lv in FINE_AGE_LEVELS:
        lo_band = classify_one(lv["min"], AGE)
        hi_band = classify_one(lv["max"], AGE)
        assert lo_band == hi_band, lv


def test_merge_sparse_joins_closest_rate_neighbor():
    levels = categorical_levels([([z], str(z)) for z in range(1, 5)])
    merged = merge_sparse_levels(levels, [30, 30, 39, 1], [0.05, 0.06, 0.10, 0.11], 0.05)
    assert [m["label"] for m in merged] == ["1", "2", "3-4"]
    assert merged[2]["values"] == [3, 4]


def test_merge_sparse_picks_neighbor_by_rate_not_position():
    levels = categorical_levels([([z], str(z)) for z in range(1, 4)])
    merged = merge_sparse_levels(levels, [50, 2, 48], [0.05, 0.095, 0.10], 0.05)
    assert [m["values"] for m in merged] == [[1], [2, 3]]


def test_merge_sparse_repeats_until_threshold():
    levels = categorical_levels([([z], str(z)) for z in range(1, 5)])
    merged = merge_sparse_levels(levels, [90, 4, 3, 3], [0.1, 0.2, 0.21, 0.22], 0.08)
    assert all(sum(1 for _ in m["values"]) >= 1 for m in merged)
    assert [m["label"] for m in merged] == ["1", "2-4"]


def test_no_merge_when_every_level_is_large_enough():
    levels = categorical_levels([([z], str(z)) for z in range(1, 4)])
    assert merge_sparse_levels(levels, [30, 30, 40], [1, 2, 3], 0.1) == levels


def test_truth_tables_are_consistent():
    for f in TRUTH_BY_NAME.values():
        assert len(f.freq) == len(f.levels) == len(f.sev)
        assert f.freq[f.base_index] == 1.0 and f.sev[f.base_index] == 1.0
