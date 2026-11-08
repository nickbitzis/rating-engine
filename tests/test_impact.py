import numpy as np
import pytest

from ratingengine.impact import apply_rate_cap, cap_cost, change_counts, histogram, segment_impact


def test_cap_increases_only():
    out = apply_rate_cap([100, 100], [130, 70], 0.15, None)
    np.testing.assert_allclose(out, [115, 70])


def test_cap_both_sides():
    out = apply_rate_cap([100, 100, 100], [130, 90, 70], 0.2, 0.2)
    np.testing.assert_allclose(out, [120, 90, 80])


def test_cap_cost_hand_worked():
    cur, pro = np.array([100, 100, 100.0]), np.array([130, 90, 70.0])
    capped = apply_rate_cap(cur, pro, 0.2, 0.2)
    c = cap_cost(cur, pro, capped, np.array([1, 1, 0.5]))
    assert c["premium_lost_to_cap"] == pytest.approx(10)
    assert c["premium_kept_by_floor"] == pytest.approx(5)
    assert c["net_cost"] == pytest.approx(5)
    assert c["policies_capped"] == 1 and c["policies_floored"] == 1
    assert c["rate_change_uncapped"] == pytest.approx(0.02)
    assert c["rate_change_capped"] == pytest.approx(0.0)


def test_no_cap_costs_nothing():
    cur, pro = np.array([100, 100.0]), np.array([200, 50.0])
    c = cap_cost(cur, pro, apply_rate_cap(cur, pro, None, None), np.ones(2))
    assert c["net_cost"] == 0 and c["policies_capped"] == 0


def test_histogram_counts_every_policy():
    pct = np.array([-0.5, -0.12, -0.01, 0.0, 0.07, 0.11, 0.9])
    h = histogram(pct)
    assert sum(b["count"] for b in h) == len(pct)
    assert h[0]["lo"] is None and h[-1]["hi"] is None
    zero_bin = next(b for b in h if b["hi"] == 0.0)
    assert zero_bin["count"] == 2  # bins are (lo, hi]: -0.01 and 0.0


def test_change_counts_threshold():
    c = change_counts(np.array([-0.2, -0.1, 0.0, 0.1, 0.25, 0.3]), 0.10)
    assert c["up_more_than"] == 2 and c["down_more_than"] == 1 and c["within"] == 3


def test_segment_impact_hand_worked():
    codes = np.array([0, 0, 1])
    cur = np.array([100, 100, 200.0])
    pro = np.array([120, 100, 150.0])
    capped = apply_rate_cap(cur, pro, 0.15, None)
    out = segment_impact(codes, ["a", "b"], cur, pro, capped, np.ones(3))
    assert out[0]["change"] == pytest.approx(0.10)
    assert out[0]["change_capped"] == pytest.approx(0.075)
    assert out[0]["up_share"] == pytest.approx(0.5)
    assert out[1]["down_share"] == pytest.approx(1.0)


def test_review_impact_is_consistent(review):
    imp = review.impact(cap_up=0.15)
    assert imp["counts"]["policies"] == review.n
    assert sum(b["count"] for b in imp["histogram"]) == review.n
    # After the cap no policy rises more than 15%.
    assert all(b["count"] == 0 for b in imp["histogram_capped"] if b["lo"] is not None and b["lo"] >= 0.15)
    assert imp["counts_capped"]["max_increase"] <= 0.15 + 1e-9
    assert imp["cap"]["rate_change_capped"] < imp["cap"]["rate_change_uncapped"]
    assert {"territory", "driver_age", "credit_band"} <= set(imp["segments"])


def test_looser_cap_costs_less(review):
    tight = review.impact(cap_up=0.10)["cap"]["net_cost"]
    loose = review.impact(cap_up=0.30)["cap"]["net_cost"]
    assert tight > loose > 0
