import numpy as np
import pytest

from ratingengine import evaluation as ev


def test_gini_two_policies_hand_worked():
    assert ev.gini([1, 2], [0, 1], [1, 1]) == pytest.approx(0.5)


def test_gini_four_policies_unequal_exposure_hand_worked():
    # x: 0 .2 .4 .8 1, y: 0 0 .25 .25 1, area = .25, Gini = 1 - 2 x .25
    assert ev.gini([1, 2, 3, 4], [0, 10, 0, 30], [1, 1, 2, 1]) == pytest.approx(0.5)


def test_gini_reversed_order_is_negative_and_scale_free():
    pred = np.array([1, 2, 3, 4.0])
    loss = np.array([0, 1, 2, 3.0])
    g = ev.gini(pred, loss)
    assert g > 0
    assert ev.gini(-pred, loss) == pytest.approx(-g)
    assert ev.gini(pred * 1000, loss) == pytest.approx(g)


def test_gini_random_scores_near_zero():
    rng = np.random.default_rng(0)
    loss = rng.gamma(1, 1, 20000)
    assert abs(ev.gini(rng.random(20000), loss)) < 0.03


def test_exposure_quantiles_equal_exposure():
    g = ev.exposure_quantiles(np.arange(10.0), np.ones(10), q=5)
    assert list(g) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_exposure_quantiles_respect_order_not_position():
    g = ev.exposure_quantiles(np.array([4, 3, 2, 1.0]), np.ones(4), q=2)
    assert list(g) == [1, 1, 0, 0]


def test_lift_table_hand_worked():
    t = ev.lift_table(np.array([1, 2, 3, 4.0]), np.array([0, 2, 1, 5.0]), np.ones(4), q=2)
    assert [r["predicted"] for r in t] == [1.5, 3.5]
    assert [r["actual"] for r in t] == [1.0, 3.0]
    assert ev.lift_ratio(t) == pytest.approx(3.0)


def test_double_lift_hand_worked():
    cur = np.array([100, 100, 100, 100.0])
    pro = np.array([50, 80, 120, 150.0])
    loss = np.array([40, 60, 100, 120.0])
    t = ev.double_lift(cur, pro, loss, np.ones(4), q=2)
    assert [r["current"] for r in t] == pytest.approx([1, 1])
    assert [r["proposed"] for r in t] == pytest.approx([0.65, 1.35])
    assert [r["actual"] for r in t] == pytest.approx([0.625, 1.375])
    assert [r["loss_ratio_current"] for r in t] == pytest.approx([0.5, 1.1])
    err = ev.double_lift_error(t)
    assert err["current"] == pytest.approx(0.375)
    assert err["proposed"] == pytest.approx(0.025)


def test_double_lift_rebased_lines_average_to_one():
    rng = np.random.default_rng(2)
    cur = rng.uniform(500, 1500, 1000)
    pro = cur * rng.uniform(0.7, 1.3, 1000)
    e = rng.uniform(0.1, 1, 1000)
    t = ev.double_lift(cur, pro, rng.gamma(1, 600, 1000) * e, e)
    w = np.array([r["exposure"] for r in t])
    for k in ("current", "proposed", "actual"):
        assert np.sum(w * [r[k] for r in t]) / w.sum() == pytest.approx(1.0)


def test_actual_vs_expected_one_way():
    out = ev.actual_vs_expected(
        np.array([0, 0, 1]), ["a", "b"], np.array([10, 20, 5.0]), np.array([15, 15, 10.0]), np.ones(3)
    )
    assert out[0]["ae"] == pytest.approx(1.0)
    assert out[1]["ae"] == pytest.approx(0.5)


def test_actual_vs_expected_two_way():
    out = ev.actual_vs_expected_2way(
        np.array([0, 0, 1, 1]),
        ["r0", "r1"],
        np.array([0, 1, 0, 1]),
        ["c0", "c1"],
        np.array([1, 2, 3, 4.0]),
        np.array([2, 2, 3, 2.0]),
    )
    assert out["ae"] == [[0.5, 1.0], [1.0, 2.0]]


def test_loss_ratio_by_segment():
    out = ev.loss_ratio_by(
        np.array([0, 1, 1]),
        ["x", "y"],
        np.array([70, 30, 30.0]),
        {"current": np.array([100, 50, 50.0]), "proposed": np.array([90, 60, 60.0])},
    )
    assert out[0]["loss_ratio_current"] == pytest.approx(0.7)
    assert out[1]["loss_ratio_proposed"] == pytest.approx(0.5)


def test_spread_ignores_none():
    assert ev.spread([0.6, None, 0.9]) == pytest.approx(0.3)
