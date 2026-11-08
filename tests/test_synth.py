import numpy as np

from ratingengine.factors import TERRITORY_SHARE
from ratingengine.synth import generate_portfolio, split_holdout


def test_same_seed_same_portfolio():
    a, b = generate_portfolio(2000, 11), generate_portfolio(2000, 11)
    np.testing.assert_array_equal(a["incurred_loss"], b["incurred_loss"])
    np.testing.assert_array_equal(a["driver_age"], b["driver_age"])


def test_different_seed_differs():
    a, b = generate_portfolio(2000, 11), generate_portfolio(2000, 12)
    assert not np.array_equal(a["driver_age"], b["driver_age"])


def test_exposure_is_a_fraction_of_a_year(portfolio):
    e = portfolio["exposure"]
    assert e.min() > 0 and e.max() <= 1.0
    assert 0.6 < np.mean(e == 1.0) < 0.7


def test_years_licensed_consistent_with_age(portfolio):
    assert np.all(portfolio["years_licensed"] <= portfolio["driver_age"] - 16)
    assert np.all(portfolio["years_licensed"] >= 0)


def test_claim_count_matches_true_expected(portfolio):
    expected = np.sum(portfolio["exposure"] * portfolio["true_frequency"])
    assert abs(portfolio["claim_count"].sum() - expected) < 4 * np.sqrt(expected)


def test_losses_only_where_claims(portfolio):
    has = portfolio["claim_count"] > 0
    assert np.all(portfolio["incurred_loss"][~has] == 0)
    assert np.all(portfolio["incurred_loss"][has] > 0)


def test_territory_mix(portfolio):
    share = np.bincount(portfolio["territory"], weights=portfolio["exposure"], minlength=11)[1:]
    np.testing.assert_allclose(share / share.sum(), TERRITORY_SHARE, atol=0.01)


def test_holdout_split_is_seeded_and_about_30_percent():
    a, b = split_holdout(10_000, 5), split_holdout(10_000, 5)
    np.testing.assert_array_equal(a, b)
    assert 0.28 < a.mean() < 0.32


def test_policy_row_is_rateable(portfolio):
    p = portfolio.policy(0)
    assert isinstance(p["multi_policy"], bool)
    assert set(p) >= {"driver_age", "territory", "coverage", "credit_band"}
