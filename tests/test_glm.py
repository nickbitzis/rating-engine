import math

import numpy as np
import pytest
from scipy import optimize, stats

from ratingengine.glm import GAMMA, POISSON, GLMError, fit_glm, tweedie


def dummies(codes, k):
    X = np.zeros((len(codes), k))
    X[:, 0] = 1
    for j in range(1, k):
        X[:, j] = codes == j
    return X


def test_poisson_intercept_only_with_offset_is_claims_over_exposure():
    y = np.array([0, 1, 2, 0, 3.0])
    e = np.array([0.5, 1.0, 1.0, 0.25, 0.75])
    r = fit_glm(np.ones((5, 1)), y, POISSON, offset=np.log(e))
    assert math.exp(r.beta[0]) == pytest.approx(y.sum() / e.sum(), rel=1e-10)


def test_poisson_one_factor_closed_form_relativities():
    codes = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    y = np.array([1, 0, 2, 3, 1, 0, 1, 4.0])
    e = np.array([1, 0.5, 1, 1, 1, 0.5, 0.5, 1.0])
    r = fit_glm(dummies(codes, 3), y, POISSON, offset=np.log(e))
    rate = [y[codes == j].sum() / e[codes == j].sum() for j in range(3)]
    assert math.exp(r.beta[0]) == pytest.approx(rate[0], rel=1e-9)
    assert math.exp(r.beta[1]) == pytest.approx(rate[1] / rate[0], rel=1e-9)
    assert math.exp(r.beta[2]) == pytest.approx(rate[2] / rate[0], rel=1e-9)


def test_poisson_saturated_two_way_fits_cell_rates():
    a = np.array([0, 0, 1, 1] * 3)
    b = np.array([0, 1, 0, 1] * 3)
    y = np.array([2, 5, 3, 9, 1, 4, 2, 8, 3, 6, 4, 7.0])
    e = np.linspace(0.5, 1.6, 12)
    X = np.column_stack([np.ones(12), a, b, a * b])
    r = fit_glm(X, y, POISSON, offset=np.log(e))
    mu = r.predict(X, np.log(e))
    for i in (0, 1):
        for j in (0, 1):
            m = (a == i) & (b == j)
            assert mu[m].sum() == pytest.approx(y[m].sum(), rel=1e-8)


def test_exposure_offset_aggregation_invariance():
    # Splitting one policy-year into two half-years must not change the fit.
    codes = np.array([0, 1, 1, 0])
    y = np.array([1, 2, 0, 3.0])
    e = np.array([1.0, 1.0, 0.5, 0.8])
    r1 = fit_glm(dummies(codes, 2), y, POISSON, offset=np.log(e))
    codes2 = np.concatenate([codes, codes])
    y2 = np.concatenate([y / 2, y / 2])
    e2 = np.concatenate([e / 2, e / 2])
    r2 = fit_glm(dummies(codes2, 2), y2, POISSON, offset=np.log(e2))
    np.testing.assert_allclose(r1.beta, r2.beta, rtol=1e-9)


def test_offset_shift_moves_only_the_intercept():
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 3, 400)
    e = rng.uniform(0.2, 1, 400)
    y = rng.poisson(e * np.array([0.1, 0.2, 0.3])[codes]).astype(float)
    r1 = fit_glm(dummies(codes, 3), y, POISSON, offset=np.log(e))
    r2 = fit_glm(dummies(codes, 3), y, POISSON, offset=np.log(e) + math.log(12))
    assert r2.beta[0] == pytest.approx(r1.beta[0] - math.log(12), abs=1e-9)
    np.testing.assert_allclose(r1.beta[1:], r2.beta[1:], atol=1e-9)


def test_exposure_offset_differs_from_ignoring_exposure():
    codes = np.array([0, 0, 1, 1])
    y = np.array([1, 1, 1, 1.0])
    e = np.array([1, 1, 0.25, 0.25])
    with_off = fit_glm(dummies(codes, 2), y, POISSON, offset=np.log(e))
    without = fit_glm(dummies(codes, 2), y, POISSON)
    assert math.exp(with_off.beta[1]) == pytest.approx(4.0, rel=1e-9)
    assert math.exp(without.beta[1]) == pytest.approx(1.0, rel=1e-9)


def test_gamma_intercept_only_is_weighted_mean():
    y = np.array([1000, 2500, 400, 8000.0])
    w = np.array([1, 2, 1, 3.0])
    r = fit_glm(np.ones((4, 1)), y, GAMMA, weights=w)
    assert math.exp(r.beta[0]) == pytest.approx(np.sum(w * y) / w.sum(), rel=1e-9)


def test_gamma_one_factor_closed_form():
    codes = np.array([0, 0, 1, 1, 1])
    y = np.array([100, 300, 500, 700, 900.0])
    w = np.array([1, 1, 2, 1, 1.0])
    r = fit_glm(dummies(codes, 2), y, GAMMA, weights=w)
    m0 = np.average(y[:2], weights=w[:2])
    m1 = np.average(y[2:], weights=w[2:])
    assert math.exp(r.beta[1]) == pytest.approx(m1 / m0, rel=1e-9)


def test_tweedie_one_factor_closed_form():
    codes = np.array([0, 0, 0, 1, 1, 1])
    y = np.array([0, 0, 900, 0, 1500, 600.0])
    w = np.array([1, 0.5, 1, 1, 1, 0.5])
    r = fit_glm(dummies(codes, 2), y, tweedie(1.5), weights=w)
    m0 = np.average(y[:3], weights=w[:3])
    m1 = np.average(y[3:], weights=w[3:])
    assert math.exp(r.beta[0]) == pytest.approx(m0, rel=1e-8)
    assert math.exp(r.beta[1]) == pytest.approx(m1 / m0, rel=1e-8)


def test_noise_free_data_recovers_coefficients():
    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(300), rng.normal(size=300), rng.integers(0, 2, 300)])
    beta = np.array([0.5, -0.3, 0.8])
    y = np.exp(X @ beta)
    for fam in (POISSON, GAMMA, tweedie(1.3)):
        r = fit_glm(X, y, fam)
        np.testing.assert_allclose(r.beta, beta, atol=1e-7)
        assert r.deviance == pytest.approx(0.0, abs=1e-8)


def test_irls_converges_quickly():
    rng = np.random.default_rng(4)
    X = np.column_stack([np.ones(2000), rng.normal(size=2000)])
    y = rng.poisson(np.exp(0.2 + 0.4 * X[:, 1])).astype(float)
    r = fit_glm(X, y, POISSON)
    assert r.converged and r.iterations < 15


def test_irls_matches_direct_likelihood_maximisation():
    rng = np.random.default_rng(5)
    X = np.column_stack([np.ones(500), rng.normal(size=500), rng.uniform(-1, 1, 500)])
    e = rng.uniform(0.1, 1, 500)
    y = rng.poisson(e * np.exp(X @ [-1.0, 0.5, -0.7])).astype(float)
    r = fit_glm(X, y, POISSON, offset=np.log(e))

    def nll(b):
        mu = e * np.exp(X @ b)
        return -(y * np.log(mu) - mu).sum()

    opt = optimize.minimize(nll, np.zeros(3), method="BFGS", options={"gtol": 1e-10})
    np.testing.assert_allclose(r.beta, opt.x, atol=1e-5)


def test_poisson_loglik_and_aic_match_scipy():
    rng = np.random.default_rng(6)
    X = np.column_stack([np.ones(200), rng.integers(0, 2, 200)])
    y = rng.poisson(1.5, 200).astype(float)
    r = fit_glm(X, y, POISSON)
    ll = stats.poisson.logpmf(y, r.predict(X)).sum()
    assert r.llf == pytest.approx(ll, rel=1e-10)
    assert r.aic == pytest.approx(-2 * ll + 4, rel=1e-10)


def test_gamma_loglik_matches_scipy_at_estimated_scale():
    rng = np.random.default_rng(7)
    y = rng.gamma(2.0, 500.0, 300)
    r = fit_glm(np.ones((300, 1)), y, GAMMA)
    mu = r.predict(np.ones((300, 1)))
    shape = 1 / r.scale
    ll = stats.gamma.logpdf(y, a=shape, scale=mu / shape).sum()
    assert r.llf == pytest.approx(ll, rel=1e-9)
    assert r.scale == pytest.approx(0.5, rel=0.2)


def test_poisson_intercept_standard_error_is_one_over_root_claims():
    y = np.array([3, 1, 0, 2, 4.0])
    r = fit_glm(np.ones((5, 1)), y, POISSON)
    assert r.se[0] == pytest.approx(1 / math.sqrt(y.sum()), rel=1e-6)


def test_poisson_deviance_handles_zero_claims():
    y = np.array([0, 0, 1.0])
    mu = np.array([0.5, 0.2, 1.0])
    d = POISSON.unit_deviance(y, mu)
    np.testing.assert_allclose(d, [1.0, 0.4, 0.0])


def test_null_deviance_not_below_model_deviance():
    rng = np.random.default_rng(8)
    codes = rng.integers(0, 3, 500)
    y = rng.poisson(np.array([0.5, 1, 2])[codes]).astype(float)
    r = fit_glm(dummies(codes, 3), y, POISSON)
    assert r.null_deviance > r.deviance


def test_invalid_inputs_raise():
    X = np.ones((3, 1))
    with pytest.raises(GLMError):
        fit_glm(X, np.array([1, -1, 0.0]), POISSON)
    with pytest.raises(GLMError):
        fit_glm(X, np.array([1, 0, 2.0]), GAMMA)
    with pytest.raises(GLMError):
        tweedie(2.5)


def test_singular_design_raises():
    X = np.column_stack([np.ones(4), np.ones(4)])
    with pytest.raises(GLMError):
        fit_glm(X, np.array([1, 2, 1, 0.0]), POISSON)
