import numpy as np
import pytest

from ratingengine.evaluation import gini
from ratingengine.factors import TRUTH_BY_NAME, categorical_levels
from ratingengine.gbm import fit_gbm
from ratingengine.glm import POISSON, fit_glm
from ratingengine.modeling import (
    FactorSpec,
    design_matrix,
    merged_territory_spec,
    true_relativities,
    truth_spec,
)
from ratingengine.synth import Portfolio, generate_portfolio


def test_relativity_recovery_on_clean_two_factor_case():
    rng = np.random.default_rng(42)
    n = 150_000
    a = rng.integers(0, 3, n)
    b = rng.integers(0, 2, n)
    e = rng.uniform(0.2, 1.0, n)
    true_a, true_b = np.array([1.0, 1.5, 0.7]), np.array([1.0, 1.3])
    y = rng.poisson(e * 0.2 * true_a[a] * true_b[b]).astype(float)
    X = np.column_stack([np.ones(n), a == 1, a == 2, b == 1]).astype(float)
    r = fit_glm(X, y, POISSON, offset=np.log(e))
    np.testing.assert_allclose(np.exp(r.beta), [0.2, 1.5, 0.7, 1.3], rtol=0.03)
    lo = np.exp(r.beta - 1.96 * r.se)
    hi = np.exp(r.beta + 1.96 * r.se)
    assert np.all((lo < [0.2, 1.5, 0.7, 1.3]) | np.isclose(lo, [0.2, 1.5, 0.7, 1.3], rtol=0.02))
    assert np.all(hi > [0.2, 1.5, 0.7, 1.3] * np.array([0.98, 0.98, 0.98, 0.98]))


def test_design_matrix_drops_the_base_level(portfolio):
    spec = truth_spec("coverage")
    X, names = design_matrix(portfolio, [spec])
    assert names == ["intercept", "coverage=liability", "coverage=full"]
    std = portfolio["coverage"] == "standard"
    assert np.all(X[std, 1:] == 0)
    assert np.all(X[:, 1:].sum(axis=1) <= 1)


def test_noise_factor_dropped_by_aic(review):
    sel = review.results["models"]["selection"]
    assert "vehicle_color" in [s["dropped"] for s in sel["frequency"]["steps"]]
    # At this test size AIC keeps vehicle_color in severity by chance; the 200,000 policy run drops it.
    assert {"credit_band", "mileage"} <= {s["dropped"] for s in sel["severity"]["steps"]}
    for step in sel["frequency"]["steps"]:
        assert step["aic_after"] < step["aic_before"]


def test_kept_factors_cost_aic_if_dropped(review):
    delta = review.results["models"]["selection"]["frequency"]["aic_increase_if_dropped"]
    assert delta and all(v > 0 for v in delta.values())


def test_severity_model_drops_frequency_only_factors(review):
    sev = {s.name for s in review.sev.specs}
    assert {"vehicle_age", "value_band"} <= sev
    assert "credit_band" not in sev and "mileage" not in sev


def test_sparse_territories_merged(review):
    labels = [lv["label"] for lv in review.results["models"]["territory_banding"]]
    assert "9-10" in labels
    assert len(labels) < 10


def test_merged_territory_base_is_zone_4(portfolio):
    spec = merged_territory_spec(portfolio)
    assert 4 in spec.levels[spec.base_index]["values"]


def test_true_relativities_unmerged_equal_truth(portfolio):
    t = TRUTH_BY_NAME["credit_band"]
    tr = true_relativities(portfolio, truth_spec("credit_band"))
    np.testing.assert_allclose(tr["freq"], t.freq, rtol=1e-12)


def test_true_relativity_of_merged_level_is_between_its_parts(portfolio):
    spec = FactorSpec(
        "territory", "territory", categorical_levels([([1, 2, 3], "1-3")] + [([z], str(z)) for z in range(4, 11)]), "4"
    )
    tr = true_relativities(portfolio, spec)
    assert 0.80 < tr["freq"][0] < 0.95


def test_fitted_intervals_cover_truth_for_most_levels(review):
    inside = total = 0
    for t in review.results["relativities"]:
        if not t["in_frequency"]:
            continue
        for lv in t["levels"]:
            if lv["base"]:
                continue
            total += 1
            inside += lv["freq_lo"] <= lv["freq_true"] <= lv["freq_hi"]
    assert inside / total >= 0.8


def test_banded_model_has_lower_aic_than_fine_full_model(review):
    rows = review.results["models"]["frequency"]
    assert rows[-1]["aic"] < rows[0]["aic"]
    assert rows[-1]["params"] < rows[0]["params"]
    assert all(r["converged"] for r in rows)


def test_pure_premium_relativity_is_frequency_times_severity(review):
    t = next(t for t in review.results["relativities"] if t["factor"] == "vehicle_age")
    for lv in t["levels"]:
        assert lv["pp"] == pytest.approx(lv["freq"] * lv["sev"], rel=1e-3)


def test_tweedie_and_glm_have_similar_lift(review):
    rows = {r["model"]: r for r in review.results["models"]["pure_premium"]}
    assert rows["glm_freq_x_sev"]["holdout_gini"] > 0.2
    assert abs(rows["tweedie_glm"]["holdout_gini"] - rows["glm_freq_x_sev"]["holdout_gini"]) < 0.05
    assert review.tweedie.result.converged


def test_gbm_benchmark_has_lift():
    pf = generate_portfolio(12_000, 9)
    tr = Portfolio({k: v[:8000] for k, v in pf.cols.items()}, 9)
    te = Portfolio({k: v[8000:] for k, v in pf.cols.items()}, 9)
    m = fit_gbm(tr, 9)
    assert gini(m.predict(te), te["incurred_loss"], te["exposure"]) > 0.1


def test_lift_chart_is_monotone_in_prediction(review):
    pred = [r["predicted"] for r in review.results["lift"]]
    assert pred == sorted(pred)
    assert review.results["lift"][-1]["actual"] > 3 * review.results["lift"][0]["actual"]
