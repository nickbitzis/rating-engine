import json
import math

import numpy as np
import pytest

from ratingengine.pipeline import REL_CAP, TARGET_LOSS_RATIO
from ratingengine.plan import (
    PLANS_DIR,
    PlanError,
    RatingPlan,
    cap_relativities,
    complement_from_plan,
    credibility,
    load_plans,
    off_balance,
    smooth,
)

POLICY = {
    "driver_age": 34,
    "years_licensed": 15,
    "vehicle_age": 5,
    "vehicle_value": 24000,
    "territory": 7,
    "annual_mileage": 11000,
    "prior_claims": 0,
    "credit_band": 2,
    "multi_policy": True,
    "coverage": "full",
}


def product_of_steps(out):
    p = out["base_rate"]
    for s in out["steps"][1:]:
        p *= s["multiplier"]
    return p


def test_breakdown_multiplies_to_premium(review):
    for plan in review.plans.values():
        for i in range(0, 3000, 97):
            out = plan.rate(review.portfolio.policy(i))
            assert round(product_of_steps(out), 2) == out["premium"]


def test_vectorized_rating_matches_single_policy(review):
    for plan in review.plans.values():
        vec = plan.rate_portfolio(review.portfolio)
        for i in range(0, 30000, 1499):
            assert plan.rate(review.portfolio.policy(i))["premium"] == pytest.approx(vec[i], abs=0.006)


def test_breakdown_lists_every_factor_and_discount(review):
    out = review.proposed.rate(POLICY)
    names = [s["factor"] for s in out["steps"] if s["factor"]]
    assert names == [f["name"] for f in review.proposed.factors] + ["multi_policy"]
    assert out["plan_version"] == review.proposed.version


def test_discount_applies_only_when_flag_set(review):
    plan = review.proposed
    rate = plan.discounts[0]["rate"]
    with_d = plan.rate(POLICY)["premium"]
    without = plan.rate({**POLICY, "multi_policy": False})["premium"]
    assert with_d == pytest.approx(without * (1 - rate), abs=0.01)


def test_missing_input_raises(review):
    p = dict(POLICY)
    del p["territory"]
    with pytest.raises(PlanError, match="territory"):
        review.current.rate(p)


def test_out_of_band_input_raises(review):
    with pytest.raises(PlanError):
        review.current.rate({**POLICY, "driver_age": 15})


def test_hash_is_stable_through_save_and_load(review, tmp_path):
    path = tmp_path / "p.json"
    review.proposed.save(path)
    assert RatingPlan.load(path).sha256 == review.proposed.sha256


def test_hash_changes_when_a_relativity_changes(review):
    d = review.proposed.to_dict()
    d.pop("sha256")
    d["factors"][0]["levels"][0]["relativity"] += 0.001
    assert RatingPlan.from_dict(d).sha256 != review.proposed.sha256


def test_tampered_plan_file_is_rejected(tmp_path):
    d = json.loads((PLANS_DIR / "plan-2026.01.0.json").read_text())
    d["base_rate"] += 1
    with pytest.raises(PlanError, match="sha256"):
        RatingPlan.from_dict(d)


def test_committed_plans_load_with_versions():
    plans = load_plans()
    assert [p.version for p in plans] == ["2026.01.0"]
    assert plans[0].status == "in_force"


def test_credibility_square_root_rule():
    assert credibility(1082) == 1.0
    assert credibility(5000) == 1.0
    assert credibility(270.5) == pytest.approx(0.5)
    assert credibility(0) == 0.0


def test_smoothing_toward_current_plan_complement():
    rels = [0.8, 1.0, 1.6]
    out, z = smooth(rels, [5000, 5000, 0], base_index=1, complement=[0.9, 1.0, 1.3])
    assert out[1] == 1.0 and z[1] == 1.0
    assert out[0] == pytest.approx(0.8)  # fully credible: fitted value
    assert out[2] == pytest.approx(1.3)  # no credibility: the complement


def test_smoothing_rebases_complement_to_new_base():
    out, _ = smooth([1.0, 1.4], [5000, 270.5], base_index=0, complement=[2.0, 2.0])
    assert out[1] == pytest.approx(0.5 * 1.4 + 0.5 * 1.0)


def test_complement_for_factor_missing_from_plan_is_one(review):
    lv = review.proposed.factor("credit_band")["levels"]
    assert complement_from_plan(None, "credit_band", "credit_band", review.portfolio, lv) == [1.0] * 5


def test_complement_is_exposure_weighted_current_relativity(review):
    levels = review.proposed.factor("territory")["levels"]
    comp = complement_from_plan(review.current, "territory", "territory", review.portfolio, levels)
    cur = {lv["values"][0]: lv["relativity"] for lv in review.current.factor("territory")["levels"]}
    single = next(i for i, lv in enumerate(levels) if lv["values"] == [5])
    assert comp[single] == pytest.approx(cur[5])


def test_cap_relativities():
    assert cap_relativities([0.2, 1.0, 3.0], 0.45, 2.25) == [0.45, 1.0, 2.25]


def test_off_balance_hand_worked():
    base = off_balance(100.0, np.array([1.0, 2.0]), np.array([1.0, 0.5]), 300.0)
    assert base == pytest.approx(150.0)


def test_proposed_plan_hits_target_premium(review):
    pf = review.portfolio
    total = float(np.sum(review.proposed.rate_portfolio(pf) * pf["exposure"]))
    target = pf["incurred_loss"].sum() / TARGET_LOSS_RATIO
    assert total == pytest.approx(target, rel=2e-4)
    assert review.results["overall_loss_ratio"]["proposed"] == pytest.approx(TARGET_LOSS_RATIO, abs=1e-3)


def test_proposed_relativities_within_caps(review):
    lo, hi = REL_CAP
    for f in review.proposed.factors:
        for lv in f["levels"]:
            assert lo <= lv["relativity"] <= hi
    for c in review.proposed.notes["capped_levels"]:
        assert c["after"] in (lo, hi)
        assert c["before"] < lo or c["before"] > hi


def test_off_balance_factor_recorded(review):
    n = review.proposed.notes
    assert n["off_balance_factor"] == pytest.approx(review.proposed.base_rate / n["indicated_base_rate"], rel=1e-3)
    assert math.isclose(n["target_loss_ratio"], TARGET_LOSS_RATIO)


def test_proposed_plan_excludes_noise_factor(review):
    assert "vehicle_color" not in [f["name"] for f in review.proposed.factors]
    assert "credit_band" in [f["name"] for f in review.proposed.factors]


def test_proposed_plan_beats_current_on_double_lift(review):
    err = review.results["double_lift_error"]
    assert err["proposed"] < err["current"]
    assert review.results["plan_gini"]["proposed"] >= review.results["plan_gini"]["current"] - 0.01
