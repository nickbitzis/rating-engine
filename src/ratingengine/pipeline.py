"""End to end pricing review: data, model fitting, the proposed rating plan, model quality and impact.

build() returns a Review holding the portfolio, the fitted models, both rating plans and a JSON-ready results
dictionary that the API and the CLI serve.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import evaluation as ev
from .factors import TRUTH, TRUTH_BY_NAME, classify
from .glm import deviance as glm_deviance
from .glm import tweedie
from .impact import impact_report
from .modeling import (
    FittedModel,
    backward_select,
    fine_age_spec,
    fit_frequency,
    fit_severity,
    fit_tweedie,
    holdout_deviances,
    level_summary,
    merged_territory_spec,
    severity_rows,
    true_relativities,
    truth_spec,
)
from .plan import PLANS_DIR, RatingPlan, cap_relativities, complement_from_plan, off_balance, plan_levels, smooth
from .synth import Portfolio, generate_portfolio, split_holdout

TARGET_LOSS_RATIO = 0.70
REL_CAP = (0.45, 2.25)
PLAN_FACTOR_ORDER = [
    "age_band",
    "experience",
    "vehicle_age",
    "value_band",
    "territory",
    "mileage",
    "prior_claims",
    "credit_band",
    "coverage",
]
EXCLUDED_FROM_PLAN = {"vehicle_color": "not a permitted rating factor and no signal"}
MAIN_FACTORS = ["age_band", "territory", "prior_claims", "credit_band", "mileage", "coverage"]
CURRENT_VERSION = "2026.01.0"
PROPOSED_VERSION = "2027.01.0"


@dataclass
class Review:
    n: int
    seed: int
    portfolio: Portfolio
    holdout: np.ndarray
    freq: FittedModel
    sev: FittedModel
    tweedie: FittedModel
    current: RatingPlan
    proposed: RatingPlan
    results: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    @property
    def plans(self) -> dict[str, RatingPlan]:
        return {self.current.version: self.current, self.proposed.version: self.proposed}

    def impact(self, cap_up=0.15, cap_down=None, threshold=0.10) -> dict:
        pf = self.portfolio
        cur = self.current.rate_portfolio(pf)
        pro = self.proposed.rate_portfolio(pf)
        segs = {
            "territory": TRUTH_BY_NAME["territory"].levels,
            "driver_age": TRUTH_BY_NAME["age_band"].levels,
            "credit_band": TRUTH_BY_NAME["credit_band"].levels,
        }
        out = impact_report(pf, cur, pro, segs, cap_up=cap_up, cap_down=cap_down, threshold=threshold)
        out["current_version"] = self.current.version
        out["proposed_version"] = self.proposed.version
        return out


def round_floats(x, nd=4):
    if isinstance(x, float):
        return round(x, nd) if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: round_floats(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [round_floats(v, nd) for v in x]
    return x


def _model_row(label: str, m: FittedModel, holdout_dev: float) -> dict:
    r = m.result
    return {
        "model": label,
        "factors": [s.name for s in m.specs],
        "params": r.n_params,
        "deviance": r.deviance,
        "aic": r.aic,
        "holdout_deviance": holdout_dev,
        "iterations": r.iterations,
        "converged": r.converged,
    }


def relativity_table(freq: FittedModel, sev: FittedModel, pf: Portfolio, name: str) -> dict:
    spec = freq.spec(name) or sev.spec(name)
    truth = true_relativities(pf, spec)
    summ = level_summary(pf, spec)
    fr = freq.relativities(name) if freq.spec(name) else None
    sr = sev.relativities(name) if sev.spec(name) else None
    rows = []
    for i, label in enumerate(spec.labels):
        f = fr[i] if fr else {"relativity": 1.0, "lo": 1.0, "hi": 1.0, "log": 0.0, "log_se": 0.0}
        s = sr[i] if sr else {"relativity": 1.0, "lo": 1.0, "hi": 1.0, "log": 0.0, "log_se": 0.0}
        lg = f["log"] + s["log"]
        se = math.sqrt(f["log_se"] ** 2 + s["log_se"] ** 2)
        rows.append(
            {
                "level": label,
                "base": i == spec.base_index,
                "exposure": summ["exposure"][i],
                "claims": summ["claims"][i],
                "freq": f["relativity"],
                "freq_lo": f["lo"],
                "freq_hi": f["hi"],
                "freq_true": truth["freq"][i],
                "sev": s["relativity"],
                "sev_lo": s["lo"],
                "sev_hi": s["hi"],
                "sev_true": truth["sev"][i],
                "pp": math.exp(lg),
                "pp_lo": math.exp(lg - 1.96 * se),
                "pp_hi": math.exp(lg + 1.96 * se),
                "pp_true": truth["freq"][i] * truth["sev"][i],
            }
        )
    return {
        "factor": name,
        "field": spec.field,
        "in_frequency": fr is not None,
        "in_severity": sr is not None,
        "levels": rows,
    }


def build_proposed_plan(
    freq: FittedModel, sev: FittedModel, train: Portfolio, full: Portfolio, current: RatingPlan | None = None
) -> RatingPlan:
    factors, capped, smoothed = [], [], []
    for name in PLAN_FACTOR_ORDER:
        if not (freq.spec(name) or sev.spec(name)):
            continue
        tab = relativity_table(freq, sev, train, name)
        spec = freq.spec(name) or sev.spec(name)
        fitted = [r["pp"] for r in tab["levels"]]
        claims = [r["claims"] for r in tab["levels"]]
        comp = complement_from_plan(current, name, spec.field, train, spec.levels)
        sm, zs = smooth(fitted, claims, spec.base_index, comp)
        cp = cap_relativities(sm, *REL_CAP)
        for lv, a, b, c, z in zip(spec.labels, fitted, sm, cp, zs, strict=True):
            if z < 1.0:
                smoothed.append(
                    {
                        "factor": name,
                        "level": lv,
                        "fitted": round(a, 3),
                        "smoothed": round(b, 3),
                        "complement": round(comp[spec.labels.index(lv)] / comp[spec.base_index], 3),
                        "credibility": round(z, 3),
                    }
                )
            if abs(b - c) > 1e-12:
                capped.append({"factor": name, "level": lv, "before": round(b, 3), "after": round(c, 3)})
        factors.append({"name": name, "input": spec.field, "levels": plan_levels(spec.levels, cp)})
    discounts = []
    if freq.spec("multi_policy") or sev.spec("multi_policy"):
        tab = relativity_table(freq, sev, train, "multi_policy")
        yes = next(r for r in tab["levels"] if r["level"] == "yes")
        discounts.append({"name": "multi_policy", "input": "multi_policy", "rate": round(max(0.0, 1 - yes["pp"]), 3)})
    base_pp = math.exp(freq.result.coef("intercept") + sev.result.coef("intercept"))
    indicated = base_pp / TARGET_LOSS_RATIO
    draft = RatingPlan(PROPOSED_VERSION, "2027-01-01", "proposed", "", indicated, factors, discounts)
    target = float(full["incurred_loss"].sum()) / TARGET_LOSS_RATIO
    rel_product = draft.rate_portfolio(full) / indicated
    base = round(off_balance(indicated, rel_product, full["exposure"], target), 2)
    plan = RatingPlan(
        version=PROPOSED_VERSION,
        effective_date="2027-01-01",
        status="proposed",
        description="GLM indicated plan: frequency x severity relativities, credibility smoothed, capped, "
        "off-balanced to a 70% loss ratio.",
        base_rate=base,
        factors=factors,
        discounts=discounts,
    )
    achieved = float(np.sum(plan.rate_portfolio(full) * full["exposure"]))
    plan.notes = round_floats(
        {
            "target_loss_ratio": TARGET_LOSS_RATIO,
            "base_pure_premium": base_pp,
            "indicated_base_rate": indicated,
            "off_balance_factor": base / indicated,
            "target_premium": target,
            "achieved_premium": achieved,
            "relativity_cap": list(REL_CAP),
            "capped_levels": capped,
            "smoothed_levels": smoothed,
            "excluded_factors": EXCLUDED_FROM_PLAN,
            "fitted_on": {"policies": len(train), "claims": int(train["claim_count"].sum())},
        }
    )
    return plan


def build(n: int = 200_000, seed: int = 2026, plans_dir=PLANS_DIR, with_gbm: bool = True) -> Review:
    t0 = time.perf_counter()
    timings = {}
    pf = generate_portfolio(n, seed)
    ho = split_holdout(n, seed)
    tr, te = pf.subset(~ho), pf.subset(ho)
    tr_cl = severity_rows(tr)
    timings["synth_s"] = time.perf_counter() - t0

    # 1. Every candidate factor with single-year young ages, five-year bands after that, and all ten zones.
    fine = [fine_age_spec()] + [truth_spec(f.name) for f in TRUTH if f.name != "age_band"]
    t = time.perf_counter()
    f_full = fit_frequency(tr, fine)
    s_full = fit_severity(tr_cl, fine)
    # 2. Banding: eight age bands and sparse territories merged (the same banding in both models).
    terr = merged_territory_spec(tr)
    banded = [truth_spec("age_band") if s.name == "age_band" else terr if s.name == "territory" else s for s in fine]
    f_banded = fit_frequency(tr, banded)
    s_banded = fit_severity(tr_cl, banded)
    # 3. Backward elimination by AIC on the banded factors.
    f_sel, f_steps, f_delta = backward_select(lambda s: fit_frequency(tr, s), banded)
    s_sel, s_steps, s_delta = backward_select(lambda s: fit_severity(tr_cl, s), banded)
    freq = fit_frequency(tr, f_sel)
    sev = fit_severity(tr_cl, s_sel)
    union = [s for s in banded if s.name in {x.name for x in freq.specs} | {x.name for x in sev.specs}]
    tw = fit_tweedie(tr, union)
    timings["fit_s"] = time.perf_counter() - t

    def hd(fm, sm):
        return holdout_deviances(fm, sm, te)

    labels = [
        "all candidates, fine age, 10 territories",
        "8 age bands, sparse zones merged",
        "banded, after AIC selection",
    ]
    freq_rows = [
        _model_row(lab, m, hd(m, sev)["frequency_poisson_deviance"])
        for lab, m in zip(labels, [f_full, f_banded, freq], strict=True)
    ]
    sev_rows = [
        _model_row(lab, m, hd(freq, m)["severity_gamma_deviance"])
        for lab, m in zip(labels, [s_full, s_banded, sev], strict=True)
    ]

    # Pure premium on the holdout: GLM frequency x severity, Tweedie GLM, gradient boosting benchmark.
    e_te = te["exposure"]
    loss_te = te["incurred_loss"]
    pp_glm = freq.predict(te) * sev.predict(te)
    pp_tw = tw.predict(te)
    pp_true = te["true_frequency"] * te["true_severity"]
    tw_fam = tweedie(1.6)
    cost_te = loss_te / e_te
    pp_models = {"glm_freq_x_sev": pp_glm, "tweedie_glm": pp_tw}
    t = time.perf_counter()
    if with_gbm:
        from .gbm import fit_gbm

        gbm = fit_gbm(tr, seed)
        pp_models["gradient_boosting"] = gbm.predict(te)
    timings["gbm_s"] = time.perf_counter() - t
    pp_models["true_expected"] = pp_true
    pure_rows = []
    for name, pred in pp_models.items():
        pure_rows.append(
            {
                "model": name,
                "holdout_gini": ev.gini(pred, loss_te, e_te),
                "holdout_tweedie_deviance": glm_deviance(tw_fam, cost_te, pred, e_te),
                "lift_ratio": ev.lift_ratio(ev.lift_table(pred, loss_te, e_te)),
            }
        )

    current = RatingPlan.load(plans_dir / f"plan-{CURRENT_VERSION}.json")
    proposed = build_proposed_plan(freq, sev, tr, pf, current)
    cur_te = current.rate_portfolio(te)
    pro_te = proposed.rate_portfolio(te)
    dl = ev.double_lift(cur_te, pro_te, loss_te, e_te)

    # Actual vs expected on the holdout, expected = GLM pure premium x exposure.
    expected = pp_glm * e_te
    ae = {}
    for f in TRUTH:
        codes = classify(te[f.field], f.levels)
        ae[f.name] = ev.actual_vs_expected(codes, f.labels, loss_te, expected, e_te)
    age_t, terr_t = TRUTH_BY_NAME["age_band"], TRUTH_BY_NAME["territory"]
    ae2 = ev.actual_vs_expected_2way(
        classify(te["driver_age"], age_t.levels),
        age_t.labels,
        classify(te["territory"], terr_t.levels),
        terr_t.labels,
        loss_te,
        expected,
    )

    # Loss ratio by segment on the full portfolio, earned premium under each plan.
    cur_all = current.rate_portfolio(pf) * pf["exposure"]
    pro_all = proposed.rate_portfolio(pf) * pf["exposure"]
    lr = {}
    for name in ["territory", "age_band", "credit_band", "experience", "coverage"]:
        f = TRUTH_BY_NAME[name]
        lr[name] = ev.loss_ratio_by(
            classify(pf[f.field], f.levels), f.labels, pf["incurred_loss"], {"current": cur_all, "proposed": pro_all}
        )
    total_loss = float(pf["incurred_loss"].sum())

    rel_tables = [relativity_table(freq, sev, tr, s.name) for s in union]
    review = Review(n, seed, pf, ho, freq, sev, tw, current, proposed, timings=timings)
    impact = review.impact()
    gate = gate_check(rel_tables, pure_rows)
    results = {
        "meta": {
            "insurer": "Harrow Lane Mutual (fictional)",
            "policies": n,
            "seed": seed,
            "train_policies": len(tr),
            "holdout_policies": len(te),
            "exposure": float(pf["exposure"].sum()),
            "claims": int(pf["claim_count"].sum()),
            "incurred_loss": total_loss,
            "target_loss_ratio": TARGET_LOSS_RATIO,
        },
        "models": {
            "frequency": freq_rows,
            "severity": sev_rows,
            "pure_premium": pure_rows,
            "selection": {
                "frequency": {"steps": f_steps, "aic_increase_if_dropped": f_delta},
                "severity": {"steps": s_steps, "aic_increase_if_dropped": s_delta},
            },
            "territory_banding": terr.levels,
            "tweedie_power": 1.6,
        },
        "relativities": rel_tables,
        "lift": ev.lift_table(pp_glm, loss_te, e_te),
        "double_lift": dl,
        "double_lift_error": ev.double_lift_error(dl),
        "plan_gini": {"current": ev.gini(cur_te, loss_te, e_te), "proposed": ev.gini(pro_te, loss_te, e_te)},
        "actual_vs_expected": ae,
        "actual_vs_expected_2way": ae2,
        "loss_ratios": lr,
        "overall_loss_ratio": {
            "current": total_loss / float(cur_all.sum()),
            "proposed": total_loss / float(pro_all.sum()),
        },
        "plans": {
            v: {"version": p.version, "status": p.status, "effective_date": p.effective_date, "sha256": p.sha256}
            for v, p in review.plans.items()
        },
        "gate": gate,
    }
    timings["total_s"] = time.perf_counter() - t0
    results["meta"]["timings"] = timings
    results["impact"] = impact
    review.results = round_floats(results)
    return review


def gate_check(
    rel_tables: list[dict],
    pure_rows: list[dict],
    tolerance: float = 0.12,
    gini_floor: float = 0.25,
    min_exposure_share: float = 0.03,
) -> dict:
    """Release gate: main-factor frequency relativities within tolerance of truth, holdout Gini above a floor.

    Levels with less than min_exposure_share of exposure are reported but not gated (their intervals are wide).
    """
    failures, checked = [], 0
    worst = 0.0
    for t in rel_tables:
        if t["factor"] not in MAIN_FACTORS or not t["in_frequency"]:
            continue
        total = sum(r["exposure"] for r in t["levels"])
        for r in t["levels"]:
            if r["base"] or r["exposure"] / total < min_exposure_share:
                continue
            checked += 1
            err = abs(r["freq"] / r["freq_true"] - 1)
            worst = max(worst, err)
            if err > tolerance:
                failures.append(f"{t['factor']}={r['level']}: fitted {r['freq']:.3f} vs true {r['freq_true']:.3f}")
    missing = [f for f in MAIN_FACTORS if not any(t["factor"] == f and t["in_frequency"] for t in rel_tables)]
    failures += [f"main factor {f} not selected" for f in missing]
    g = next(r["holdout_gini"] for r in pure_rows if r["model"] == "glm_freq_x_sev")
    if g < gini_floor:
        failures.append(f"holdout Gini {g:.3f} below floor {gini_floor}")
    return {
        "passed": not failures,
        "tolerance": tolerance,
        "gini_floor": gini_floor,
        "levels_checked": checked,
        "worst_relative_error": worst,
        "glm_holdout_gini": g,
        "failures": failures,
    }
