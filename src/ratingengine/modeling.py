"""Frequency, severity and pure premium models: design matrices, AIC selection, banding, relativity tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .factors import FINE_AGE_LEVELS, TRUTH, TRUTH_BY_NAME, classify, merge_sparse_levels
from .glm import GAMMA, POISSON, GLMResult, ci_multiplier, deviance, fit_glm, tweedie
from .synth import Portfolio

TWEEDIE_POWER = 1.6
SPARSE_SHARE = 0.025  # territories under 2.5% of exposure are merged into a neighbor


@dataclass(frozen=True)
class FactorSpec:
    name: str
    field: str
    levels: list[dict]
    base: str

    @property
    def labels(self) -> list[str]:
        return [lv["label"] for lv in self.levels]

    @property
    def base_index(self) -> int:
        return self.labels.index(self.base)


def truth_spec(name: str) -> FactorSpec:
    t = TRUTH_BY_NAME[name]
    return FactorSpec(t.name, t.field, t.levels, t.base)


def base_label_for(levels: list[dict], raw_base) -> str:
    code = int(classify(np.array([raw_base]), levels)[0])
    return levels[code]["label"]


def design_matrix(pf: Portfolio, specs: list[FactorSpec]) -> tuple[np.ndarray, list[str]]:
    """Intercept plus one dummy column per non-base level of each factor."""
    n = len(pf)
    cols = [np.ones(n)]
    names = ["intercept"]
    for s in specs:
        codes = classify(pf[s.field], s.levels)
        for i, label in enumerate(s.labels):
            if i == s.base_index:
                continue
            cols.append((codes == i).astype(float))
            names.append(f"{s.name}={label}")
    return np.column_stack(cols), names


@dataclass
class FittedModel:
    kind: str  # frequency, severity, pure_premium
    specs: list[FactorSpec]
    result: GLMResult

    def spec(self, name: str) -> FactorSpec | None:
        return next((s for s in self.specs if s.name == name), None)

    def predict(self, pf: Portfolio) -> np.ndarray:
        """Frequency per car-year, severity per claim, or loss cost per car-year."""
        X, _ = design_matrix(pf, self.specs)
        return self.result.predict(X)

    def relativities(self, name: str, level: float = 0.95) -> list[dict]:
        s = self.spec(name)
        z = ci_multiplier(level)
        out = []
        for i, label in enumerate(s.labels):
            if i == s.base_index:
                out.append({"level": label, "relativity": 1.0, "lo": 1.0, "hi": 1.0, "log": 0.0, "log_se": 0.0})
                continue
            j = self.result.names.index(f"{name}={label}")
            b, se = float(self.result.beta[j]), float(self.result.se[j])
            out.append(
                {
                    "level": label,
                    "relativity": float(np.exp(b)),
                    "lo": float(np.exp(b - z * se)),
                    "hi": float(np.exp(b + z * se)),
                    "log": b,
                    "log_se": se,
                }
            )
        return out


def fit_frequency(pf: Portfolio, specs: list[FactorSpec]) -> FittedModel:
    X, names = design_matrix(pf, specs)
    res = fit_glm(X, pf["claim_count"], POISSON, offset=np.log(pf["exposure"]), names=names)
    return FittedModel("frequency", specs, res)


def severity_rows(pf: Portfolio) -> Portfolio:
    return pf.subset(pf["claim_count"] > 0)


def fit_severity(pf_claims: Portfolio, specs: list[FactorSpec]) -> FittedModel:
    X, names = design_matrix(pf_claims, specs)
    y = pf_claims["incurred_loss"] / pf_claims["claim_count"]
    res = fit_glm(X, y, GAMMA, weights=pf_claims["claim_count"], names=names)
    return FittedModel("severity", specs, res)


def fit_tweedie(pf: Portfolio, specs: list[FactorSpec], power: float = TWEEDIE_POWER) -> FittedModel:
    X, names = design_matrix(pf, specs)
    y = pf["incurred_loss"] / pf["exposure"]
    res = fit_glm(X, y, tweedie(power), weights=pf["exposure"], names=names)
    return FittedModel("pure_premium", specs, res)


def backward_select(fit, specs: list[FactorSpec]) -> tuple[list[FactorSpec], list[dict], dict[str, float]]:
    """Backward elimination by AIC. Drops the factor whose removal lowers AIC most, until none does.

    Returns the kept specs, the elimination steps, and the AIC increase from dropping each kept factor.
    """
    current = list(specs)
    steps = []
    aic = fit(current).result.aic
    while len(current) > 1:
        trials = {s.name: fit([c for c in current if c.name != s.name]).result.aic for s in current}
        name, best = min(trials.items(), key=lambda kv: kv[1])
        if best >= aic:
            delta = {k: round(v - aic, 2) for k, v in trials.items()}
            return current, steps, delta
        steps.append({"dropped": name, "aic_before": round(aic, 2), "aic_after": round(best, 2)})
        current = [c for c in current if c.name != name]
        aic = best
    return current, steps, {}


def merged_territory_spec(pf: Portfolio) -> FactorSpec:
    t = TRUTH_BY_NAME["territory"]
    codes = classify(pf["territory"], t.levels)
    k = len(t.levels)
    expo = np.bincount(codes, weights=pf["exposure"], minlength=k)
    claims = np.bincount(codes, weights=pf["claim_count"], minlength=k)
    rate = claims / np.maximum(expo, 1e-9)
    levels = merge_sparse_levels(t.levels, expo, rate, SPARSE_SHARE)
    return FactorSpec("territory", "territory", levels, base_label_for(levels, 4))


def fine_age_spec() -> FactorSpec:
    return FactorSpec("age_band", "driver_age", FINE_AGE_LEVELS, base_label_for(FINE_AGE_LEVELS, 30))


def true_relativities(pf: Portfolio, spec: FactorSpec) -> dict[str, list[float]]:
    """True frequency and severity relativity of each level of a (possibly re-banded) factor.

    A level that groups several true levels gets the exposure-weighted true frequency effect and the
    expected-claims-weighted true severity effect, both relative to the base level.
    """
    t = next(f for f in TRUTH if f.field == spec.field)
    tcodes = classify(pf[spec.field], t.levels)
    tf = np.asarray(t.freq)[tcodes]
    ts = np.asarray(t.sev)[tcodes]
    e = pf["exposure"]
    claims_w = e * pf["true_frequency"]
    codes = classify(pf[spec.field], spec.levels)
    k = len(spec.levels)
    f = np.bincount(codes, weights=e * tf, minlength=k) / np.maximum(np.bincount(codes, weights=e, minlength=k), 1e-12)
    s = np.bincount(codes, weights=claims_w * ts, minlength=k) / np.maximum(
        np.bincount(codes, weights=claims_w, minlength=k), 1e-12
    )
    b = spec.base_index
    return {"freq": list(f / f[b]), "sev": list(s / s[b])}


def level_summary(pf: Portfolio, spec: FactorSpec) -> dict[str, list[float]]:
    codes = classify(pf[spec.field], spec.levels)
    k = len(spec.levels)
    return {
        "exposure": list(np.bincount(codes, weights=pf["exposure"], minlength=k)),
        "claims": list(np.bincount(codes, weights=pf["claim_count"], minlength=k)),
        "loss": list(np.bincount(codes, weights=pf["incurred_loss"], minlength=k)),
    }


def holdout_deviances(freq: FittedModel, sev: FittedModel, pf: Portfolio) -> dict[str, float]:
    mu_f = freq.predict(pf) * pf["exposure"]
    claims = severity_rows(pf)
    mu_s = sev.predict(claims)
    return {
        "frequency_poisson_deviance": deviance(POISSON, pf["claim_count"], mu_f),
        "severity_gamma_deviance": deviance(
            GAMMA, claims["incurred_loss"] / claims["claim_count"], mu_s, claims["claim_count"]
        ),
    }
