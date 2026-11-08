"""Gradient boosting benchmark for lift only. It is not used for rating: its output is not a filed table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .synth import COLORS, COVERAGES, Portfolio

FEATURES = [
    "driver_age",
    "years_licensed",
    "vehicle_age",
    "vehicle_value",
    "territory",
    "annual_mileage",
    "prior_claims",
    "credit_band",
    "multi_policy",
    "coverage",
    "vehicle_color",
]


def features(pf: Portfolio) -> np.ndarray:
    cols = []
    for f in FEATURES:
        v = pf[f]
        if f == "coverage":
            v = (v[:, None] == COVERAGES[None, :]).argmax(axis=1)
        elif f == "vehicle_color":
            v = (v[:, None] == COLORS[None, :]).argmax(axis=1)
        cols.append(np.asarray(v, float))
    return np.column_stack(cols)


@dataclass
class GBMModel:
    model: object

    def predict(self, pf: Portfolio) -> np.ndarray:
        return self.model.predict(features(pf))


def fit_gbm(pf: Portfolio, seed: int = 2026) -> GBMModel:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits

    m = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )
    y = pf["incurred_loss"] / pf["exposure"]
    # BLAS threads left spinning by the GLM fits slow the OpenMP trees down about tenfold; pin BLAS to one.
    with threadpool_limits(limits=1, user_api="blas"):
        m.fit(features(pf), y, sample_weight=pf["exposure"])
    return GBMModel(m)
