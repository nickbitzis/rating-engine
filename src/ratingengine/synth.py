"""Seeded synthetic personal auto portfolio for Harrow Lane Mutual, a made-up insurer.

Each row is one policy-year. Claim counts are Poisson with the exposure as a multiplier and a frequency that
is the product of the true relativities in factors.TRUTH. Each claim cost is gamma with a mean that is the
product of the true severity relativities. Because the truth is known, fitted relativities can be checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .factors import BASE_FREQUENCY, BASE_SEVERITY, GAMMA_SHAPE, TERRITORY_SHARE, TRUTH, classify

COLORS = np.array(["black", "blue", "gray", "red", "white"])
COVERAGES = np.array(["liability", "standard", "full"])
RAW_FIELDS = [
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


@dataclass
class Portfolio:
    """Column arrays, one entry per policy-year."""

    cols: dict[str, np.ndarray]
    seed: int

    def __len__(self) -> int:
        return len(self.cols["exposure"])

    def __getitem__(self, key: str) -> np.ndarray:
        return self.cols[key]

    def subset(self, mask: np.ndarray) -> Portfolio:
        return Portfolio({k: v[mask] for k, v in self.cols.items()}, self.seed)

    def policy(self, i: int) -> dict:
        out = {}
        for f in RAW_FIELDS:
            v = self.cols[f][i]
            out[f] = v.item() if hasattr(v, "item") else v
        out["multi_policy"] = bool(out["multi_policy"])
        return out


def _banded(rng, n, probs, bands):
    """Draw a band per row, then a uniform integer inside it."""
    idx = rng.choice(len(bands), size=n, p=np.asarray(probs) / np.sum(probs))
    lo = np.array([b[0] for b in bands])[idx]
    hi = np.array([b[1] for b in bands])[idx]
    return rng.integers(lo, hi + 1)


def generate_portfolio(n: int = 200_000, seed: int = 2026) -> Portfolio:
    rng = np.random.default_rng(seed)
    age = _banded(
        rng,
        n,
        [0.04, 0.06, 0.09, 0.19, 0.19, 0.18, 0.15, 0.10],
        [(16, 20), (21, 24), (25, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 90)],
    )
    # Most people license at 16 to 19; about 12% license later in life.
    start = 16 + np.floor(rng.exponential(1.5, n)).astype(int)
    late = rng.random(n) < 0.12
    start = np.where(late, 16 + rng.integers(4, 30, n), start)
    years_licensed = np.clip(age - start, 0, None)

    vehicle_age = _banded(rng, n, [0.30, 0.30, 0.27, 0.13], [(0, 3), (4, 7), (8, 12), (13, 22)])
    value = 42_000 * np.exp(-0.075 * vehicle_age) * rng.lognormal(0.0, 0.45, n)
    vehicle_value = np.round(np.clip(value, 1_500, 180_000), -2)
    territory = rng.choice(np.arange(1, 11), size=n, p=TERRITORY_SHARE)
    mileage = _banded(
        rng, n, [0.22, 0.38, 0.28, 0.12], [(2_000, 7_499), (7_500, 11_999), (12_000, 17_999), (18_000, 30_000)]
    )
    prior = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.82, 0.14, 0.03, 0.007, 0.003])
    credit = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.15, 0.22, 0.30, 0.20, 0.13])
    multi = (rng.random(n) < 0.42).astype(np.int64)
    coverage = rng.choice(COVERAGES, size=n, p=[0.25, 0.45, 0.30])
    color = rng.choice(COLORS, size=n)
    exposure = np.where(rng.random(n) < 0.65, 1.0, np.round(rng.uniform(0.05, 1.0, n), 3))

    cols: dict[str, np.ndarray] = {
        "policy_id": np.array([f"HLM-{i:06d}" for i in range(n)]),
        "exposure": exposure,
        "driver_age": age,
        "years_licensed": years_licensed,
        "vehicle_age": vehicle_age,
        "vehicle_value": vehicle_value,
        "territory": territory,
        "annual_mileage": mileage,
        "prior_claims": prior,
        "credit_band": credit,
        "multi_policy": multi,
        "coverage": coverage,
        "vehicle_color": color,
    }
    freq = np.full(n, BASE_FREQUENCY)
    sev = np.full(n, BASE_SEVERITY)
    for f in TRUTH:
        codes = classify(cols[f.field], f.levels)
        freq *= np.asarray(f.freq)[codes]
        sev *= np.asarray(f.sev)[codes]
    claims = rng.poisson(exposure * freq)
    # The sum of k gamma(shape, scale) draws is gamma(k * shape, scale).
    scale = sev / GAMMA_SHAPE
    loss = np.zeros(n)
    has = claims > 0
    loss[has] = rng.gamma(claims[has] * GAMMA_SHAPE, scale[has])
    cols["claim_count"] = claims
    cols["incurred_loss"] = np.round(loss, 2)
    cols["true_frequency"] = freq
    cols["true_severity"] = sev
    return Portfolio(cols, seed)


def split_holdout(n: int, seed: int, holdout_share: float = 0.3) -> np.ndarray:
    """Boolean mask, True for holdout rows. Seeded separately from the data so it is stable."""
    rng = np.random.default_rng(seed + 7)
    return rng.random(n) < holdout_share
