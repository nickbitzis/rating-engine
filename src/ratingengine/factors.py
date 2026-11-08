"""Rating factors: banding definitions, the true relativities behind the synthetic data, and level merging.

A banding is a list of levels. A numeric level is {"label", "min", "max"} (max None means open ended) and a
categorical level is {"label", "values"}. The same structure is used by the models and by the rating plans,
so a plan can be read without the code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BandingError(ValueError):
    pass


def numeric_levels(edges: list[tuple[int | float, int | float | None, str]]) -> list[dict]:
    return [{"label": label, "min": lo, "max": hi} for lo, hi, label in edges]


def categorical_levels(groups: list[tuple[list, str]]) -> list[dict]:
    return [{"label": label, "values": list(values)} for values, label in groups]


def is_numeric(levels: list[dict]) -> bool:
    return "min" in levels[0]


def classify(values, levels: list[dict]) -> np.ndarray:
    """Map raw values to level codes (0..k-1). Raises BandingError for values no level covers."""
    arr = np.asarray(values)
    if is_numeric(levels):
        mins = np.array([lv["min"] for lv in levels], dtype=float)
        if np.any(np.diff(mins) <= 0):
            raise BandingError("numeric levels must have increasing min values")
        x = arr.astype(float)
        if np.any(x < mins[0]):
            raise BandingError(f"value below the first band ({mins[0]})")
        last_max = levels[-1]["max"]
        if last_max is not None and np.any(x > last_max):
            raise BandingError(f"value above the last band ({last_max})")
        return np.searchsorted(mins, x, side="right") - 1
    lookup = {}
    for i, lv in enumerate(levels):
        for v in lv["values"]:
            lookup[_key(v)] = i
    uniq, inverse = np.unique(arr.ravel(), return_inverse=True)
    codes = np.empty(len(uniq), dtype=np.int64)
    for j, v in enumerate(uniq):
        k = _key(v.item() if hasattr(v, "item") else v)
        if k not in lookup:
            raise BandingError(f"no level for value {v!r}")
        codes[j] = lookup[k]
    return codes[inverse].reshape(arr.shape)


def classify_one(value, levels: list[dict]) -> int:
    return int(classify(np.array([value]), levels)[0])


def _key(v):
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (int, float, np.integer, np.floating)) and float(v).is_integer():
        return int(v)
    return str(v)


@dataclass(frozen=True)
class FactorTruth:
    """A rating factor with the multiplicative effects used to generate the data."""

    name: str
    field: str
    levels: list[dict]
    base: str
    freq: list[float]
    sev: list[float]
    ordered: bool = True

    @property
    def labels(self) -> list[str]:
        return [lv["label"] for lv in self.levels]

    @property
    def base_index(self) -> int:
        return self.labels.index(self.base)


TERRITORY_SHARE = [0.020, 0.090, 0.120, 0.180, 0.160, 0.130, 0.110, 0.090, 0.088, 0.012]

TRUTH: list[FactorTruth] = [
    FactorTruth(
        "age_band",
        "driver_age",
        numeric_levels(
            [
                (16, 20, "16-20"),
                (21, 24, "21-24"),
                (25, 29, "25-29"),
                (30, 39, "30-39"),
                (40, 49, "40-49"),
                (50, 59, "50-59"),
                (60, 69, "60-69"),
                (70, 95, "70+"),
            ]
        ),
        "30-39",
        [2.10, 1.55, 1.20, 1.00, 0.92, 0.88, 0.95, 1.18],
        [1.10, 1.06, 1.02, 1.00, 1.00, 1.00, 1.00, 1.04],
    ),
    FactorTruth(
        "experience",
        "years_licensed",
        numeric_levels([(0, 2, "0-2"), (3, 5, "3-5"), (6, None, "6+")]),
        "6+",
        [1.35, 1.10, 1.00],
        [1.00, 1.00, 1.00],
    ),
    FactorTruth(
        "vehicle_age",
        "vehicle_age",
        numeric_levels([(0, 3, "0-3"), (4, 7, "4-7"), (8, 12, "8-12"), (13, None, "13+")]),
        "0-3",
        [1.00, 1.04, 1.08, 1.12],
        [1.00, 0.90, 0.80, 0.72],
    ),
    FactorTruth(
        "value_band",
        "vehicle_value",
        numeric_levels([(0, 14999, "<15k"), (15000, 29999, "15-30k"), (30000, 49999, "30-50k"), (50000, None, "50k+")]),
        "15-30k",
        [1.00, 1.00, 1.00, 1.00],
        [0.85, 1.00, 1.20, 1.45],
    ),
    FactorTruth(
        "territory",
        "territory",
        categorical_levels([([z], str(z)) for z in range(1, 11)]),
        "4",
        [0.80, 0.88, 0.95, 1.00, 1.08, 1.15, 1.25, 1.35, 1.50, 1.70],
        [0.92, 0.94, 0.97, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12],
    ),
    FactorTruth(
        "mileage",
        "annual_mileage",
        numeric_levels([(0, 7499, "<7.5k"), (7500, 11999, "7.5-12k"), (12000, 17999, "12-18k"), (18000, None, "18k+")]),
        "7.5-12k",
        [0.85, 1.00, 1.15, 1.32],
        [1.00, 1.00, 1.00, 1.00],
    ),
    FactorTruth(
        "prior_claims",
        "prior_claims",
        numeric_levels([(0, 0, "0"), (1, 1, "1"), (2, None, "2+")]),
        "0",
        [1.00, 1.30, 1.65],
        [1.00, 1.00, 1.00],
    ),
    FactorTruth(
        "credit_band",
        "credit_band",
        categorical_levels([([b], str(b)) for b in range(1, 6)]),
        "3",
        [0.80, 0.90, 1.00, 1.15, 1.35],
        [1.00, 1.00, 1.00, 1.00, 1.00],
    ),
    FactorTruth(
        "multi_policy",
        "multi_policy",
        categorical_levels([([0], "no"), ([1], "yes")]),
        "no",
        [1.00, 0.90],
        [1.00, 1.00],
        ordered=False,
    ),
    FactorTruth(
        "coverage",
        "coverage",
        categorical_levels([(["liability"], "liability"), (["standard"], "standard"), (["full"], "full")]),
        "standard",
        [0.70, 1.00, 1.20],
        [0.95, 1.00, 1.08],
        ordered=False,
    ),
    FactorTruth(
        "vehicle_color",
        "vehicle_color",
        categorical_levels([([c], c) for c in ["black", "blue", "gray", "red", "white"]]),
        "white",
        [1.0] * 5,
        [1.0] * 5,
        ordered=False,
    ),
]

TRUTH_BY_NAME = {f.name: f for f in TRUTH}
BASE_FREQUENCY = 0.072  # claims per car-year at every base level
BASE_SEVERITY = 5200.0  # mean cost per claim at every base level
GAMMA_SHAPE = 1.6  # severity coefficient of variation is 1/sqrt(1.6), about 0.79

# Single-year ages up to 24 then five-year bands: used to show that the coarse age bands lose nothing.
FINE_AGE_LEVELS = numeric_levels(
    [(a, a, str(a)) for a in range(16, 25)]
    + [(lo, lo + 4, f"{lo}-{lo + 4}") for lo in range(25, 80, 5)]
    + [(80, 95, "80+")]
)


def merge_sparse_levels(levels: list[dict], exposure_by_level, rate_by_level, min_share: float) -> list[dict]:
    """Merge categorical levels with less than min_share of exposure into an adjacent level.

    The sparse level joins the neighbor (previous or next in the list) whose one-way rate is closest. Repeats
    until every level meets the threshold. Returns a new categorical banding.
    """
    groups = [list(lv["values"]) for lv in levels]
    expo = [float(x) for x in exposure_by_level]
    rate = [float(x) for x in rate_by_level]
    labels_src = [[lv["label"]] for lv in levels]
    while len(groups) > 1:
        total = sum(expo)
        shares = [e / total for e in expo]
        i = int(np.argmin(shares))
        if shares[i] >= min_share:
            break
        cands = [j for j in (i - 1, i + 1) if 0 <= j < len(groups)]
        j = min(cands, key=lambda c: abs(rate[c] - rate[i]))
        a, b = sorted((i, j))
        new_rate = (rate[a] * expo[a] + rate[b] * expo[b]) / (expo[a] + expo[b]) if expo[a] + expo[b] else 0.0
        groups[a] = groups[a] + groups[b]
        labels_src[a] = labels_src[a] + labels_src[b]
        expo[a] += expo[b]
        rate[a] = new_rate
        del groups[b], expo[b], rate[b], labels_src[b]
    out = []
    for g, src in zip(groups, labels_src, strict=True):
        label = src[0] if len(src) == 1 else f"{src[0]}-{src[-1]}"
        out.append({"label": label, "values": g})
    return out
