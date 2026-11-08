"""Rating plans: a base rate, a relativity table per factor, and discounts. Versioned and hashed.

premium = base rate x relativity(factor 1) x ... x relativity(factor k) x (1 - discount 1) x ...

A plan is plain JSON, so underwriting and regulatory reviewers can read it without the code.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .factors import BandingError, classify, classify_one, is_numeric
from .synth import Portfolio

PLANS_DIR = Path(__file__).resolve().parents[2] / "plans"
FULL_CREDIBILITY_CLAIMS = 1082  # limited fluctuation standard: within 5% of the true mean 90% of the time


class PlanError(ValueError):
    pass


@dataclass
class RatingPlan:
    version: str
    effective_date: str
    status: str
    description: str
    base_rate: float
    factors: list[dict]
    discounts: list[dict] = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    @property
    def content(self) -> dict:
        return {"base_rate": self.base_rate, "factors": self.factors, "discounts": self.discounts}

    @property
    def sha256(self) -> str:
        blob = json.dumps(self.content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "effective_date": self.effective_date,
            "status": self.status,
            "description": self.description,
            "sha256": self.sha256,
            "base_rate": self.base_rate,
            "factors": copy.deepcopy(self.factors),
            "discounts": copy.deepcopy(self.discounts),
            "notes": copy.deepcopy(self.notes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> RatingPlan:
        plan = cls(
            version=d["version"],
            effective_date=d["effective_date"],
            status=d.get("status", "in_force"),
            description=d.get("description", ""),
            base_rate=float(d["base_rate"]),
            factors=d["factors"],
            discounts=d.get("discounts", []),
            notes=d.get("notes", {}),
        )
        if d.get("sha256") and d["sha256"] != plan.sha256:
            raise PlanError(f"plan {plan.version} content does not match its recorded sha256")
        return plan

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> RatingPlan:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def factor(self, name: str) -> dict:
        for f in self.factors:
            if f["name"] == name:
                return f
        raise KeyError(name)

    def inputs(self) -> list[str]:
        return [f["input"] for f in self.factors] + [d["input"] for d in self.discounts]

    def rate(self, policy: dict) -> dict:
        """Premium for one policy-year with every step of the calculation."""
        missing = [k for k in self.inputs() if k not in policy]
        if missing:
            raise PlanError(f"missing rating inputs: {', '.join(sorted(set(missing)))}")
        running = self.base_rate
        steps = [
            {"step": "base rate", "factor": None, "input": None, "level": None, "multiplier": None, "running": running}
        ]
        for f in self.factors:
            value = policy[f["input"]]
            try:
                i = classify_one(value, f["levels"])
            except BandingError as e:
                raise PlanError(f"{f['name']}: {e}") from e
            lv = f["levels"][i]
            running *= lv["relativity"]
            steps.append(
                {
                    "step": "relativity",
                    "factor": f["name"],
                    "input": value,
                    "level": lv["label"],
                    "multiplier": lv["relativity"],
                    "running": running,
                }
            )
        for d in self.discounts:
            applies = bool(policy[d["input"]])
            mult = 1.0 - d["rate"] if applies else 1.0
            running *= mult
            steps.append(
                {
                    "step": "discount",
                    "factor": d["name"],
                    "input": applies,
                    "level": f"{d['rate']:.0%} off" if applies else "not applied",
                    "multiplier": mult,
                    "running": running,
                }
            )
        return {
            "plan_version": self.version,
            "plan_sha256": self.sha256,
            "effective_date": self.effective_date,
            "base_rate": self.base_rate,
            "steps": steps,
            "premium": round(running, 2),
        }

    def rate_portfolio(self, pf: Portfolio) -> np.ndarray:
        """Annual premium per row (vectorized; same math as rate())."""
        p = np.full(len(pf), self.base_rate)
        for f in self.factors:
            rel = np.array([lv["relativity"] for lv in f["levels"]])
            p *= rel[classify(pf[f["input"]], f["levels"])]
        for d in self.discounts:
            p *= np.where(pf[d["input"]].astype(bool), 1.0 - d["rate"], 1.0)
        return p


def load_plans(plans_dir: str | Path = PLANS_DIR) -> list[RatingPlan]:
    return sorted((RatingPlan.load(p) for p in Path(plans_dir).glob("*.json")), key=lambda p: p.version)


# ---- ratemaking steps -------------------------------------------------------------------------------------


def credibility(n_claims: float, full: float = FULL_CREDIBILITY_CLAIMS) -> float:
    """Square root rule: Z = min(1, sqrt(n / n_full))."""
    return min(1.0, math.sqrt(max(n_claims, 0.0) / full))


def smooth(rels: list[float], claims: list[float], base_index: int, complement: list[float]) -> tuple[list, list]:
    """Credibility-weight each relativity toward a complement: Z x fitted + (1 - Z) x complement.

    The complement is the current plan's relativity for the same policies, rebased to the new base level, which
    is the usual choice when a rating plan is refreshed. The base level stays exactly 1.0.
    """
    comp = [c / complement[base_index] for c in complement]
    out, zs = [], []
    for i, r in enumerate(rels):
        if i == base_index:
            out.append(1.0)
            zs.append(1.0)
            continue
        z = credibility(claims[i])
        out.append(z * r + (1 - z) * comp[i])
        zs.append(z)
    return out, zs


def complement_from_plan(plan: RatingPlan | None, factor: str, field: str, pf: Portfolio, levels: list[dict]):
    """Exposure-weighted relativity a plan charges the policies in each new level (1.0 if it lacks the factor)."""
    k = len(levels)
    if plan is None or factor not in [f["name"] for f in plan.factors]:
        return [1.0] * k
    f = plan.factor(factor)
    rel = np.array([lv["relativity"] for lv in f["levels"]])[classify(pf[f["input"]], f["levels"])]
    codes = classify(pf[field], levels)
    e = pf["exposure"]
    num = np.bincount(codes, weights=e * rel, minlength=k)
    den = np.bincount(codes, weights=e, minlength=k)
    return [float(n / d) if d > 0 else 1.0 for n, d in zip(num, den, strict=True)]


def cap_relativities(rels: list[float], lo: float, hi: float) -> list[float]:
    return [min(max(r, lo), hi) for r in rels]


def off_balance(base_rate: float, rel_product: np.ndarray, exposure: np.ndarray, target_premium: float) -> float:
    """Base rate that makes total earned premium equal the target: target / sum(exposure x base x relativities)."""
    current = float(np.sum(exposure * base_rate * rel_product))
    if current <= 0:
        raise PlanError("cannot off-balance a plan with zero premium")
    return base_rate * target_premium / current


def plan_levels(spec_levels: list[dict], rels: list[float]) -> list[dict]:
    out = []
    for lv, r in zip(spec_levels, rels, strict=True):
        d = dict(lv)
        d["relativity"] = round(float(r), 3)
        out.append(d)
    return out


def describe_levels(levels: list[dict]) -> list[str]:
    if is_numeric(levels):
        return [lv["label"] for lv in levels]
    return [f"{lv['label']} ({', '.join(str(v) for v in lv['values'])})" for lv in levels]
