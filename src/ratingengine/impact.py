"""Rate change impact: per-policy premium changes, rate capping and its cost, and impact by segment.

Totals are earned premium (annual premium x exposure) on the experience period, so the overall change here
matches the off-balance target. Policy counts are counts of policy-years.
"""

from __future__ import annotations

import numpy as np

from .factors import classify

HIST_EDGES = [-0.40, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


def premium_change(current: np.ndarray, proposed: np.ndarray) -> np.ndarray:
    return np.asarray(proposed, float) / np.asarray(current, float) - 1.0


def apply_rate_cap(current, proposed, cap_up: float | None, cap_down: float | None) -> np.ndarray:
    """Limit each policy's change to +cap_up and -cap_down. None means no limit on that side."""
    cur = np.asarray(current, float)
    out = np.asarray(proposed, float).copy()
    if cap_up is not None:
        out = np.minimum(out, cur * (1 + cap_up))
    if cap_down is not None:
        out = np.maximum(out, cur * (1 - cap_down))
    return out


def cap_cost(current, proposed, capped, exposure) -> dict:
    """Premium given up by capping increases and premium kept by flooring decreases (earned basis)."""
    e = np.asarray(exposure, float)
    pro = np.asarray(proposed, float)
    cap = np.asarray(capped, float)
    diff = (pro - cap) * e
    lost = float(diff[diff > 0].sum())
    kept = float(-diff[diff < 0].sum()) + 0.0  # avoid -0.0
    cur_total = float(np.sum(np.asarray(current, float) * e))
    pro_total = float(np.sum(pro * e))
    cap_total = float(np.sum(cap * e))
    return {
        "premium_lost_to_cap": lost,
        "premium_kept_by_floor": kept,
        "net_cost": lost - kept,
        "net_cost_share_of_proposed": (lost - kept) / pro_total,
        "policies_capped": int(np.sum(pro > cap + 1e-9)),
        "policies_floored": int(np.sum(pro < cap - 1e-9)),
        "rate_change_uncapped": pro_total / cur_total - 1,
        "rate_change_capped": cap_total / cur_total - 1,
        # Base rate increase that would recover the cap cost if it were spread over every policy.
        "recovery_factor": pro_total / cap_total,
    }


def histogram(pct: np.ndarray, edges=HIST_EDGES) -> list[dict]:
    # Bins are (lo, hi]; rounding first keeps a change capped at exactly +15% in the (10%, 15%] bin.
    pct = np.round(np.asarray(pct, float), 9)
    bounds = [-np.inf] + list(edges) + [np.inf]
    out = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        n = int(np.sum((pct > lo) & (pct <= hi)))
        out.append({"lo": None if np.isinf(lo) else lo, "hi": None if np.isinf(hi) else hi, "count": n})
    return out


def change_counts(pct: np.ndarray, threshold: float = 0.10) -> dict:
    pct = np.asarray(pct, float)
    n = len(pct)
    up = int(np.sum(pct > threshold))
    down = int(np.sum(pct < -threshold))
    return {
        "policies": n,
        "up_more_than": up,
        "down_more_than": down,
        "within": n - up - down,
        "up_share": up / n,
        "down_share": down / n,
        "threshold": threshold,
        "median_change": float(np.median(pct)),
        "p05_change": float(np.quantile(pct, 0.05)),
        "p95_change": float(np.quantile(pct, 0.95)),
        "max_increase": float(pct.max()),
        "max_decrease": float(pct.min()),
    }


def segment_impact(codes, labels, current, proposed, capped, exposure, threshold: float = 0.10) -> list[dict]:
    e = np.asarray(exposure, float)
    pct = premium_change(current, proposed)
    k = len(labels)
    cur = np.bincount(codes, weights=current * e, minlength=k)
    pro = np.bincount(codes, weights=proposed * e, minlength=k)
    cap = np.bincount(codes, weights=capped * e, minlength=k)
    n = np.bincount(codes, minlength=k)
    up = np.bincount(codes, weights=(pct > threshold).astype(float), minlength=k)
    down = np.bincount(codes, weights=(pct < -threshold).astype(float), minlength=k)
    out = []
    for i in range(k):
        if n[i] == 0:
            continue
        out.append(
            {
                "segment": labels[i],
                "policies": int(n[i]),
                "avg_current": float(cur[i] / e[codes == i].sum()),
                "avg_proposed": float(pro[i] / e[codes == i].sum()),
                "change": float(pro[i] / cur[i] - 1),
                "change_capped": float(cap[i] / cur[i] - 1),
                "up_share": float(up[i] / n[i]),
                "down_share": float(down[i] / n[i]),
            }
        )
    return out


def impact_report(
    pf, current, proposed, segments: dict[str, list[dict]], cap_up=0.15, cap_down=None, threshold: float = 0.10
) -> dict:
    """segments maps a field name to a banding (levels list) used to group policies."""
    e = pf["exposure"]
    capped = apply_rate_cap(current, proposed, cap_up, cap_down)
    pct = premium_change(current, proposed)
    pct_capped = premium_change(current, capped)
    seg = {}
    for field, levels in segments.items():
        codes = classify(pf[field], levels)
        seg[field] = segment_impact(codes, [lv["label"] for lv in levels], current, proposed, capped, e, threshold)
    return {
        "cap_up": cap_up,
        "cap_down": cap_down,
        "counts": change_counts(pct, threshold),
        "counts_capped": change_counts(pct_capped, threshold),
        "histogram": histogram(pct),
        "histogram_capped": histogram(pct_capped),
        "cap": cap_cost(current, proposed, capped, e),
        "segments": seg,
    }
