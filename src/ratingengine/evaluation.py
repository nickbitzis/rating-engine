"""Model quality measures used in pricing reviews: lift, double lift, Gini, actual vs expected, loss ratios."""

from __future__ import annotations

import numpy as np


def exposure_quantiles(score: np.ndarray, exposure: np.ndarray, q: int = 10) -> np.ndarray:
    """Group index 0..q-1 by sorting on score and cutting into groups of equal exposure.

    A row belongs to the group that holds the midpoint of its exposure. Ties keep input order (stable sort).
    """
    score = np.asarray(score, float)
    e = np.asarray(exposure, float)
    order = np.argsort(score, kind="stable")
    cum = np.cumsum(e[order])
    mid = (cum - e[order] / 2.0) / cum[-1]
    g = np.minimum((mid * q).astype(int), q - 1)
    out = np.empty(len(score), dtype=int)
    out[order] = g
    return out


def gini(pred: np.ndarray, loss: np.ndarray, exposure: np.ndarray | None = None) -> float:
    """Ordered Lorenz Gini: sort by predicted loss cost ascending, plot cumulative exposure share against
    cumulative loss share, and return 1 - 2 x the area under that curve. 0 is no lift; higher is better."""
    pred = np.asarray(pred, float)
    loss = np.asarray(loss, float)
    e = np.ones(len(pred)) if exposure is None else np.asarray(exposure, float)
    order = np.argsort(pred, kind="stable")
    x = np.concatenate([[0.0], np.cumsum(e[order]) / e.sum()])
    y = np.concatenate([[0.0], np.cumsum(loss[order]) / loss.sum()])
    area = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))
    return 1.0 - 2.0 * area


def lift_table(pred_rate: np.ndarray, loss: np.ndarray, exposure: np.ndarray, q: int = 10) -> list[dict]:
    """Decile of predicted loss cost against actual loss cost per car-year."""
    g = exposure_quantiles(pred_rate, exposure, q)
    e = np.bincount(g, weights=exposure, minlength=q)
    pred_loss = np.bincount(g, weights=pred_rate * exposure, minlength=q)
    act = np.bincount(g, weights=loss, minlength=q)
    return [
        {
            "decile": i + 1,
            "exposure": float(e[i]),
            "predicted": float(pred_loss[i] / e[i]) if e[i] else 0.0,
            "actual": float(act[i] / e[i]) if e[i] else 0.0,
        }
        for i in range(q)
    ]


def lift_ratio(table: list[dict]) -> float:
    """Actual loss cost of the top group divided by the bottom group."""
    return table[-1]["actual"] / table[0]["actual"] if table[0]["actual"] else float("inf")


def double_lift(
    current: np.ndarray, proposed: np.ndarray, loss: np.ndarray, exposure: np.ndarray, q: int = 10
) -> list[dict]:
    """Sort by proposed / current, cut into equal exposure groups, and compare each plan's average premium
    (rebased to the overall average) with the actual loss cost (rebased the same way) in every group.

    The plan whose line tracks actual is the better one. Loss ratios per group are reported too.
    """
    current = np.asarray(current, float)
    proposed = np.asarray(proposed, float)
    e = np.asarray(exposure, float)
    ratio = proposed / current
    g = exposure_quantiles(ratio, e, q)
    ee = np.bincount(g, weights=e, minlength=q)
    cur = np.bincount(g, weights=current * e, minlength=q)
    pro = np.bincount(g, weights=proposed * e, minlength=q)
    act = np.bincount(g, weights=loss, minlength=q)
    cur_avg = cur.sum() / ee.sum()
    pro_avg = pro.sum() / ee.sum()
    act_avg = act.sum() / ee.sum()
    out = []
    for i in range(q):
        out.append(
            {
                "group": i + 1,
                "exposure": float(ee[i]),
                "ratio": float(pro[i] / cur[i]),
                "current": float(cur[i] / ee[i] / cur_avg),
                "proposed": float(pro[i] / ee[i] / pro_avg),
                "actual": float(act[i] / ee[i] / act_avg),
                "loss_ratio_current": float(act[i] / cur[i]),
                "loss_ratio_proposed": float(act[i] / pro[i]),
            }
        )
    return out


def double_lift_error(table: list[dict]) -> dict[str, float]:
    """Exposure-weighted mean absolute gap between each plan and actual, both rebased. Lower wins."""
    e = np.array([r["exposure"] for r in table])
    act = np.array([r["actual"] for r in table])
    out = {}
    for k in ("current", "proposed"):
        v = np.array([r[k] for r in table])
        out[k] = float(np.sum(e * np.abs(v - act)) / e.sum())
    return out


def actual_vs_expected(codes: np.ndarray, labels: list[str], loss: np.ndarray, expected: np.ndarray, exposure) -> list:
    k = len(labels)
    a = np.bincount(codes, weights=loss, minlength=k)
    x = np.bincount(codes, weights=expected, minlength=k)
    e = np.bincount(codes, weights=exposure, minlength=k)
    return [
        {
            "level": labels[i],
            "exposure": float(e[i]),
            "actual": float(a[i]),
            "expected": float(x[i]),
            "ae": float(a[i] / x[i]) if x[i] else None,
        }
        for i in range(k)
    ]


def actual_vs_expected_2way(codes_a, labels_a, codes_b, labels_b, loss, expected) -> dict:
    ka, kb = len(labels_a), len(labels_b)
    idx = codes_a * kb + codes_b
    a = np.bincount(idx, weights=loss, minlength=ka * kb).reshape(ka, kb)
    x = np.bincount(idx, weights=expected, minlength=ka * kb).reshape(ka, kb)
    ae = np.where(x > 0, a / np.maximum(x, 1e-12), np.nan)
    return {
        "rows": labels_a,
        "cols": labels_b,
        "ae": [[None if np.isnan(v) else round(float(v), 3) for v in row] for row in ae],
    }


def loss_ratio_by(codes, labels, loss, earned_by_plan: dict[str, np.ndarray]) -> list[dict]:
    k = len(labels)
    a = np.bincount(codes, weights=loss, minlength=k)
    out = []
    prem = {name: np.bincount(codes, weights=p, minlength=k) for name, p in earned_by_plan.items()}
    for i in range(k):
        row = {"level": labels[i], "loss": float(a[i])}
        for name, p in prem.items():
            row[f"premium_{name}"] = float(p[i])
            row[f"loss_ratio_{name}"] = float(a[i] / p[i]) if p[i] else None
        out.append(row)
    return out


def spread(values: list[float | None]) -> float:
    """Max minus min of a list, ignoring None. Used to show how much flatter one plan's loss ratios are."""
    v = [x for x in values if x is not None]
    return max(v) - min(v) if v else 0.0
