"""Command line: serve, review (markdown summary), gate (CI release check), export-plan, rate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .pipeline import build
from .plan import PLANS_DIR, RatingPlan, load_plans


def cmd_serve(args):
    import uvicorn

    uvicorn.run(
        "ratingengine.app:create_app",
        factory=True,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", args.port)),
        access_log=False,
    )


def pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def review_markdown(review) -> str:
    r = review.results
    m = r["meta"]
    lines = [
        f"## Pricing review: {m['policies']:,} policy-years, seed {m['seed']}",
        "",
        f"{m['claims']:,} claims, ${m['incurred_loss']:,.0f} incurred, {m['exposure']:,.0f} car-years. "
        f"Fit on {m['train_policies']:,}, holdout {m['holdout_policies']:,}.",
        "",
        "### Fitted vs true frequency relativities (fit on the 70% training split, 95% intervals)",
        "",
        "| Factor | Level | Fitted | 95% CI | True | Error |",
        "|---|---|---:|---|---:|---:|",
    ]
    for t in r["relativities"]:
        if not t["in_frequency"]:
            continue
        for lv in t["levels"]:
            if lv["base"]:
                continue
            err = lv["freq"] / lv["freq_true"] - 1
            lines.append(
                f"| {t['factor']} | {lv['level']} | {lv['freq']:.3f} | {lv['freq_lo']:.3f} to {lv['freq_hi']:.3f} "
                f"| {lv['freq_true']:.3f} | {pct(err)} |"
            )
    lines += [
        "",
        "### Pure premium models on the holdout",
        "",
        "| Model | Gini | Top / bottom decile |",
        "|---|---:|---:|",
    ]
    for row in r["models"]["pure_premium"]:
        lines.append(f"| {row['model']} | {row['holdout_gini']:.3f} | {row['lift_ratio']:.2f}x |")
    imp = r["impact"]
    c, cap = imp["counts"], imp["cap"]
    lines += [
        "",
        f"### Rate change impact, {imp['current_version']} to {imp['proposed_version']}",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Overall rate change | {pct(cap['rate_change_uncapped'])} |",
        f"| Policies up more than 10% | {c['up_more_than']:,} ({c['up_share']:.1%}) |",
        f"| Policies down more than 10% | {c['down_more_than']:,} ({c['down_share']:.1%}) |",
        f"| Median change | {pct(c['median_change'])} |",
        f"| 5th to 95th percentile | {pct(c['p05_change'])} to {pct(c['p95_change'])} |",
        f"| Policies capped at +{imp['cap_up']:.0%} | {cap['policies_capped']:,} |",
        f"| Premium given up by the cap | ${cap['net_cost']:,.0f} "
        f"({cap['net_cost_share_of_proposed']:.1%} of proposed) |",
        f"| Overall rate change after the cap | {pct(cap['rate_change_capped'])} |",
        "",
        f"Loss ratio: current plan {r['overall_loss_ratio']['current']:.1%}, proposed "
        f"{r['overall_loss_ratio']['proposed']:.1%} (target {m['target_loss_ratio']:.0%}). "
        f"Double lift mean gap to actual: current {r['double_lift_error']['current']:.3f}, "
        f"proposed {r['double_lift_error']['proposed']:.3f}.",
    ]
    g = r["gate"]
    lines += [
        "",
        f"Gate: {'PASSED' if g['passed'] else 'FAILED'} ({g['levels_checked']} levels checked, "
        f"worst error {g['worst_relative_error']:.1%}, tolerance {g['tolerance']:.0%}, "
        f"GLM holdout Gini {g['glm_holdout_gini']:.3f}, floor {g['gini_floor']})",
    ]
    lines += [f"- {f}" for f in g["failures"]]
    return "\n".join(lines) + "\n"


def cmd_review(args):
    review = build(args.n, args.seed, with_gbm=not args.no_gbm)
    md = review_markdown(review)
    print(md)
    if args.step_summary and os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write(md)
    if args.json:
        Path(args.json).write_text(json.dumps(review.results, indent=1))


def cmd_gate(args):
    from .pipeline import gate_check

    review = build(args.n, args.seed, with_gbm=False)
    g = gate_check(
        review.results["relativities"],
        review.results["models"]["pure_premium"],
        tolerance=args.tolerance,
        gini_floor=args.gini_floor,
    )
    print(json.dumps(g, indent=2))
    if args.step_summary and os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write(review_markdown(review))
    sys.exit(0 if g["passed"] else 1)


def cmd_export_plan(args):
    review = build(args.n, args.seed, with_gbm=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    review.proposed.save(out)
    print(f"wrote {out} version {review.proposed.version} sha256 {review.proposed.sha256[:12]}")


def cmd_rate(args):
    policy = json.loads(Path(args.file).read_text()) if args.file else json.loads(sys.stdin.read())
    plans = {p.version: p for p in load_plans(args.plans_dir)}
    if args.plan_file:
        p = RatingPlan.load(args.plan_file)
        plans[p.version] = p
    plan = plans[args.plan_version or max(plans)]
    print(json.dumps(plan.rate(policy), indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ratingengine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8870)
    s.set_defaults(func=cmd_serve)

    for name, func in [("review", cmd_review), ("gate", cmd_gate), ("export-plan", cmd_export_plan)]:
        p = sub.add_parser(name)
        p.add_argument("--n", type=int, default=200_000)
        p.add_argument("--seed", type=int, default=2026)
        p.set_defaults(func=func)
        if name == "review":
            p.add_argument("--no-gbm", action="store_true")
            p.add_argument("--json")
            p.add_argument("--step-summary", action="store_true")
        if name == "gate":
            p.add_argument("--tolerance", type=float, default=0.12)
            p.add_argument("--gini-floor", type=float, default=0.25)
            p.add_argument("--step-summary", action="store_true")
        if name == "export-plan":
            p.add_argument("--out", default=str(PLANS_DIR / "proposed" / "plan-2027.01.0.json"))

    r = sub.add_parser("rate")
    r.add_argument("file", nargs="?")
    r.add_argument("--plan-version")
    r.add_argument("--plan-file", help="a plan JSON outside plans/, such as the exported proposed plan")
    r.add_argument("--plans-dir", default=str(PLANS_DIR))
    r.set_defaults(func=cmd_rate)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
