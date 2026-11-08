"""FastAPI service: rate a policy with its full breakdown, plans, model results, impact, health and metrics."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field, model_validator

from . import __version__
from .logs import get_logger, log_event
from .metrics import Metrics
from .pipeline import Review, build, round_floats
from .plan import PlanError

STATIC = Path(__file__).parent / "static"


class Policy(BaseModel):
    driver_age: int = Field(ge=16, le=95)
    years_licensed: int = Field(ge=0, le=79)
    vehicle_age: int = Field(ge=0, le=40)
    vehicle_value: float = Field(gt=0, le=500_000)
    territory: int = Field(ge=1, le=10)
    annual_mileage: int = Field(ge=0, le=150_000)
    prior_claims: int = Field(ge=0, le=20)
    credit_band: int = Field(ge=1, le=5)
    multi_policy: bool
    coverage: Literal["liability", "standard", "full"]

    @model_validator(mode="after")
    def licensed_after_16(self):
        if self.years_licensed > self.driver_age - 16:
            raise ValueError("years_licensed cannot exceed driver_age - 16")
        return self


class RateRequest(BaseModel):
    policy: Policy
    plan_version: str | None = None


def parse_cap(value: str, name: str, upper: float) -> float | None:
    if value.strip().lower() in ("", "none"):
        return None
    try:
        v = float(value)
    except ValueError as e:
        raise HTTPException(422, f"{name} must be a number or 'none'") from e
    if not 0 <= v <= upper:
        raise HTTPException(422, f"{name} must be between 0 and {upper}")
    return v


def create_app(review: Review | None = None, n: int | None = None, seed: int | None = None) -> FastAPI:
    log = get_logger()
    metrics = Metrics()
    t0 = time.perf_counter()
    if review is None:
        review = build(
            n=n or int(os.environ.get("RE_N_POLICIES", "200000")),
            seed=seed or int(os.environ.get("RE_SEED", "2026")),
            with_gbm=os.environ.get("RE_GBM", "1") != "0",
        )
    built_s = time.perf_counter() - t0
    plans = review.plans
    latest = max(plans)
    res = review.results
    log_event(
        log,
        "review built",
        policies=review.n,
        seed=review.seed,
        seconds=round(built_s, 2),
        plans=sorted(plans),
        gate_passed=res["gate"]["passed"],
    )
    metrics.set_gauge("re_glm_holdout_gini", "Holdout Gini of the GLM pure premium.", res["gate"]["glm_holdout_gini"])
    metrics.set_gauge("re_portfolio_policies", "Policy-years in the synthetic portfolio.", review.n)
    metrics.set_gauge("re_gate_passed", "1 if the relativity and Gini release gate passed.", int(res["gate"]["passed"]))

    app = FastAPI(title="Rating Engine", version=__version__)
    app.state.review = review
    app.state.metrics = metrics

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        t = time.perf_counter()
        response = await call_next(request)
        dt = time.perf_counter() - t
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        metrics.observe_http(path, response.status_code, dt)
        response.headers["x-request-id"] = rid
        log_event(
            log,
            "request",
            request_id=rid,
            method=request.method,
            route=path,
            status=response.status_code,
            duration_ms=round(dt * 1000, 2),
        )
        return response

    def get_plan(version: str | None):
        v = version or latest
        if v not in plans:
            raise HTTPException(404, f"no rating plan version {v}")
        return plans[v]

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "version": __version__,
            "policies": review.n,
            "seed": review.seed,
            "plans": sorted(plans),
            "build_seconds": round(built_s, 2),
            "gate_passed": res["gate"]["passed"],
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def prom():
        return metrics.render()

    @app.get("/api/plans")
    def list_plans():
        return [
            {
                "version": p.version,
                "status": p.status,
                "effective_date": p.effective_date,
                "sha256": p.sha256,
                "base_rate": p.base_rate,
                "factors": [f["name"] for f in p.factors],
                "discounts": [d["name"] for d in p.discounts],
                "description": p.description,
            }
            for p in sorted(plans.values(), key=lambda p: p.version)
        ]

    @app.get("/api/plans/{version}")
    def plan_detail(version: str):
        return get_plan(version).to_dict()

    @app.post("/rate")
    def rate(req: RateRequest):
        plan = get_plan(req.plan_version)
        pol = req.policy.model_dump()
        try:
            out = plan.rate(pol)
        except PlanError as e:
            metrics.rating_errors += 1
            raise HTTPException(422, str(e)) from e
        out["other_plans"] = {v: p.rate(pol)["premium"] for v, p in sorted(plans.items()) if v != plan.version}
        metrics.observe_rating(plan.version)
        log_event(log, "rated", plan_version=plan.version, premium=out["premium"])
        return out

    @app.get("/api/results")
    def results():
        return res

    @app.get("/api/relativities")
    def relativities():
        return res["relativities"]

    @app.get("/api/impact")
    def impact(
        cap_up: str = Query("0.15", description="maximum increase per policy, or 'none'"),
        cap_down: str = Query("none", description="maximum decrease per policy, or 'none'"),
        threshold: float = Query(0.10, gt=0, le=1),
    ):
        up = parse_cap(cap_up, "cap_up", 5.0)
        down = parse_cap(cap_down, "cap_down", 1.0)
        metrics.impact_runs += 1
        return round_floats(review.impact(cap_up=up, cap_down=down, threshold=threshold))

    @app.get("/api/sample-policy/{index}")
    def sample_policy(index: int):
        if not 0 <= index < review.n:
            raise HTTPException(404, "no such policy")
        pf = review.portfolio
        return {"policy_id": str(pf["policy_id"][index]), "policy": pf.policy(index)}

    return app
