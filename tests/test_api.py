import pytest
from fastapi.testclient import TestClient

from ratingengine.app import create_app

POLICY = {
    "driver_age": 22,
    "years_licensed": 4,
    "vehicle_age": 2,
    "vehicle_value": 31000,
    "territory": 9,
    "annual_mileage": 16000,
    "prior_claims": 1,
    "credit_band": 4,
    "multi_policy": False,
    "coverage": "full",
}


@pytest.fixture(scope="module")
def client(review):
    return TestClient(create_app(review=review))


def test_healthz(client, review):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["policies"] == review.n
    assert body["plans"] == ["2026.01.0", "2027.01.0"]


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Rating Engine" in r.text


def test_rate_breakdown_multiplies_to_premium(client):
    out = client.post("/rate", json={"policy": POLICY}).json()
    p = out["base_rate"]
    for s in out["steps"][1:]:
        p *= s["multiplier"]
    assert round(p, 2) == out["premium"]
    assert out["plan_version"] == "2027.01.0"
    assert len(out["plan_sha256"]) == 64


def test_rate_with_older_plan_version(client):
    new = client.post("/rate", json={"policy": POLICY}).json()
    old = client.post("/rate", json={"policy": POLICY, "plan_version": "2026.01.0"}).json()
    assert old["plan_version"] == "2026.01.0"
    assert old["premium"] != new["premium"]
    assert new["other_plans"]["2026.01.0"] == old["premium"]


def test_unknown_plan_version_404(client):
    r = client.post("/rate", json={"policy": POLICY, "plan_version": "1999.01.0"})
    assert r.status_code == 404


def test_invalid_policy_422(client):
    bad = {**POLICY, "years_licensed": 10}  # 22-year-old licensed 10 years
    assert client.post("/rate", json={"policy": bad}).status_code == 422
    assert client.post("/rate", json={"policy": {**POLICY, "coverage": "gold"}}).status_code == 422


def test_plans_list_and_detail(client):
    plans = client.get("/api/plans").json()
    assert [p["version"] for p in plans] == ["2026.01.0", "2027.01.0"]
    d = client.get("/api/plans/2027.01.0").json()
    assert d["sha256"] == plans[1]["sha256"]
    assert d["notes"]["off_balance_factor"] > 0
    assert client.get("/api/plans/0.0.0").status_code == 404


def test_results_endpoint(client):
    r = client.get("/api/results").json()
    for key in [
        "relativities",
        "lift",
        "double_lift",
        "actual_vs_expected",
        "actual_vs_expected_2way",
        "loss_ratios",
        "impact",
        "gate",
        "models",
    ]:
        assert key in r
    assert len(r["lift"]) == 10 and len(r["double_lift"]) == 10


def test_impact_endpoint_cap_parameter(client):
    a = client.get("/api/impact?cap_up=0.10").json()
    b = client.get("/api/impact?cap_up=5").json()
    assert a["cap"]["net_cost"] > 0
    assert b["cap"]["policies_capped"] == 0
    assert client.get("/api/impact?cap_up=-1").status_code == 422
    assert client.get("/api/impact?cap_up=abc").status_code == 422
    none = client.get("/api/impact?cap_up=none").json()
    assert none["cap_up"] is None and none["cap"]["net_cost"] == 0
    both = client.get("/api/impact?cap_up=0.15&cap_down=0.15").json()
    assert both["cap"]["policies_floored"] > 0


def test_sample_policy_endpoint_is_rateable(client):
    s = client.get("/api/sample-policy/12").json()
    assert client.post("/rate", json={"policy": s["policy"]}).status_code == 200
    assert client.get("/api/sample-policy/99999999").status_code == 404


def test_metrics_after_rating(client):
    client.post("/rate", json={"policy": POLICY})
    text = client.get("/metrics").text
    assert 're_ratings_total{plan_version="2027.01.0"}' in text
    assert "re_glm_holdout_gini" in text
    assert "re_request_duration_seconds_bucket" in text


def test_request_id_echoed(client):
    r = client.get("/healthz", headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"
