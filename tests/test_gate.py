from ratingengine.cli import review_markdown
from ratingengine.logs import JsonFormatter
from ratingengine.pipeline import gate_check


def table(factor, rows):
    return {
        "factor": factor,
        "in_frequency": True,
        "levels": [{"level": lv, "base": b, "exposure": e, "freq": f, "freq_true": t} for lv, b, e, f, t in rows],
    }


MAIN = ["age_band", "territory", "prior_claims", "credit_band", "mileage", "coverage"]
GOOD = {"model": "glm_freq_x_sev", "holdout_gini": 0.3}


def good_tables():
    return [table(f, [("a", True, 50, 1.0, 1.0), ("b", False, 50, 1.05, 1.0)]) for f in MAIN]


def test_gate_passes_within_tolerance():
    g = gate_check(good_tables(), [GOOD], tolerance=0.10, gini_floor=0.2)
    assert g["passed"] and g["levels_checked"] == len(MAIN)


def test_gate_fails_on_relativity_outside_tolerance():
    t = good_tables()
    t[0]["levels"][1]["freq"] = 1.2
    g = gate_check(t, [GOOD], tolerance=0.10)
    assert not g["passed"] and "age_band=b" in g["failures"][0]


def test_gate_ignores_sparse_levels():
    t = good_tables()
    t[0]["levels"].append({"level": "tiny", "base": False, "exposure": 1, "freq": 3.0, "freq_true": 1.0})
    assert gate_check(t, [GOOD], tolerance=0.10)["passed"]


def test_gate_fails_on_low_gini():
    g = gate_check(good_tables(), [{"model": "glm_freq_x_sev", "holdout_gini": 0.1}], gini_floor=0.2)
    assert not g["passed"] and "Gini" in g["failures"][0]


def test_gate_fails_when_main_factor_missing():
    g = gate_check(good_tables()[1:], [GOOD])
    assert not g["passed"] and "age_band" in g["failures"][0]


def test_review_markdown_has_tables(review):
    md = review_markdown(review)
    assert "| Factor | Level | Fitted |" in md
    assert "Policies up more than 10%" in md
    assert "Gate:" in md


def test_json_log_format():
    import json
    import logging

    rec = logging.LogRecord("ratingengine", logging.INFO, __file__, 1, "rated", None, None)
    rec.fields = {"plan_version": "2027.01.0", "premium": 812.4}
    d = json.loads(JsonFormatter().format(rec))
    assert d["msg"] == "rated" and d["premium"] == 812.4 and d["level"] == "info"
