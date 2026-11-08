"""Prometheus text exposition without extra dependencies."""

from __future__ import annotations

import threading
from collections import defaultdict

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


def _labels(d: dict) -> str:
    return "{" + ",".join(f'{k}="{v}"' for k, v in sorted(d.items())) + "}" if d else ""


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.http = defaultdict(int)  # (route, status)
        self.ratings = defaultdict(int)  # plan version
        self.rating_errors = 0
        self.impact_runs = 0
        self.buckets = [0] * len(BUCKETS)
        self.dur_sum = 0.0
        self.dur_count = 0
        self.gauges: dict[str, tuple[str, float]] = {}

    def observe_http(self, route: str, status: int, seconds: float) -> None:
        with self.lock:
            self.http[(route, str(status))] += 1
            for i, b in enumerate(BUCKETS):
                if seconds <= b:
                    self.buckets[i] += 1
            self.dur_sum += seconds
            self.dur_count += 1

    def observe_rating(self, version: str) -> None:
        with self.lock:
            self.ratings[version] += 1

    def set_gauge(self, name: str, help_text: str, value: float) -> None:
        with self.lock:
            self.gauges[name] = (help_text, value)

    def render(self) -> str:
        out: list[str] = []
        with self.lock:
            out += [
                "# HELP re_http_requests_total HTTP requests by route and status.",
                "# TYPE re_http_requests_total counter",
            ]
            out += [
                f"re_http_requests_total{_labels({'route': r, 'status': s})} {v}"
                for (r, s), v in sorted(self.http.items())
            ]
            out += ["# HELP re_ratings_total Policies rated by plan version.", "# TYPE re_ratings_total counter"]
            out += [f"re_ratings_total{_labels({'plan_version': p})} {v}" for p, v in sorted(self.ratings.items())]
            out += [
                "# HELP re_rating_errors_total Rating requests rejected for invalid inputs.",
                "# TYPE re_rating_errors_total counter",
                f"re_rating_errors_total {self.rating_errors}",
            ]
            out += [
                "# HELP re_impact_runs_total Rate change impact analyses run.",
                "# TYPE re_impact_runs_total counter",
                f"re_impact_runs_total {self.impact_runs}",
            ]
            for name, (help_text, value) in sorted(self.gauges.items()):
                out += [f"# HELP {name} {help_text}", f"# TYPE {name} gauge", f"{name} {value}"]
            out += [
                "# HELP re_request_duration_seconds Request latency.",
                "# TYPE re_request_duration_seconds histogram",
            ]
            for b, c in zip(BUCKETS, self.buckets, strict=True):
                out.append(f're_request_duration_seconds_bucket{{le="{b}"}} {c}')
            out.append(f're_request_duration_seconds_bucket{{le="+Inf"}} {self.dur_count}')
            out.append(f"re_request_duration_seconds_sum {self.dur_sum:.6f}")
            out.append(f"re_request_duration_seconds_count {self.dur_count}")
        return "\n".join(out) + "\n"
