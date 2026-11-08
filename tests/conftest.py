from __future__ import annotations

import pytest

from ratingengine.pipeline import build

TEST_N = 80_000


@pytest.fixture(scope="session")
def review():
    # A smaller book than the 200,000 default keeps the suite fast; the CI gate runs the full size.
    return build(TEST_N, 2026, with_gbm=False)


@pytest.fixture(scope="session")
def portfolio(review):
    return review.portfolio
