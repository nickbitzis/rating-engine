"""Generalized linear models with a log link, fit by iteratively reweighted least squares (IRLS) in numpy.

Families: Poisson (frequency, variance mu), gamma (severity, variance mu^2) and Tweedie with power p in (1, 2)
(pure premium, variance mu^p). With a log link the IRLS working weight is w * mu^(2 - p) and the working
response is eta - offset + (y - mu) / mu.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaln


class GLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class Family:
    name: str
    power: float  # variance function V(mu) = mu^power
    fixed_scale: bool

    def unit_deviance(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        p = self.power
        if p == 1.0:
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(y > 0, y * np.log(y / mu), 0.0)
            return 2.0 * (t - (y - mu))
        if p == 2.0:
            return 2.0 * (-np.log(y / mu) + (y - mu) / mu)
        a = np.where(y > 0, np.power(np.maximum(y, 0.0), 2 - p), 0.0) / ((1 - p) * (2 - p))
        b = y * np.power(mu, 1 - p) / (1 - p)
        c = np.power(mu, 2 - p) / (2 - p)
        return 2.0 * (a - b + c)


POISSON = Family("poisson", 1.0, True)
GAMMA = Family("gamma", 2.0, False)


def tweedie(power: float = 1.6) -> Family:
    if not 1.0 < power < 2.0:
        raise GLMError("Tweedie power must be between 1 and 2")
    return Family(f"tweedie(p={power})", power, False)


@dataclass
class GLMResult:
    family: Family
    names: list[str]
    beta: np.ndarray
    cov: np.ndarray
    scale: float
    deviance: float
    null_deviance: float
    n_obs: int
    iterations: int
    converged: bool
    llf: float | None
    extra: dict = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        return len(self.beta)

    @property
    def df_resid(self) -> int:
        return self.n_obs - self.n_params

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @property
    def aic(self) -> float | None:
        if self.llf is None:
            return None
        k = self.n_params + (0 if self.family.fixed_scale else 1)
        return -2.0 * self.llf + 2.0 * k

    def predict(self, X: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
        eta = X @ self.beta
        if offset is not None:
            eta = eta + offset
        return np.exp(eta)

    def coef(self, name: str) -> float:
        return float(self.beta[self.names.index(name)])


def fit_glm(
    X: np.ndarray,
    y: np.ndarray,
    family: Family,
    weights: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    names: list[str] | None = None,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> GLMResult:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    if np.any(y < 0):
        raise GLMError("response must be non-negative")
    if family.power >= 2.0 and np.any(y <= 0):
        raise GLMError("gamma response must be positive")
    if np.any(w < 0):
        raise GLMError("weights must be non-negative")

    # Start from the weighted mean rate, blended with y, so log(mu) is defined everywhere.
    rate = np.sum(w * y) / np.sum(w * np.exp(off))
    mu = (y + rate * np.exp(off)) / 2.0
    mu = np.maximum(mu, 1e-10)
    eta = np.log(mu)
    dev = float(np.sum(w * family.unit_deviance(y, mu)))
    converged = False
    beta = np.zeros(k)
    it = 0
    for it in range(1, max_iter + 1):
        z = eta - off + (y - mu) / mu
        wt = w * np.power(mu, 2.0 - family.power)
        xtw = X.T * wt
        xtwx = xtw @ X
        try:
            beta_new = np.linalg.solve(xtwx, xtw @ z)
        except np.linalg.LinAlgError as e:
            raise GLMError(f"singular design matrix: {e}") from e
        eta_new = X @ beta_new + off
        mu_new = np.exp(np.clip(eta_new, -700, 700))
        dev_new = float(np.sum(w * family.unit_deviance(y, mu_new)))
        # Step halving if the deviance went up (rare with canonical-ish links, cheap insurance).
        halvings = 0
        while dev_new > dev * (1 + 1e-12) + 1e-12 and halvings < 20 and it > 1:
            beta_new = (beta + beta_new) / 2.0
            eta_new = X @ beta_new + off
            mu_new = np.exp(eta_new)
            dev_new = float(np.sum(w * family.unit_deviance(y, mu_new)))
            halvings += 1
        change = abs(dev - dev_new) / (abs(dev_new) + 0.1)
        beta, eta, mu, dev = beta_new, eta_new, mu_new, dev_new
        if change < tol:
            converged = True
            break
    wt = w * np.power(mu, 2.0 - family.power)
    xtwx = (X.T * wt) @ X
    if family.fixed_scale:
        scale = 1.0
    else:
        pearson = np.sum(w * (y - mu) ** 2 / np.power(mu, family.power))
        scale = float(pearson / max(n - k, 1))
    cov = scale * np.linalg.inv(xtwx)
    null_mu = np.exp(off) * np.sum(w * y) / np.sum(w * np.exp(off))
    null_dev = float(np.sum(w * family.unit_deviance(y, null_mu)))
    llf = _loglik(family, y, mu, w, scale)
    return GLMResult(
        family=family,
        names=names or [f"x{i}" for i in range(k)],
        beta=beta,
        cov=cov,
        scale=scale,
        deviance=dev,
        null_deviance=null_dev,
        n_obs=n,
        iterations=it,
        converged=converged,
        llf=llf,
    )


def _loglik(family: Family, y, mu, w, scale) -> float | None:
    if family.power == 1.0:
        return float(np.sum(w * (y * np.log(mu) - mu - gammaln(y + 1))))
    if family.power == 2.0:
        ws = w / scale
        ymu = y / mu
        return float(np.sum(ws * np.log(ws * ymu) - ws * ymu - gammaln(ws) - np.log(y)))
    return None  # the Tweedie density is an infinite series; compare Tweedie models by deviance instead


def deviance(family: Family, y, mu, weights=None) -> float:
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    return float(np.sum(w * family.unit_deviance(np.asarray(y, float), np.asarray(mu, float))))


def z_975() -> float:
    return 1.959963984540054


def ci_multiplier(level: float = 0.95) -> float:
    if abs(level - 0.95) < 1e-12:
        return z_975()
    from scipy.stats import norm

    return float(norm.ppf(0.5 + level / 2))
