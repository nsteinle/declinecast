"""Three baselines the network must beat, each with intervals at the same quantile levels.

* ``multiplier``: the well's own cumulative to month H, times the reference wells' ratio of
  cumulative at T to cumulative at H. Interval: the reference wells' spread of that ratio.
* ``type_curve``: the lateral length only, as the reference wells' cumulative at T per foot of
  lateral times this well's lateral. Interval: the reference wells' spread.
* ``arps_ls``: a per-well least-squares fit of the same modified Arps law to the log rate from
  the peak. Interval: the fit's parameter covariance and its residual scatter, the usual way a
  single fit is given an interval.

The multiplier is the strongest simple benchmark: it is a nonparametric, empirical forecast that
is calibrated on average by construction. The type curve is what can be said before a well
produces. The least-squares fit is the analyst's default, and the pre-registered comparison
(17 October 2026) is against it. The two empirical baselines are fit on reference wells only:
development wells for the locked test, and the other folds in development cross-validation.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from declinecast import arps
from declinecast.forecasting import LEVELS_Q, UptimeTable, observed_cumulative
from declinecast.priors import LOWER, UPPER
from declinecast.quantities import DAYS_PER_MONTH

__all__ = ["multiplier", "type_curve", "arps_ls", "B_FALLBACK"]

#: b used when a history has too few months after the peak to fit it: the median of per-well
#: fits from the peak in development wells (FINDINGS section 2).
B_FALLBACK = 0.77


def _q(values):
    return np.quantile(values, LEVELS_Q)


def multiplier(reference, target, history: int, horizon: int) -> np.ndarray:
    """(n, len(LEVELS_Q)) quantiles of cumulative to ``horizon`` for ``target`` wells."""
    ok = reference.available >= horizon
    c_h = observed_cumulative(reference, history)[ok]
    c_t = reference.oil[ok, :horizon].sum(1)
    good = c_h > 0
    lq = _q(np.log10(c_t[good] / c_h[good]))
    seen = observed_cumulative(target, history)
    with np.errstate(divide="ignore"):
        out = np.log10(seen)[:, None] + lq[None, :]
    out = 10.0 ** out
    out[seen <= 0] = np.nan
    return out


def type_curve(reference, target, horizon: int) -> np.ndarray:
    ok = (reference.available >= horizon) & (reference.lateral_ft > 0)
    per_ft = reference.oil[ok, :horizon].sum(1) / reference.lateral_ft[ok]
    q = _q(per_ft)
    return target.lateral_ft[:, None] * q[None, :]


def _fit_one(k, y, b_free):
    """Least squares on natural-log rate; returns (params, covariance, residual sd)."""
    def model(p, kk):
        lq, ld = p[0], p[1]
        b = p[2] if b_free else B_FALLBACK
        return np.log(arps.rate(kk, 10.0 ** lq, 10.0 ** ld, b))

    p0 = [np.log10(np.exp(y[0])), np.log10(3.0)] + ([B_FALLBACK] if b_free else [])
    lo = [LOWER[0] - 1.0, LOWER[1]] + ([LOWER[2]] if b_free else [])
    hi = [UPPER[0] + 1.0, UPPER[1]] + ([UPPER[2]] if b_free else [])
    p0 = np.clip(p0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
    res = least_squares(lambda p: model(p, k) - y, p0, bounds=(lo, hi))
    dof = max(len(y) - len(p0), 1)
    s = float(np.sqrt(2.0 * res.cost / dof)) if len(y) > len(p0) else np.nan
    jtj = res.jac.T @ res.jac
    cov = np.linalg.pinv(jtj) * (s ** 2 if np.isfinite(s) else 0.0)
    params = res.x if b_free else np.r_[res.x, B_FALLBACK]
    if not b_free:
        cov = np.pad(cov, ((0, 1), (0, 1)))
    return params, cov, s


def arps_ls(target, history: int, horizon: int, uptime: UptimeTable, n_draws: int = 500,
            seed: int = 0, sigma_floor: float = 0.05) -> np.ndarray:
    """Quantiles of cumulative to ``horizon``, one fit per well from its observed peak within
    the first six months of the history. Fewer than two months after the peak gives NaN; fewer
    than four fixes b at ``B_FALLBACK``. A residual sd that cannot be estimated, or one below
    ``sigma_floor``, is set to ``sigma_floor``."""
    rng = np.random.default_rng(seed)
    n = len(target.api)
    seen = observed_cumulative(target, history)
    out = np.full((n, len(LEVELS_Q)), np.nan)
    for i in range(n):
        r = target.rate[i, :history]
        obs = np.isfinite(r) & (r > 0)
        if not obs.any():
            continue
        head = np.where(obs[:min(6, history)], r[:min(6, history)], -1.0)
        peak = int(head.argmax())
        k = np.flatnonzero(obs & (np.arange(history) >= peak))
        if len(k) < 2:
            continue
        y = np.log(r[k])
        params, cov, s = _fit_one((k - peak).astype(float), y, b_free=len(k) >= 4)
        s = sigma_floor if not np.isfinite(s) else max(s, sigma_floor)
        draws = rng.multivariate_normal(params, cov, size=n_draws, check_valid="ignore")
        lo = np.array([LOWER[0] - 1.0, LOWER[1], LOWER[2]])
        hi = np.array([UPPER[0] + 1.0, UPPER[1], UPPER[2]])
        draws = np.clip(draws, lo, hi)
        t = np.arange(history, horizon, dtype=float) - peak
        mu = arps.rate(t[None, :], 10.0 ** draws[:, :1], 10.0 ** draws[:, 1:2], draws[:, 2:3])
        noise = s * rng.standard_normal(mu.shape)
        u = uptime.sample(history, horizon, n_draws, rng)
        total = seen[i] + (mu * np.exp(noise)).sum(1) * u * DAYS_PER_MONTH
        out[i] = np.quantile(total, LEVELS_Q)
    return out
