"""Forecast a well's cumulative oil from its first months, with intervals.

The forecast of cumulative production to month ``T`` from ``H`` months of history is

    observed oil in months 0 .. H-1  +  predicted oil in months H .. T-1

and the second term is drawn from the posterior predictive distribution:

1. posterior draws of the six parameters given the well's history (``LoadedModel.posterior``);
2. for each draw, the noise-free curve for the future months;
3. the AR(1) noise continued from the last observed month's residual, so a well running above
   its curve stays above it for a while, as fitted residuals do (median lag-one correlation 0.47);
4. future producing days: the curve is per producing day, and wells are down part of the time.
   Each draw takes one per-well uptime fraction from the development wells' distribution of
   uptime over the same months of life (median 0.98, 10th percentile about 0.79 over months
   12 to 35). Early and later uptime correlate only weakly (0.23), so it is drawn independently
   of the history.

Every method scored against the network uses the same observed-oil term and the same uptime
draws, so the comparison is between decline models, not between bookkeeping conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from declinecast.quantities import DAYS_PER_MONTH, R0_MEAN
from declinecast.simulator import T_DOF, curve, encode

__all__ = ["UptimeTable", "uptime_table", "forecast", "observed_cumulative",
           "future_factor"]


@dataclass
class UptimeTable:
    """Quantiles (101) of the per-well mean uptime in months [H, T), keyed by (H, T)."""

    quantiles: dict

    def at(self, history: int, horizon: int, p):
        """The uptime at probability levels ``p``."""
        q = np.asarray(self.quantiles[(history, horizon)])
        return np.interp(p, np.linspace(0, 1, len(q)), q)

    def sample(self, history: int, horizon: int, size, rng: np.random.Generator):
        return self.at(history, horizon, rng.random(size))

    def to_dict(self) -> dict:
        return {f"{h}-{t}": list(map(float, q)) for (h, t), q in self.quantiles.items()}

    @classmethod
    def from_dict(cls, d: dict) -> UptimeTable:
        return cls({tuple(int(v) for v in k.split("-")): q for k, q in d.items()})


def uptime_table(arrays, pairs) -> UptimeTable:
    """Build the table from development wells (``arrays`` must hold only those). A well counts
    for (H, T) if it has at least T months of record."""
    out = {}
    frac = np.clip(arrays.days / DAYS_PER_MONTH, 0.0, 1.0)
    for h, t in pairs:
        ok = arrays.available >= t
        u = frac[ok, h:t].mean(1)
        out[(int(h), int(t))] = np.quantile(u, np.linspace(0, 1, 101)).round(5).tolist()
    return UptimeTable(out)


def observed_cumulative(arrays, history: int) -> np.ndarray:
    """Barrels actually reported in months 0 .. history-1."""
    return arrays.oil[:, :history].sum(1)


def future_factor(model, history: int, horizon: int) -> float:
    """The recalibration factor for the predicted (future) oil, from the model manifest's
    ``forecast_adjustment`` table (fit on development wells by ``scripts/recalibrate.py``;
    FINDINGS section 5). Histories between table entries are interpolated in log months and
    held constant beyond them; a horizon not in the table takes the nearest one. 1.0 when the
    model has no table."""
    table = getattr(model, "manifest", {}).get("forecast_adjustment", {}).get("factors")
    if not table:
        return 1.0
    pairs = [tuple(int(v) for v in k.split("-")) + (f,) for k, f in table.items()]
    horizons = sorted({t for _, t, _ in pairs})
    t_use = min(horizons, key=lambda t: abs(t - horizon))
    pts = sorted((h, f) for h, t, f in pairs if t == t_use)
    hs, fs = np.log([p[0] for p in pts]), [p[1] for p in pts]
    return float(np.interp(np.log(history), hs, fs))


def _t(rng, shape):
    return rng.standard_t(T_DOF, size=shape) * np.sqrt((T_DOF - 2.0) / T_DOF)


def forecast(model, arrays, history: int, horizons, uptime: UptimeTable,
             n_draws: int = 500, seed: int = 0, rows=None, chunk: int = 64,
             return_draws: bool = False, adjust: bool = True) -> dict:
    """Predictive draws of cumulative oil (barrels) to each month count in ``horizons`` (an
    int or a sequence) for the wells in ``arrays`` (or the subset ``rows``), each seeing
    ``history`` months. One set of posterior and noise draws serves every horizon.

    Returns a dict with, per horizon, the ``quantiles`` (n, len(LEVELS_Q)) and, if asked, the
    ``draws`` (n, n_draws); also ``observed`` (the reported oil to date), and the posterior's
    ``acceptance``. ``adjust=False`` skips the development-well recalibration of the predicted
    oil (``future_factor``)."""
    horizons = tuple(np.atleast_1d(horizons).astype(int))
    horizon = max(horizons)
    rows = np.arange(len(arrays.api)) if rows is None else np.asarray(rows)
    rng = np.random.default_rng(seed)
    x = encode(arrays.rate[rows], np.full(len(rows), history), arrays.lateral_ft[rows])
    seen = observed_cumulative(arrays, history)[rows]
    k_future = np.arange(history, horizon, dtype=float)
    q_out = {t: np.empty((len(rows), len(LEVELS_Q))) for t in horizons}
    q_raw = {t: np.empty((len(rows), len(LEVELS_Q))) for t in horizons}
    keep = {t: np.empty((len(rows), n_draws)) for t in horizons} if return_draws else None
    acc = np.empty(len(rows))
    factor = {t: (future_factor(model, history, t) if adjust else 1.0) for t in horizons}
    for s in range(0, len(rows), chunk):
        idx = rows[s:s + chunk]
        b = len(idx)
        theta, acc[s:s + b] = model.posterior(x[s:s + chunk], n_draws, seed=seed + s)
        theta = theta.astype(float)                                   # (b, n, 6)
        lq, ld, bb, tp = (theta[..., i:i + 1] for i in range(4))
        sigma = 10.0 ** theta[..., 4]                                 # (b, n), natural log
        rho = theta[..., 5]
        # last observed month inside the history, and its residual under each draw
        r = arrays.rate[idx, :history]
        obs = np.isfinite(r) & (r > 0)
        last = np.where(obs.any(1), history - 1 - np.argmax(obs[:, ::-1], axis=1), -1)
        e = np.zeros((b, n_draws))
        has = last >= 0
        if has.any():
            kl = last[has].astype(float)[:, None, None]
            mu_last = curve(kl, lq[has], ld[has], bb[has], tp[has], R0_MEAN)[..., 0]
            e[has] = np.log(r[has, last[has]])[:, None] - np.log(mu_last)
        # carry the AR(1) from the last observed month through to the horizon
        # (a well with no observed month starts from the stationary distribution)
        start = np.where(has, last + 1, 0)
        m0 = int(start.min())
        z = _t(rng, (b, n_draws, horizon - m0))
        noise = np.empty((b, n_draws, len(k_future)))
        scale = sigma * np.sqrt(1.0 - rho ** 2)
        for j in range(horizon - m0):
            month = m0 + j
            fresh = ((month == start) & ~has)[:, None]
            new = np.where(fresh, sigma * z[..., j], rho * e + scale * z[..., j])
            e = np.where((month >= start)[:, None], new, e)
            if month >= history:
                noise[..., month - history] = e
        mu = curve(k_future, lq, ld, bb, tp, R0_MEAN)                 # (b, n, F)
        running = np.cumsum(mu * np.exp(noise), axis=-1)
        p_up = rng.random((b, n_draws))  # one uptime rank per draw, shared by the horizons
        for t in horizons:
            u = uptime.at(history, t, p_up)
            future = (running[..., t - history - 1] if t > history else 0.0) * u * DAYS_PER_MONTH
            if factor[t] != 1.0:
                q_raw[t][s:s + b] = np.quantile(seen[s:s + b, None] + future, LEVELS_Q, axis=1).T
            future = future * factor[t]
            total = seen[s:s + b, None] + future
            q_out[t][s:s + b] = np.quantile(total, LEVELS_Q, axis=1).T
            if return_draws:
                keep[t][s:s + b] = total
    out = {"quantiles": q_out, "observed": seen, "acceptance": acc, "rows": rows,
           "factors": factor}
    if any(v != 1.0 for v in factor.values()):
        out["quantiles_unadjusted"] = q_raw
    if return_draws:
        out["draws"] = keep
    return out


#: Quantile levels every forecast reports: the median and the central 50, 68, 80, 90 and 95%
#: intervals (80% is P90-P10 in reserves language).
LEVELS_Q = (0.025, 0.05, 0.10, 0.16, 0.25, 0.50, 0.75, 0.84, 0.90, 0.95, 0.975)
