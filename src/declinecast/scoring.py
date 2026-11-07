"""Scores for quantile forecasts of cumulative production, and a clustered bootstrap.

All scores are on log10 barrels, because errors in production forecasts are multiplicative: a
20% miss on a 50,000-barrel well matters as much as one on a 500,000-barrel well, and a score in
barrels would be set by the few largest wells.

* **WIS**, the weighted interval score (Bracher et al. 2021, PLOS Comput. Biol. 17, e1008618):
  a proper score for a set of central intervals plus the median, rewarding narrow intervals
  and penalizing misses by the distance outside. Lower is better; with one interval and no
  median it is the interval score of Gneiting and Raftery (2007).
* **Coverage** of the central 50, 68, 80, 90 and 95% intervals. The 80% interval is P90-P10.
* **Median error**: median absolute log10 error of the forecast median, also shown as a
  percentage (10^error - 1), and the median signed error (bias).

Wells are not independent: wells on one pad share a schedule and a completion design, and wells
of one operator share practices. ``cluster_bootstrap`` resamples whole clusters, so the
intervals on scores and on score differences reflect the number of clusters, which is far
smaller than the well count (32 first operators among the 5,415 development wells, 27 among the
2,929 test wells).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from declinecast.forecasting import LEVELS_Q

__all__ = ["INTERVALS", "wis", "covered", "score_table", "cluster_bootstrap"]

INTERVALS = (0.50, 0.68, 0.80, 0.90, 0.95)


def _bounds(q: np.ndarray, level: float):
    lv = list(np.round(LEVELS_Q, 4))
    return q[:, lv.index(round((1 - level) / 2, 4))], q[:, lv.index(round((1 + level) / 2, 4))]


def wis(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-forecast WIS on log10 values. ``q`` (n, len(LEVELS_Q)) in barrels, ``y`` (n,)."""
    lq, ly = np.log10(q), np.log10(y)
    med = lq[:, list(np.round(LEVELS_Q, 4)).index(0.5)]
    total = 0.5 * np.abs(ly - med)
    for level in INTERVALS:
        alpha = 1.0 - level
        lo, hi = _bounds(lq, level)
        score = (hi - lo) + 2.0 / alpha * (np.maximum(lo - ly, 0) + np.maximum(ly - hi, 0))
        total = total + alpha / 2.0 * score
    return total / (len(INTERVALS) + 0.5)


def covered(q: np.ndarray, y: np.ndarray, level: float) -> np.ndarray:
    lo, hi = _bounds(q, level)
    return (y >= lo) & (y <= hi)


def _metrics(q, y) -> dict:
    med = q[:, list(np.round(LEVELS_Q, 4)).index(0.5)]
    err = np.log10(med) - np.log10(y)
    out = {"n": int(len(y)), "wis": float(np.mean(wis(q, y))),
           "median_abs_log_error": float(np.median(np.abs(err))),
           "median_abs_pct_error": float(10 ** np.median(np.abs(err)) - 1),
           "median_log_bias": float(np.median(err))}
    for level in INTERVALS:
        out[f"coverage_{int(level * 100)}"] = float(covered(q, y, level).mean())
    lo, hi = _bounds(np.log10(q), 0.80)
    out["width_80_log10"] = float(np.median(hi - lo))
    return out


def score_table(forecasts: dict[str, np.ndarray], y: np.ndarray, **labels) -> pd.DataFrame:
    """One row per method; rows where any method has no forecast are dropped for all, so every
    method is scored on the same wells."""
    ok = np.isfinite(y) & (y > 0)
    for q in forecasts.values():
        ok &= np.all(np.isfinite(q) & (q > 0), axis=1)
    rows = [{**labels, "method": m, **_metrics(q[ok], y[ok])} for m, q in forecasts.items()]
    return pd.DataFrame(rows)


def cluster_bootstrap(per_well: dict[str, np.ndarray], clusters: np.ndarray,
                      reference: str, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Means of per-well scores (e.g. WIS, or 0/1 coverage) and their differences from
    ``reference``, with 95% percentile intervals from resampling whole clusters."""
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(clusters, return_inverse=True)
    n_c = len(codes)
    sums = {m: np.bincount(inv, weights=v, minlength=n_c) for m, v in per_well.items()}
    counts = np.bincount(inv, minlength=n_c).astype(float)
    picks = rng.integers(0, n_c, size=(n_boot, n_c))
    weight = np.stack([np.bincount(p, minlength=n_c) for p in picks])   # (n_boot, n_c)
    denom = weight @ counts
    boot = {m: (weight @ s) / denom for m, s in sums.items()}
    rows = []
    for m, v in per_well.items():
        d = boot[m] - boot[reference]
        rows.append({"method": m, "mean": float(np.mean(v)),
                     "ci_low": float(np.quantile(boot[m], 0.025)),
                     "ci_high": float(np.quantile(boot[m], 0.975)),
                     "diff_vs_" + reference: float(np.mean(v) - np.mean(per_well[reference])),
                     "diff_ci_low": float(np.quantile(d, 0.025)),
                     "diff_ci_high": float(np.quantile(d, 0.975)),
                     "clusters": n_c})
    return pd.DataFrame(rows)
