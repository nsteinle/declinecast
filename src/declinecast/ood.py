"""Flags for wells the model was not trained to describe.

A posterior is only as good as the simulator's coverage of the well in front of it. A well is
flagged when any of these holds:

* its lateral is outside the 1st to 99th percentile of development laterals;
* an observed monthly rate in its history is outside the 0.5 to 99.5% band of simulated rates for
  that month of life (prior predictive);
* fewer than 80% of the flow's raw posterior draws fall inside the prior's support (leakage, the
  paper's sign of a poorly conditioned posterior).

Flagged wells are still forecast; their scores are reported separately.
"""

from __future__ import annotations

import numpy as np

from declinecast.simulator import N_MONTHS, simulate_series

__all__ = ["RateBand", "rate_band", "flags"]

MIN_ACCEPTANCE = 0.8


class RateBand:
    def __init__(self, lo, hi, lateral_lo, lateral_hi):
        self.lo, self.hi = np.asarray(lo), np.asarray(hi)
        self.lateral_lo, self.lateral_hi = float(lateral_lo), float(lateral_hi)

    def to_dict(self):
        return {"log10_rate_lo": self.lo.round(4).tolist(),
                "log10_rate_hi": self.hi.round(4).tolist(),
                "lateral_lo": self.lateral_lo, "lateral_hi": self.lateral_hi}


def rate_band(prior, n: int = 50_000, seed: int = 7) -> RateBand:
    rng = np.random.default_rng(seed)
    lateral = prior.sample_lateral(n, rng)
    rate, _ = simulate_series(prior.sample(n, rng, lateral), rng,
                              mask_prob=prior.sample_mask_prob(n, rng))
    lr = np.log10(rate)
    lo, hi = np.nanquantile(lr, [0.005, 0.995], axis=0)
    q = np.asarray(prior.lateral_quantiles)
    return RateBand(lo, hi, q[1], q[99])


def flags(arrays, history: int, band: RateBand, acceptance=None) -> dict[str, np.ndarray]:
    """Boolean flags per well, and ``any``."""
    h = min(history, N_MONTHS)
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.log10(arrays.rate[:, :h])
    outside = np.isfinite(lr) & ((lr < band.lo[:h]) | (lr > band.hi[:h]))
    out = {"lateral": (arrays.lateral_ft < band.lateral_lo) | (arrays.lateral_ft > band.lateral_hi),
           "rate": outside.any(1)}
    if acceptance is not None:
        out["leakage"] = np.asarray(acceptance) < MIN_ACCEPTANCE
    out["any"] = np.logical_or.reduce(list(out.values()))
    return out
