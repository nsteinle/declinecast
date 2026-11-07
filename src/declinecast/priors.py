"""The prior over the six decline parameters, and the empirical distributions of the two inputs
that are not parameters (lateral length and downtime), all set from development wells only.

* log10 qi: normal given the lateral, truncated to [1.3, 3.7]. Lateral length explains 22.5% of
  the variance of peak rate in development wells; the width is the fitted residual sd widened
  1.7 times, so the prior is not tighter than the data.
* log10 Di: uniform on [log10 0.2, log10 40] per year, which covers the 2nd to 98th percentiles
  of per-well fits (0.6 to 17.5) with room on both sides.
* b: uniform on [0.01, 2.0]; fits run from 0.01 to 1.57 at the 2nd and 98th percentiles.
* t_peak: uniform on [0, 5] months; the peak falls in months 0 to 5 for development wells.
* log10 sigma: uniform on [log10 0.04, log10 1.2]; fitted residual sd runs 0.09 to 0.76 at the
  5th and 95th percentiles.
* rho: uniform on [0, 0.95]; fitted lag-one autocorrelation runs 0.15 to 0.74 at the 10th and
  90th.

This is the decline-curve counterpart of the paper's joint conditional prior: there the donor
mass was drawn given the accretor mass; here the initial rate is drawn given the lateral.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from declinecast.simulator import PARAMS

__all__ = ["Prior", "LOWER", "UPPER"]

LOWER = np.array([1.3, np.log10(0.2), 0.01, 0.0, np.log10(0.04), 0.0])
UPPER = np.array([3.7, np.log10(40.0), 2.0, 5.0, np.log10(1.2), 0.95])


@dataclass
class Prior:
    """The prior and the input distributions, with every number needed to rebuild it."""

    qi_intercept: float          # log10 qi at a 10,000 ft lateral
    qi_slope: float              # per unit log10(lateral / 10,000 ft)
    qi_sd: float                 # prior sd of log10 qi given the lateral
    lateral_quantiles: list[float] = field(default_factory=list)   # 101 quantiles, ft
    mask_quantiles: list[float] = field(default_factory=list)      # 101 quantiles of the
                                                                     # per-well unobserved fraction
    names: tuple = PARAMS

    # ------------------------------------------------------------------ inputs
    @staticmethod
    def _from_quantiles(q, n, rng):
        q = np.asarray(q, float)
        return np.interp(rng.random(n), np.linspace(0, 1, len(q)), q)

    def sample_lateral(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self._from_quantiles(self.lateral_quantiles, n, rng)

    def sample_mask_prob(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.clip(self._from_quantiles(self.mask_quantiles, n, rng), 0.0, 0.9)

    # ------------------------------------------------------------------ parameters
    def qi_mean(self, lateral_ft) -> np.ndarray:
        return self.qi_intercept + self.qi_slope * np.log10(np.asarray(lateral_ft, float) / 1e4)

    def sample(self, n: int, rng: np.random.Generator, lateral_ft) -> np.ndarray:
        """Draw (n, 6) parameters given the laterals."""
        mean = self.qi_mean(lateral_ft)
        lq = np.empty(n)
        todo = np.arange(n)
        while todo.size:  # truncated normal by rejection; the truncation removes < 1%
            draw = rng.normal(mean[todo], self.qi_sd)
            ok = (draw >= LOWER[0]) & (draw <= UPPER[0])
            lq[todo[ok]] = draw[ok]
            todo = todo[~ok]
        rest = rng.uniform(LOWER[1:], UPPER[1:], size=(n, 5))
        return np.column_stack([lq, rest])

    def log_prob(self, theta, lateral_ft) -> np.ndarray:
        """Log density of (n, 6) parameters given the laterals (-inf outside the support)."""
        from scipy.stats import truncnorm

        theta = np.asarray(theta, float)
        mean = self.qi_mean(lateral_ft)
        a, b = (LOWER[0] - mean) / self.qi_sd, (UPPER[0] - mean) / self.qi_sd
        lp = truncnorm.logpdf(theta[:, 0], a, b, loc=mean, scale=self.qi_sd)
        inside = np.all((theta[:, 1:] >= LOWER[1:]) & (theta[:, 1:] <= UPPER[1:]), axis=1)
        return np.where(inside, lp - np.log(UPPER[1:] - LOWER[1:]).sum(), -np.inf)

    def support(self):
        """A box over the support, for sbi (sampling and leakage only; NPE trains on the joint
        draws, so the box's own density is never used)."""
        from sbi.utils import BoxUniform

        return BoxUniform(torch.as_tensor(LOWER, dtype=torch.float32),
                          torch.as_tensor(UPPER, dtype=torch.float32))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["names"] = list(self.names)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Prior:
        d = dict(d)
        d["names"] = tuple(d.get("names", PARAMS))
        return cls(**d)


def fit_prior(arrays, qi_sd_inflation: float = 1.7, months: int = 36) -> Prior:
    """Set the prior's data-driven parts from development wells (``arrays`` must hold only
    development wells): the lateral-to-peak-rate regression, the lateral distribution and the
    per-well fraction of unobserved months in the first ``months`` months."""
    rate, lat = arrays.rate, arrays.lateral_ft
    head = np.where(np.isfinite(rate[:, :6]), rate[:, :6], -np.inf)
    peak = np.where(np.isfinite(head).any(1), head.max(1), np.nan)  # NaN: no observed month
    ok = np.isfinite(peak) & (lat > 0)
    x, y = np.log10(lat[ok] / 1e4), np.log10(peak[ok])
    slope, intercept = np.polyfit(x, y, 1)
    resid_sd = float(np.std(y - (intercept + slope * x)))
    span = np.minimum(arrays.available, months)
    obs = np.isfinite(rate[:, :months]) & (np.arange(months)[None, :] < span[:, None])
    unobserved = 1.0 - obs.sum(1) / span
    q = np.linspace(0, 1, 101)
    return Prior(qi_intercept=float(intercept), qi_slope=float(slope),
                 qi_sd=float(resid_sd * qi_sd_inflation),
                 lateral_quantiles=np.quantile(lat[lat > 0], q).round(1).tolist(),
                 mask_quantiles=np.quantile(unobserved, q).round(5).tolist())
