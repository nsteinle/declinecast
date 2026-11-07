"""Quantities derived from the decline parameters: the noise-free capacity cumulative and EUR.

Single parameters such as ``b`` are poorly determined by a short history (FINDINGS section 3),
but the cumulative production they imply is well determined. That makes the derived quantities
the ones to calibrate and report, as the chirp mass was in the paper's gravitational-wave
analogue.

All quantities are discrete monthly sums of the noise-free curve, in barrels, at full uptime
(``DAYS_PER_MONTH`` producing days every month), matching how real cumulatives are summed from
monthly reports. The ramp before the peak needs its starting fraction ``r0``, which is not a
parameter; it is set to its prior mean, 0.5. Any fixed function of the parameters is a valid
target for the calibration checks, and the ramp is at most five months of a 360-month sum.
"""

from __future__ import annotations

import numpy as np

from declinecast.arps import DMIN_NOMINAL
from declinecast.simulator import curve

__all__ = ["DAYS_PER_MONTH", "R0_MEAN", "capacity_curve", "capacity_cumulative", "eur",
           "DERIVED"]

DAYS_PER_MONTH = 365.25 / 12.0
R0_MEAN = 0.5
#: The derived quantities the validation reports, as (name, months).
DERIVED = (("capacity_cum_36", 36), ("capacity_cum_60", 60), ("eur_30y", 360))


def capacity_curve(theta: np.ndarray, months: int, r0: float = R0_MEAN,
                   dmin: float = DMIN_NOMINAL) -> np.ndarray:
    """Noise-free rate (bbl per producing day) at months 0 .. months-1, for parameter rows
    ``theta`` of shape (..., 6). Returns shape (..., months). ``dmin`` is the nominal terminal
    decline per year (for the sensitivity check in FINDINGS)."""
    theta = np.asarray(theta, float)
    k = np.arange(months, dtype=float)
    lq, ld, b, tp = (theta[..., i:i + 1] for i in range(4))
    return curve(k, lq, ld, b, tp, r0, dmin)


def capacity_cumulative(theta: np.ndarray, months: int, r0: float = R0_MEAN) -> np.ndarray:
    """Barrels in months 0 .. months-1 at full uptime and no noise. Shape ``theta.shape[:-1]``."""
    return capacity_curve(theta, months, r0).sum(-1) * DAYS_PER_MONTH


def eur(theta: np.ndarray, years: int = 30) -> np.ndarray:
    """Noise-free cumulative to ``years`` years (the modified-Arps EUR at a 30-year limit)."""
    return capacity_cumulative(theta, 12 * years)


def derived(theta: np.ndarray, block: int = 8192) -> dict[str, np.ndarray]:
    """Every quantity in ``DERIVED`` for parameter rows ``theta``, as log10 barrels. Computed
    in blocks of ``block`` parameter vectors, so memory stays near block x 360 x 8 bytes per
    temporary whatever the number of draws."""
    theta = np.asarray(theta, float)
    flat = theta.reshape(-1, theta.shape[-1])
    longest = max(m for _, m in DERIVED)
    out = {name: np.empty(len(flat)) for name, _ in DERIVED}
    for s in range(0, len(flat), block):
        c = np.cumsum(capacity_curve(flat[s:s + block], longest), axis=-1) * DAYS_PER_MONTH
        for name, m in DERIVED:
            out[name][s:s + block] = np.log10(c[:, m - 1])
    return {k: v.reshape(theta.shape[:-1]) for k, v in out.items()}
