"""The modified hyperbolic Arps decline, in NumPy (simulation, forecasting) and in closed form for
the cumulative (derived quantities).

The rate follows the hyperbolic law

    q(t) = qi (1 + b Di t)^(-1/b)

until its instantaneous nominal decline D(t) = Di / (1 + b Di t) falls to the terminal decline
Dmin, then declines exponentially at Dmin. Without the switch, b >= 1 gives an unbounded
cumulative; in the development wells 24% of per-well fits have b > 1 (FINDINGS section 2).

Time is in months, Di and Dmin are nominal rates per year, and rates are barrels of oil per
producing day.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["DMIN_EFFECTIVE", "DMIN_NOMINAL", "switch_time", "rate", "cumulative"]

#: Terminal decline: 5% per year, effective. Ryder Scott's modified-Arps example uses 5%, and the
#: same source warns that 10% or more can under-predict EUR with high b (Reservoir Solutions,
#: Sept.-Nov. 2011, quoting S. Wilson and J. Lee). FINDINGS section 6 reports the sensitivity.
DMIN_EFFECTIVE = 0.05
DMIN_NOMINAL = -math.log(1.0 - DMIN_EFFECTIVE)  # 0.0513 per year


def switch_time(di, b, dmin=DMIN_NOMINAL):
    """Months after the peak at which the hyperbola's decline reaches ``dmin`` (0 if it starts
    below it). ``di`` and ``dmin`` are nominal rates per year."""
    di_m, dmin_m = np.asarray(di, float) / 12.0, dmin / 12.0
    b = np.asarray(b, float)
    return np.where(di_m > dmin_m, (di_m / dmin_m - 1.0) / (b * di_m), 0.0)


def rate(t, qi, di, b, dmin=DMIN_NOMINAL):
    """Rate at months ``t`` after the peak (t >= 0). Parameters broadcast against ``t``."""
    t = np.asarray(t, float)
    qi, di, b = (np.asarray(v, float) for v in (qi, di, b))
    di_m, dmin_m = di / 12.0, dmin / 12.0
    ts = switch_time(di, b, dmin)
    hyp = qi * np.power(1.0 + b * di_m * np.minimum(t, ts), -1.0 / b)
    return np.where(t <= ts, hyp, hyp * np.exp(-dmin_m * (t - ts)))


def _hyp_cum(t, qi, di_m, b):
    """Integral of the hyperbola from 0 to t (rate x months), b in (0, 2]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        near_one = np.abs(b - 1.0) < 1e-6
        general = qi / (di_m * (1.0 - b)) * (1.0 - np.power(1.0 + b * di_m * t, 1.0 - 1.0 / b))
        harmonic = qi / di_m * np.log1p(di_m * t)
    return np.where(near_one, harmonic, general)


def cumulative(t, qi, di, b, dmin=DMIN_NOMINAL):
    """Closed-form integral of ``rate`` from 0 to ``t`` months, in barrels per producing day x
    months. Multiply by days per month (and uptime) for barrels."""
    t = np.asarray(t, float)
    qi, di, b = (np.asarray(v, float) for v in (qi, di, b))
    di_m, dmin_m = di / 12.0, dmin / 12.0
    ts = switch_time(di, b, dmin)
    head = _hyp_cum(np.minimum(t, ts), qi, di_m, b)
    q_s = qi * np.power(1.0 + b * di_m * ts, -1.0 / b)
    tail = q_s / dmin_m * (1.0 - np.exp(-dmin_m * np.maximum(t - ts, 0.0)))
    return head + np.where(t > ts, tail, 0.0)
