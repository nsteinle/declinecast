"""The forward model against analytic results."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from declinecast import arps
from declinecast.quantities import DAYS_PER_MONTH, capacity_cumulative, derived
from declinecast.simulator import (
    N_MONTHS,
    ar1_noise,
    curve,
    encode,
    simulate_series,
    x_dim,
)


@pytest.mark.parametrize("b", [0.3, 1.0, 1.6])
def test_cumulative_is_the_integral_of_the_rate(b):
    qi, di = 500.0, 3.0
    for t in (5.0, 60.0, 360.0):
        num, _ = quad(lambda s: float(arps.rate(s, qi, di, b)), 0, t, limit=400)
        assert arps.cumulative(t, qi, di, b) == pytest.approx(num, rel=1e-6)


def test_terminal_decline_switch():
    qi, di, b = 500.0, 3.0, 1.4
    ts = float(arps.switch_time(di, b))
    # the hyperbola's own decline at the switch equals Dmin
    d_at = di / (1 + b * di * ts / 12)
    assert d_at == pytest.approx(arps.DMIN_NOMINAL, rel=1e-9)
    # continuous at the switch, then exponential at Dmin: 5% per year effective
    eps = 1e-6
    assert arps.rate(ts - eps, qi, di, b) == pytest.approx(arps.rate(ts + eps, qi, di, b))
    r1, r2 = arps.rate(ts + 12, qi, di, b), arps.rate(ts + 24, qi, di, b)
    assert 1 - r2 / r1 == pytest.approx(arps.DMIN_EFFECTIVE, rel=1e-9)
    # b >= 1 still gives a finite EUR
    assert np.isfinite(arps.cumulative(1e5, qi, di, b))


def test_curve_ramp_and_peak():
    k = np.arange(10.0)
    c = curve(k[None], np.array([[2.5]]), np.array([[0.5]]), np.array([[0.8]]),
              np.array([[2.0]]), 0.4)[0]
    qi = 10 ** 2.5
    assert c[0] == pytest.approx(0.4 * qi)
    assert c[1] == pytest.approx(0.7 * qi)
    assert c[2] == pytest.approx(qi)
    assert np.all(np.diff(c[2:]) < 0)
    # t_peak = 0 starts at the peak
    c0 = curve(k[None], np.array([[2.5]]), np.array([[0.5]]), np.array([[0.8]]),
               np.array([[0.0]]), 0.4)[0]
    assert c0[0] == pytest.approx(qi)


def test_ar1_noise_has_the_stated_scale_and_correlation():
    rng = np.random.default_rng(0)
    n = 4000
    e = ar1_noise(rng, np.full(n, 0.3), np.full(n, 0.6), 200)[:, 20:]
    assert e.std() == pytest.approx(0.3, rel=0.03)
    lag1 = np.mean(e[:, 1:] * e[:, :-1]) / e.var()
    assert lag1 == pytest.approx(0.6, abs=0.02)


def test_simulate_series_masks_and_noise():
    rng = np.random.default_rng(1)
    theta = np.tile([2.6, 0.4, 0.8, 1.0, np.log10(0.2), 0.3], (500, 1))
    rate, obs = simulate_series(theta, rng, mask_prob=np.full(500, 0.25))
    assert rate.shape == (500, N_MONTHS)
    assert np.isnan(rate[~obs]).all() and np.isfinite(rate[obs]).all()
    assert (~obs).mean() == pytest.approx(0.25, abs=0.01)


def test_encode_shows_only_the_history():
    rate = np.full((2, N_MONTHS), 300.0)
    rate[1, 3] = np.nan
    x = encode(rate, np.array([6, 6]), np.array([10_000.0, 10_000.0]))
    assert x.shape == (2, x_dim()) and x.dtype == np.float32
    mask = x[:, 3 + N_MONTHS:3 + 2 * N_MONTHS]
    assert mask[0].sum() == 6 and mask[1].sum() == 5
    assert x[0, 0] == pytest.approx(6 / N_MONTHS) and x[0, 1] == pytest.approx(0.0)
    assert x[1, 2] == pytest.approx(1 / 6)  # unobserved fraction
    # later months are hidden, whatever they hold
    rate2 = rate.copy()
    rate2[:, 6:] = 1.0
    assert np.array_equal(encode(rate2, np.array([6, 6]), np.array([1e4, 1e4])), x)


def test_derived_quantities_are_consistent():
    theta = np.array([[2.6, 0.4, 0.8, 1.0, -1.0, 0.3]])
    d = derived(theta)
    assert 10 ** d["capacity_cum_36"][0] == pytest.approx(capacity_cumulative(theta, 36)[0])
    assert d["capacity_cum_36"][0] < d["capacity_cum_60"][0] < d["eur_30y"][0]
    # month sums of the curve: compare with a direct sum
    k = np.arange(36.0)
    direct = curve(k[None], *(theta[:, i:i + 1] for i in range(4)), 0.5).sum() * DAYS_PER_MONTH
    assert capacity_cumulative(theta, 36)[0] == pytest.approx(direct)
