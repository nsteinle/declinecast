"""Forecast arithmetic, baselines, scores and the clustered bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from declinecast import baselines
from declinecast.datasets import WellArrays
from declinecast.forecasting import LEVELS_Q, UptimeTable, forecast, uptime_table
from declinecast.quantities import DAYS_PER_MONTH, capacity_curve
from declinecast.scoring import cluster_bootstrap, covered, score_table, wis
from declinecast.simulator import N_MONTHS, simulate_series


class FixedPosterior:
    """Returns the given parameters as every posterior draw (a stand-in for the network). It
    sees one chunk of wells at a time, so tests pass ``chunk`` larger than the well count."""

    def __init__(self, theta):
        self.theta = np.asarray(theta, np.float32)

    def posterior(self, x, n, seed=0):
        rows = x.shape[0]
        return np.broadcast_to(self.theta[:rows, None, :], (rows, n, 6)).copy(), np.ones(rows)


def _arrays(rate, oil, available=None, lateral=10_000.0):
    n, k = rate.shape
    blank = np.array([""] * n, dtype=object)
    return WellArrays(api=np.array([f"w{i}" for i in range(n)]), rate=rate, oil=oil,
                      days=np.where(np.isfinite(rate), DAYS_PER_MONTH, 0.0),
                      available=np.full(n, k) if available is None else available,
                      lateral_ft=np.full(n, lateral), vintage=np.full(n, 2018),
                      first_operator=blank, loc_id=blank, township=blank)


def _noise_free_wells(theta, k=120):
    rate = np.concatenate([capacity_curve(theta, k)], axis=0)
    return _arrays(rate, rate * DAYS_PER_MONTH)


def test_forecast_reduces_to_the_curve_without_noise():
    theta = np.array([[2.6, 0.4, 0.8, 1.0, -9.0, 0.0], [2.2, 0.9, 1.2, 2.5, -9.0, 0.0]])
    a = _noise_free_wells(theta)
    up = UptimeTable({(12, 36): [1.0] * 101})
    f = forecast(FixedPosterior(theta), a, 12, 36, up, n_draws=50, chunk=10_000)
    q = f["quantiles"][36]
    expected = a.oil[:, :36].sum(1)
    np.testing.assert_allclose(q[:, LEVELS_Q.index(0.5)], expected, rtol=1e-6)
    np.testing.assert_allclose(q[:, 0], q[:, -1], rtol=1e-6)        # no spread at all
    np.testing.assert_allclose(f["observed"], a.oil[:, :12].sum(1))


def test_forecast_uptime_scales_only_the_future():
    theta = np.array([[2.6, 0.4, 0.8, 1.0, -9.0, 0.0]])
    a = _noise_free_wells(theta)
    up = UptimeTable({(12, 36): [0.5] * 101})
    q = forecast(FixedPosterior(theta), a, 12, 36, up, n_draws=20)["quantiles"][36]
    seen, future = a.oil[0, :12].sum(), a.oil[0, 12:36].sum()
    assert q[0, LEVELS_Q.index(0.5)] == pytest.approx(seen + 0.5 * future, rel=1e-6)


def test_forecast_intervals_cover_simulated_truth():
    """With the true parameters as the posterior, the predictive intervals for simulated wells
    must cover at their nominal rates (this checks the AR(1) continuation and the noise)."""
    rng = np.random.default_rng(5)
    n = 600
    theta = np.column_stack([rng.uniform(2.3, 2.9, n), rng.uniform(0.0, 0.8, n),
                             rng.uniform(0.3, 1.5, n), np.full(n, 1.0),
                             rng.uniform(-1.0, -0.4, n), rng.uniform(0.0, 0.8, n)])
    rate, _ = simulate_series(theta, rng, n_months=N_MONTHS, r0=np.full((n, 1), 0.5))
    a = _arrays(rate, rate * DAYS_PER_MONTH, available=np.full(n, N_MONTHS))
    up = UptimeTable({(12, 36): [1.0] * 101})
    q = forecast(FixedPosterior(theta), a, 12, 36, up, n_draws=400, seed=3,
                 chunk=10_000)["quantiles"][36]
    y = a.oil[:, :36].sum(1)
    for level in (0.5, 0.8, 0.9):
        assert covered(q, y, level).mean() == pytest.approx(level, abs=0.05)


def test_uptime_table_quantiles():
    rate = np.full((3, 40), 100.0)
    a = _arrays(rate, rate * DAYS_PER_MONTH)
    a.days[0, 12:36] = DAYS_PER_MONTH / 2
    t = uptime_table(a, [(12, 36)])
    q = np.asarray(t.quantiles[(12, 36)])
    assert q[0] == pytest.approx(0.5) and q[-1] == pytest.approx(1.0)
    assert UptimeTable.from_dict(t.to_dict()).quantiles[(12, 36)] == t.quantiles[(12, 36)]


def test_wis_and_coverage_by_hand():
    q = np.tile(np.array(LEVELS_Q) * 0 + 100.0, (1, 1))
    q[0, LEVELS_Q.index(0.5)] = 100.0
    y = np.array([100.0])
    assert wis(q, y)[0] == pytest.approx(0.0)
    # a point forecast 10x too high: WIS = (0.5 + sum(alpha/2 * 2/alpha)) * 1 dex / (K + 0.5)
    y2 = np.array([10.0])
    k = 5
    assert wis(q, y2)[0] == pytest.approx((0.5 + k) / (k + 0.5))
    lo, hi = 90.0, 110.0
    q3 = np.array([[lo] * 5 + [100.0] + [hi] * 5])
    assert covered(q3, np.array([95.0]), 0.8)[0] and not covered(q3, np.array([120.0]), 0.8)[0]


def test_score_table_uses_common_wells():
    q = np.array([[90.0] * 5 + [100.0] + [110.0] * 5] * 3)
    bad = q.copy()
    bad[0] = np.nan
    s = score_table({"a": q, "b": bad}, np.array([100.0, 105.0, 120.0]))
    assert (s.n == 2).all()


def test_cluster_bootstrap_reference_difference_is_zero():
    rng = np.random.default_rng(0)
    v = rng.random(200)
    out = cluster_bootstrap({"a": v, "b": v + 0.1}, np.repeat(np.arange(20), 10),
                            reference="a", n_boot=300)
    a, b = out.set_index("method").loc["a"], out.set_index("method").loc["b"]
    assert a.diff_ci_low == a.diff_ci_high == 0.0
    assert b.diff_ci_low == pytest.approx(0.1) and b.diff_ci_high == pytest.approx(0.1)
    assert out.clusters.iloc[0] == 20


def test_baselines_shapes_and_multiplier_logic():
    rng = np.random.default_rng(2)
    theta = np.column_stack([rng.uniform(2.3, 2.9, 300), rng.uniform(0.0, 0.8, 300),
                             rng.uniform(0.3, 1.5, 300), np.full(300, 1.0),
                             np.full(300, -1.0), np.full(300, 0.3)])
    rate, _ = simulate_series(theta, rng, n_months=72, r0=np.full((300, 1), 0.5))
    a = _arrays(rate, rate * DAYS_PER_MONTH)
    ref, tgt = a.take(np.arange(200)), a.take(np.arange(200, 300))
    m = baselines.multiplier(ref, tgt, 12, 36)
    assert m.shape == (100, len(LEVELS_Q)) and np.all(np.diff(m, axis=1) >= 0)
    t = baselines.type_curve(ref, tgt, 36)
    assert np.all(np.isfinite(t))
    up = UptimeTable({(12, 36): [1.0] * 101})
    ls = baselines.arps_ls(tgt, 12, 36, up, n_draws=100)
    y = tgt.oil[:, :36].sum(1)
    med = ls[:, LEVELS_Q.index(0.5)]
    assert np.median(np.abs(np.log10(med / y))) < 0.1


def test_recalibration_scales_only_the_predicted_oil():
    from declinecast.forecasting import future_factor

    theta = np.array([[2.6, 0.4, 0.8, 1.0, -9.0, 0.0]])
    a = _noise_free_wells(theta)
    up = UptimeTable({(12, 36): [1.0] * 101})
    model = FixedPosterior(theta)
    model.manifest = {"forecast_adjustment": {"factors": {"6-36": 0.8, "24-36": 1.0}}}
    f = future_factor(model, 12, 36)
    assert 0.8 < f < 1.0 and future_factor(model, 3, 36) == pytest.approx(0.8)
    out = forecast(model, a, 12, 36, up, n_draws=20, chunk=10_000)
    seen, future = a.oil[0, :12].sum(), a.oil[0, 12:36].sum()
    assert out["quantiles"][36][0, LEVELS_Q.index(0.5)] == pytest.approx(seen + f * future)
    raw = out["quantiles_unadjusted"][36][0, LEVELS_Q.index(0.5)]
    assert raw == pytest.approx(seen + future)
