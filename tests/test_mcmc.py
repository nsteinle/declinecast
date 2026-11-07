"""The per-well MCMC baseline: decoding, the likelihood, and recovery on simulated wells."""

from __future__ import annotations

import numpy as np

from declinecast.mcmc import MCMCPosterior, MCMCSettings, decode, log_posterior
from declinecast.simulator import encode, simulate_series
from declinecast.training import development_prior


def test_decode_inverts_encode():
    rng = np.random.default_rng(0)
    rate = np.exp(rng.normal(5, 0.3, (3, 60)))
    rate[1, 4] = np.nan
    x = encode(rate, np.array([6, 12, 60]), np.array([7000.0, 10000.0, 12000.0]))
    y, mask, h, lateral = decode(x)
    assert h.tolist() == [6, 12, 60]
    np.testing.assert_allclose(lateral, [7000, 10000, 12000], rtol=1e-4)
    assert mask[0].sum() == 6 and mask[1].sum() == 11
    np.testing.assert_allclose(y[mask], np.log(rate[:, :60])[mask], atol=1e-5)


def test_log_posterior_support(panel):
    prior = development_prior(panel)
    theta = np.array([[[2.6, 0.4, 0.8, 1.0, -1.0, 0.3, 0.5],
                       [2.6, 0.4, 2.5, 1.0, -1.0, 0.3, 0.5]]])   # b outside its support
    rate, _ = simulate_series(theta[0, :1, :6], np.random.default_rng(1),
                              r0=np.array([[0.5]]))
    y, mask, _, lat = decode(encode(rate, np.array([24]), np.array([10000.0])))
    lp = log_posterior(theta, y, mask, prior, lat)
    assert np.isfinite(lp[0, 0]) and lp[0, 1] == -np.inf


def test_mcmc_recovers_a_simulated_well(panel):
    prior = development_prior(panel)
    truth = np.array([[2.6, 0.4, 0.8, 1.5, np.log10(0.15), 0.4]])
    rate, _ = simulate_series(truth, np.random.default_rng(3), r0=np.array([[0.5]]))
    x = encode(rate, np.array([48]), np.array([10000.0]))
    mc = MCMCPosterior(prior, MCMCSettings(walkers=24, burn=600, keep=600, init_draws=1000))
    draws, acc = mc.posterior(x, 400, seed=0)
    assert draws.shape == (1, 400, 6) and acc.tolist() == [1.0]
    lo, hi = np.quantile(draws[0], [0.01, 0.99], axis=0)
    assert lo[0] <= truth[0, 0] <= hi[0]                 # initial rate
    assert 0.15 < np.median(mc.acceptance) < 0.7
