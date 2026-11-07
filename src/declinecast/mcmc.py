"""A per-well MCMC baseline: the published Bayesian decline-curve approach, fit well by well.

Probabilistic decline-curve analysis by Markov chain Monte Carlo (Gong et al. 2011 and the Texas
A&M work that followed; FINDINGS section 12) fits each well's posterior separately. This module
does the same with **the same model the network was trained on**, so that the comparison
isolates the inference method:

* the same modified Arps curve with the ramp to the peak (``simulator.curve``), the same six
  parameters plus the ramp's starting fraction ``r0`` as a nuisance parameter;
* the same prior (initial rate given the lateral, the rest uniform on the same support, ``r0``
  uniform on [0.1, 0.9]);
* the same noise model: AR(1) log-rate noise with Student-t innovations (4 degrees of freedom,
  unit variance) and unobserved months. Across a gap of g months the transition is taken as
  Student-t with the AR(1) mean rho^g e and scale sigma sqrt(1 - rho^(2g)); a sum of t
  innovations is not exactly t, so this is the one approximation in the likelihood.

The sampler is the affine-invariant ensemble ("stretch move") sampler of Goodman and Weare
(2010), the algorithm of the ``emcee`` package, written here in NumPy and vectorized over wells
and walkers. Walkers start at the best of a few thousand prior draws.

``MCMCPosterior`` has the same ``posterior(x, n)`` method as a loaded network, and reads each
well back from the network's input, so ``forecasting.forecast`` and everything after it run
unchanged on the MCMC draws.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

from declinecast.priors import LOWER, UPPER
from declinecast.simulator import N_MONTHS, N_SCALARS, OBS, T_DOF, curve

__all__ = ["MCMCSettings", "MCMCPosterior", "decode", "log_posterior"]

R0_LOW, R0_HIGH = 0.1, 0.9
_LO = np.r_[LOWER, R0_LOW]
_HI = np.r_[UPPER, R0_HIGH]
_C = np.sqrt((T_DOF - 2.0) / T_DOF)  # Student-t scale that gives unit variance
_TCONST = gammaln((T_DOF + 1) / 2) - gammaln(T_DOF / 2) - 0.5 * np.log(T_DOF * np.pi)


@dataclass
class MCMCSettings:
    walkers: int = 32
    burn: int = 2000
    keep: int = 2000           # steps kept after burn-in (walkers x keep draws, then thinned)
    init_draws: int = 2000     # prior draws per well to start the walkers from
    stretch: float = 2.0       # the stretch move's scale a


def decode(x: np.ndarray):
    """Undo ``simulator.encode``: log rate (natural), observed mask, history and lateral."""
    x = np.asarray(x, float)
    h = np.rint(x[:, 0] * N_MONTHS).astype(int)
    lateral = 10.0 ** (x[:, 1] * OBS.lateral_scale + OBS.lateral_center)
    series = x[:, N_SCALARS:].reshape(len(x), 3, N_MONTHS)
    mask = series[:, 1] > 0.5
    log10r = series[:, 0] * OBS.rate_scale + OBS.rate_center
    return np.where(mask, log10r * np.log(10.0), 0.0), mask, h, lateral


def _tlogpdf(z):
    return _TCONST - (T_DOF + 1) / 2 * np.log1p(z * z / T_DOF)


def log_posterior(theta, y, mask, prior, lateral):
    """Log posterior (up to a constant) for parameters ``theta`` (W, M, 7) of W wells with M
    walkers each; ``y`` and ``mask`` (W, 60), ``lateral`` (W,). Returns (W, M)."""
    inside = np.all((theta >= _LO) & (theta <= _HI), axis=-1)
    th = np.clip(theta, _LO, _HI)
    k = np.arange(N_MONTHS, dtype=float)
    with np.errstate(all="ignore"):
        mu = np.log(curve(k, *(th[..., i:i + 1] for i in range(4)), th[..., 6:7]))  # (W,M,60)
    sigma = 10.0 ** th[..., 4]
    rho = th[..., 5]
    e = y[:, None, :] - mu
    # previous observed month for each month (index, or -1), per well
    idx = np.where(mask, np.arange(N_MONTHS)[None, :], -1)
    prev = np.maximum.accumulate(np.concatenate([np.full((len(y), 1), -1), idx[:, :-1]], 1), 1)
    gap = np.where(prev >= 0, np.arange(N_MONTHS)[None, :] - prev, 0)          # (W, 60)
    e_prev = np.take_along_axis(e, np.maximum(prev, 0)[:, None, :].repeat(e.shape[1], 1), 2)
    rg = rho[..., None] ** gap[:, None, :]
    first = (prev < 0)[:, None, :]
    mean = np.where(first, 0.0, rg * e_prev)
    sd = sigma[..., None] * np.where(first, 1.0, np.sqrt(np.maximum(1.0 - rg * rg, 1e-12)))
    z = (e - mean) / (sd * _C)
    ll = np.where(mask[:, None, :], _tlogpdf(z) - np.log(sd * _C), 0.0).sum(-1)
    mean_q = prior.qi_mean(lateral)[:, None]
    lp = -0.5 * ((th[..., 0] - mean_q) / prior.qi_sd) ** 2
    out = ll + lp
    return np.where(inside & np.isfinite(out), out, -np.inf)


class MCMCPosterior:
    """Per-well MCMC with the network's interface. ``manifest`` is empty, so forecasts made
    with it carry no recalibration. ``seconds`` accumulates the sampling wall time."""

    def __init__(self, prior, settings: MCMCSettings | None = None):
        self.prior = prior
        self.settings = settings or MCMCSettings()
        self.manifest = {}
        self.seconds = 0.0
        self.wells = 0
        self.acceptance = []

    def posterior(self, x, n: int = 500, seed: int = 0):
        s = self.settings
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        y, mask, _, lateral = decode(x)
        w, m, d = len(y), s.walkers, len(_LO)
        # start: the best walkers among prior draws
        cand = np.empty((w, s.init_draws, d))
        for i in range(w):
            lat = np.full(s.init_draws, lateral[i])
            cand[i, :, :6] = self.prior.sample(s.init_draws, rng, lat)
        cand[..., 6] = rng.uniform(R0_LOW, R0_HIGH, (w, s.init_draws))
        lp_c = log_posterior(cand, y, mask, self.prior, lateral)
        best = np.argsort(-lp_c, axis=1)[:, :m]
        pos = np.take_along_axis(cand, best[..., None], 1)
        pos = np.clip(pos + 1e-3 * (_HI - _LO) * rng.standard_normal(pos.shape), _LO, _HI)
        lp = log_posterior(pos, y, mask, self.prior, lateral)
        half = m // 2
        kept = []
        accepted = 0
        for step in range(s.burn + s.keep):
            for a, b in ((slice(0, half), slice(half, m)), (slice(half, m), slice(0, half))):
                cur, other = pos[:, a], pos[:, b]
                nh = cur.shape[1]
                partner = other[:, rng.integers(0, other.shape[1], nh)]
                zst = ((s.stretch - 1.0) * rng.random((w, nh)) + 1.0) ** 2 / s.stretch
                prop = partner + zst[..., None] * (cur - partner)
                lp_new = log_posterior(prop, y, mask, self.prior, lateral)
                log_acc = (d - 1) * np.log(zst) + lp_new - lp[:, a]
                ok = np.log(rng.random((w, nh))) < log_acc
                pos[:, a] = np.where(ok[..., None], prop, cur)
                lp[:, a] = np.where(ok, lp_new, lp[:, a])
                if step >= s.burn:
                    accepted += ok.sum(1)
            if step >= s.burn:
                kept.append(pos[..., :6].copy())
        chain = np.stack(kept, 1).reshape(w, -1, 6)                 # (W, keep*M, 6)
        pick = np.stack([rng.choice(chain.shape[1], n, replace=False) for _ in range(w)])
        draws = np.take_along_axis(chain, pick[..., None], 1).astype(np.float32)
        acc = accepted / (s.keep * m)
        self.seconds += time.perf_counter() - t0
        self.wells += w
        self.acceptance.extend(acc.tolist())
        return draws, np.ones(w)          # all draws are inside the support by construction
