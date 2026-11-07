"""The forward model: a stochastic well simulator, and the one function that turns any monthly
series, simulated or real, into the network's input.

A simulated well, month by month from its first production (month 0):

* **Ramp.** Rate rises linearly from ``r0 * qi`` to ``qi`` until the peak at ``t_peak`` months
  (a continuous parameter in [0, 5]). In the development wells only 13.5% peak in their first
  month and 25% peak in months 3 to 5, so the peak's position is inferred, not assumed.
* **Decline.** From the peak, the modified hyperbolic Arps law (``declinecast.arps``).
* **Noise.** Log rate = log of the curve + an AR(1) process with standard deviation ``sigma``
  and lag-one correlation ``rho``, both inferred per well. Its innovations are Student-t with 4
  degrees of freedom, scaled to unit variance. Fitted residuals in the development wells have a
  median lag-one autocorrelation of 0.47 and a kurtosis of 46; independent Gaussian noise would
  make the cumulative-production intervals far too narrow.
* **Downtime.** Each month is unobserved with a per-well probability drawn from the development
  wells' distribution (shut-ins, months under 3 producing days, missing filings).

``encode`` is the only route into the network, for simulations and for real wells alike, so the
two are preprocessed identically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from declinecast import arps

__all__ = ["PARAMS", "N_MONTHS", "N_SCALARS", "N_CHANNELS", "ObsSpec", "OBS", "curve",
           "simulate_series", "encode", "simulate", "x_dim"]

PARAMS = ("log10_qi", "log10_di", "b", "t_peak", "log10_sigma", "rho")
N_MONTHS = 60    # the longest history the network reads
N_SCALARS = 3    # months of history, lateral length, fraction of history unobserved
N_CHANNELS = 3   # log rate, observed mask, month index
T_DOF = 4.0      # Student-t innovations
MIN_DAYS = 3     # a month with fewer producing days is treated as unobserved


@dataclass(frozen=True)
class ObsSpec:
    """Fixed scalings for the network's input (not learned, so real and simulated wells are
    scaled identically)."""

    rate_center: float = 2.5     # log10 bbl per producing day
    rate_scale: float = 0.5
    lateral_center: float = 4.0  # log10 ft (10,000 ft)
    lateral_scale: float = 0.15


OBS = ObsSpec()


def x_dim() -> int:
    return N_SCALARS + N_CHANNELS * N_MONTHS


def curve(k, log10_qi, log10_di, b, t_peak, r0, dmin=arps.DMIN_NOMINAL):
    """Noise-free rate (bbl per producing day) at calendar months ``k`` since first production.
    Parameter arrays have shape (n, 1) and ``k`` shape (m,), or both broadcast."""
    qi, di = 10.0 ** log10_qi, 10.0 ** log10_di
    k = np.asarray(k, float)
    after = arps.rate(np.maximum(k - t_peak, 0.0), qi, di, b, dmin)
    with np.errstate(divide="ignore", invalid="ignore"):
        ramp = qi * (r0 + (1.0 - r0) * np.where(t_peak > 0, k / t_peak, 1.0))
    return np.where(k < t_peak, ramp, after)


def _t_innovations(rng, shape):
    z = rng.standard_t(T_DOF, size=shape)
    return z * np.sqrt((T_DOF - 2.0) / T_DOF)


def ar1_noise(rng, sigma, rho, n_months: int, z=None):
    """AR(1) series with stationary standard deviation ``sigma`` and correlation ``rho``
    (arrays of shape (n,)), shape (n, n_months)."""
    sigma, rho = np.asarray(sigma, float)[:, None], np.asarray(rho, float)[:, None]
    if z is None:
        z = _t_innovations(rng, (sigma.shape[0], n_months))
    e = np.empty_like(z)
    e[:, 0] = z[:, 0]
    scale = np.sqrt(1.0 - rho[:, 0] ** 2)
    for t in range(1, n_months):
        e[:, t] = rho[:, 0] * e[:, t - 1] + scale * z[:, t]
    return sigma * e


def simulate_series(theta: np.ndarray, rng: np.random.Generator, n_months: int = N_MONTHS,
                    r0=None, mask_prob=None):
    """Noisy rates and an observed mask for parameter rows ``theta`` (n, 6).

    Returns ``(rate, observed)``, each (n, n_months); unobserved months have rate NaN."""
    theta = np.asarray(theta, float)
    n = theta.shape[0]
    lq, ld, b, tp, ls, rho = (theta[:, i:i + 1] for i in range(6))
    if r0 is None:
        r0 = rng.uniform(0.1, 0.9, size=(n, 1))
    k = np.arange(n_months, dtype=float)
    mu = curve(k[None, :], lq, ld, b, tp, np.asarray(r0, float).reshape(n, 1))
    noise = ar1_noise(rng, 10.0 ** ls[:, 0], rho[:, 0], n_months)
    rate = mu * np.exp(noise)
    if mask_prob is None:
        mask_prob = np.zeros(n)
    observed = rng.random((n, n_months)) >= np.asarray(mask_prob, float)[:, None]
    return np.where(observed, rate, np.nan), observed


def encode(rate: np.ndarray, history: np.ndarray, lateral_ft: np.ndarray) -> np.ndarray:
    """The network input for a batch of wells.

    ``rate``: (n, >= N_MONTHS) oil per producing day by calendar month since first production,
    NaN where the month is unobserved. ``history``: (n,) months of history available (only
    months ``k < history`` are shown). ``lateral_ft``: (n,). Returns float32 (n, x_dim()):
    three scalars, then the log-rate, mask and month-index channels, each N_MONTHS long."""
    rate = np.asarray(rate, float)[:, :N_MONTHS]
    n = rate.shape[0]
    history = np.asarray(history, int).reshape(n)
    k = np.arange(N_MONTHS)
    in_history = k[None, :] < history[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        observed = in_history & np.isfinite(rate) & (rate > 0)
        logr = np.where(observed, (np.log10(np.where(observed, rate, 1.0)) - OBS.rate_center)
                        / OBS.rate_scale, 0.0)
    unobserved = 1.0 - observed.sum(1) / np.maximum(np.minimum(history, N_MONTHS), 1)
    lat = (np.log10(np.asarray(lateral_ft, float).reshape(n)) - OBS.lateral_center) \
        / OBS.lateral_scale
    scalars = np.stack([np.minimum(history, N_MONTHS) / N_MONTHS, lat, unobserved], axis=1)
    position = np.broadcast_to(k / (N_MONTHS - 1), (n, N_MONTHS))
    out = np.concatenate([scalars, logr, observed.astype(float), position], axis=1)
    return out.astype(np.float32)


def simulate(prior, n: int, rng: np.random.Generator, min_history: int = 3,
             max_history: int = N_MONTHS, noise: bool = True):
    """Draw ``n`` (theta, x) pairs from the joint: lateral and downtime from the development
    wells, theta from the prior given the lateral, a history length uniform on
    [min_history, max_history]. With ``noise=False`` the noise and downtime are switched off (the
    paper's noiseless comparison). Returns ``theta`` (n, 6) and ``x`` (n, x_dim()), float32."""
    lateral = prior.sample_lateral(n, rng)
    theta = prior.sample(n, rng, lateral)
    mask_prob = prior.sample_mask_prob(n, rng) if noise else np.zeros(n)
    sim_theta = theta.copy()
    if not noise:
        sim_theta[:, 4] = -12.0  # sigma ~ 0
    rate, _ = simulate_series(sim_theta, rng, mask_prob=mask_prob)
    history = rng.integers(min_history, max_history + 1, size=n)
    return theta.astype(np.float32), encode(rate, history, lateral)
