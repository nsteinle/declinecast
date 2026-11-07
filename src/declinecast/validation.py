"""Checks that the posterior is calibrated, on simulated wells the network never saw.

Following the paper's section VI C, every check uses fresh draws from the same joint
distribution as the training set (prior, lateral, downtime, history), so the true parameters are
known:

* **Simulation-based calibration (SBC).** For each held-out well, the rank of the true value
  among the posterior draws. A calibrated posterior gives uniform ranks for every parameter and
  every derived quantity. Uniformity is tested with a Kolmogorov-Smirnov test at a 5% family-wise
  level, Bonferroni-corrected over the quantities tested, and shown as a PP plot with the
  matching KS band.
* **Coverage.** The fraction of true values inside the central 50, 68, 90 and 95% credible
  intervals, with a 95% Clopper-Pearson interval on each fraction, at fixed histories of 6, 12,
  24 and 48 months, so a miscalibration at short histories cannot hide in an average.
* **Accuracy and contraction.** Spearman correlation of the posterior median with the truth,
  bias, mean absolute error, and the posterior-to-prior width ratio (how much each history
  teaches about each parameter).
* **Known answer.** A single well with chosen parameters, which the posterior must recover.
* **Predictive check.** Summary statistics of the real development wells against the same
  statistics of simulated wells drawn from the prior: the simulator must be able to produce what
  is observed.

The paper's warning applies here too: leakage outside the prior support is filtered, and its
rate is reported, because a high rate is a sign of a poorly trained flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from declinecast.quantities import DERIVED, derived
from declinecast.simulator import PARAMS, encode, simulate_series

__all__ = ["HISTORIES", "LEVELS", "HeldOut", "held_out", "posterior_ranks", "sbc_table",
           "ks_band", "coverage_table", "accuracy_table", "known_answer", "predictive_check",
           "validate"]

HISTORIES = (6, 12, 24, 48)
LEVELS = (0.50, 0.68, 0.90, 0.95)
QUANTITIES = PARAMS + tuple(name for name, _ in DERIVED)


@dataclass
class HeldOut:
    theta: np.ndarray       # (n, 6) true parameters
    x: np.ndarray           # (n, x_dim) network input
    history: np.ndarray     # (n,) months shown
    lateral_ft: np.ndarray


def held_out(prior, n: int, seed: int, history: int | None = None) -> HeldOut:
    """Fresh wells from the training joint. ``history=None`` draws it uniformly on [3, 60] as in
    training; an integer fixes it."""
    rng = np.random.default_rng(seed)
    lateral = prior.sample_lateral(n, rng)
    theta = prior.sample(n, rng, lateral)
    rate, _ = simulate_series(theta, rng, mask_prob=prior.sample_mask_prob(n, rng))
    h = (rng.integers(3, 61, size=n) if history is None else np.full(n, int(history)))
    return HeldOut(theta, encode(rate, h, lateral), h, lateral)


def _with_derived(theta: np.ndarray) -> np.ndarray:
    """Append the derived quantities (log10 barrels) as extra columns."""
    d = derived(theta)
    return np.concatenate([theta] + [d[name][..., None] for name, _ in DERIVED], axis=-1)


def posterior_ranks(model, data: HeldOut, n_draws: int = 500, seed: int = 0,
                    chunk: int = 64) -> dict:
    """Posterior draws summarized per well: the normalized rank of the truth (in [0, 1]), the
    posterior median and the central-interval bounds, for every parameter and derived quantity.
    """
    n = len(data.theta)
    q = sorted({0.5} | {round((1 - lv) / 2, 4) for lv in LEVELS}
               | {round((1 + lv) / 2, 4) for lv in LEVELS})
    rank = np.empty((n, len(QUANTITIES)))
    quant = np.empty((len(q), n, len(QUANTITIES)))
    acc = np.empty(n)
    jitter = np.random.default_rng(seed + 1)
    for s in range(0, n, chunk):
        draws, acc[s:s + chunk] = model.posterior(data.x[s:s + chunk], n_draws, seed=seed + s)
        full = _with_derived(draws)                              # (b, n_draws, 9)
        truth = _with_derived(data.theta[s:s + chunk])[:, None]  # (b, 1, 9)
        below = (full < truth).sum(1)
        ties = (full == truth).sum(1)
        # randomized ranks break ties and turn the discrete rank into a uniform on [0, 1]
        rank[s:s + chunk] = (below + jitter.random(below.shape) * (ties + 1)) / (n_draws + 1)
        quant[:, s:s + chunk] = np.quantile(full, q, axis=1)
    return {"rank": rank, "quantiles": dict(zip(q, quant, strict=True)), "acceptance": acc,
            "truth": _with_derived(data.theta), "n_draws": n_draws}


def ks_band(n: int, alpha: float = 0.05, n_tests: int = 1) -> float:
    """Half-width of the KS band for an empirical CDF of ``n`` uniform values, at a family-wise
    level ``alpha`` over ``n_tests`` tests (Bonferroni)."""
    return float(stats.kstwo.ppf(1.0 - alpha / n_tests, n))


def sbc_table(ranks: dict, alpha: float = 0.05) -> pd.DataFrame:
    """KS test of rank uniformity per quantity, Bonferroni-corrected over all quantities."""
    r = ranks["rank"]
    m = r.shape[1]
    rows = []
    for j, name in enumerate(QUANTITIES):
        res = stats.kstest(r[:, j], "uniform")
        rows.append({"quantity": name, "ks_statistic": res.statistic, "p_value": res.pvalue,
                     "p_bonferroni": min(res.pvalue * m, 1.0),
                     "passes": bool(res.pvalue * m >= alpha),
                     "mean_rank": float(r[:, j].mean())})
    out = pd.DataFrame(rows)
    out.attrs["band"] = ks_band(len(r), alpha, m)
    out.attrs["n"] = len(r)
    return out


def _clopper_pearson(k, n, alpha=0.05):
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def coverage_table(ranks: dict, history: int | None = None) -> pd.DataFrame:
    """Coverage of the central credible intervals, per quantity and level."""
    r = ranks["rank"]
    n = len(r)
    rows = []
    for j, name in enumerate(QUANTITIES):
        for lv in LEVELS:
            k = int((np.abs(r[:, j] - 0.5) <= lv / 2).sum())
            lo, hi = _clopper_pearson(k, n)
            rows.append({"history": history, "quantity": name, "level": lv,
                         "coverage": k / n, "ci_low": lo, "ci_high": hi, "n": n,
                         "consistent": bool(lo <= lv <= hi)})
    return pd.DataFrame(rows)


def accuracy_table(ranks: dict, prior_width: np.ndarray, history: int | None = None):
    """Spearman correlation, bias and mean absolute error of the posterior median, and the
    ratio of the 90% interval's width to the prior's (1 = learned nothing)."""
    med, truth = ranks["quantiles"][0.5], ranks["truth"]
    lo, hi = ranks["quantiles"][0.05], ranks["quantiles"][0.95]
    rows = []
    for j, name in enumerate(QUANTITIES):
        rows.append({"history": history, "quantity": name,
                     "spearman": float(stats.spearmanr(med[:, j], truth[:, j]).statistic),
                     "bias": float(np.mean(med[:, j] - truth[:, j])),
                     "mae": float(np.mean(np.abs(med[:, j] - truth[:, j]))),
                     "width_ratio": float(np.median(hi[:, j] - lo[:, j]) / prior_width[j])})
    return pd.DataFrame(rows)


def prior_widths(prior, n: int = 20000, seed: int = 99) -> np.ndarray:
    """Width of the central 90% of the prior, per quantity (the reference for contraction)."""
    rng = np.random.default_rng(seed)
    lateral = prior.sample_lateral(n, rng)
    full = _with_derived(prior.sample(n, rng, lateral))
    return np.quantile(full, 0.95, axis=0) - np.quantile(full, 0.05, axis=0)


KNOWN_THETA = (2.6, 0.4, 0.8, 1.5, float(np.log10(0.15)), 0.4)


def known_answer(model, theta=KNOWN_THETA, history: int = 48,
                 lateral_ft: float = 10_000.0, n_draws: int = 2000, seed: int = 3) -> dict:
    """Simulate one well with known parameters and report where the truth falls."""
    rng = np.random.default_rng(seed)
    theta = np.asarray(theta, float)[None]
    rate, _ = simulate_series(theta, rng, r0=np.array([[0.5]]))
    x = encode(rate, np.array([history]), np.array([lateral_ft]))
    draws, acc = model.posterior(x, n_draws, seed=seed)
    full, truth = _with_derived(draws[0]), _with_derived(theta)[0]
    lo, med, hi = np.quantile(full, [0.05, 0.5, 0.95], axis=0)
    return {"quantity": list(QUANTITIES), "truth": truth.tolist(), "median": med.tolist(),
            "p05": lo.tolist(), "p95": hi.tolist(),
            "inside_90": ((truth >= lo) & (truth <= hi)).tolist(),
            "acceptance": float(acc[0])}


# ---------------------------------------------------------------------- predictive check
def _summaries(rate: np.ndarray) -> pd.DataFrame:
    r = rate[:, :36]
    with np.errstate(all="ignore"):
        head = np.where(np.isfinite(r[:, :6]), r[:, :6], -1.0)
        peak_month = head.argmax(1)
        peak = np.nanmax(r[:, :6], axis=1)
        lr = np.log10(r)
        m12 = np.nanmedian(r[:, 10:14], axis=1)
        m33 = np.nanmedian(r[:, 30:36], axis=1)
        d = np.diff(lr[:, 6:36], axis=1)

        def slope(lo, hi):
            t = np.arange(lo, hi, dtype=float)
            y = lr[:, lo:hi]
            ok = np.isfinite(y)
            tm, ym = np.where(ok, t, np.nan), np.where(ok, y, np.nan)
            tc = tm - np.nanmean(tm, 1, keepdims=True)
            yc = ym - np.nanmean(ym, 1, keepdims=True)
            return np.nansum(tc * yc, 1) / np.nansum(tc ** 2, 1)

        dc = d - np.nanmean(d, 1, keepdims=True)
        lag1 = np.nanmean(dc[:, :-1] * dc[:, 1:], 1) / np.nanvar(d, 1)
        return pd.DataFrame({
            "peak_month": peak_month, "log10_peak": np.log10(peak),
            "drop_to_12": np.log10(m12 / peak), "drop_to_33": np.log10(m33 / peak),
            "slope_6_17": slope(6, 18), "slope_24_35": slope(24, 36),
            "roughness": np.nanstd(d, axis=1), "difference_lag1": lag1,
            "unobserved": 1.0 - np.isfinite(r).mean(1)})


def predictive_check(prior, real_rate: np.ndarray, n_sim: int = 30000,
                     seed: int = 1) -> pd.DataFrame:
    """Nine summaries of the first 36 months, real (development wells with at least 36 months)
    against simulated from the prior. ``outside`` is the share of real wells beyond the simulated
    1-99% range; a well-specified simulator keeps it near 2%."""
    rng = np.random.default_rng(seed)
    lateral = prior.sample_lateral(n_sim, rng)
    theta = prior.sample(n_sim, rng, lateral)
    sim, _ = simulate_series(theta, rng, mask_prob=prior.sample_mask_prob(n_sim, rng))
    real, simulated = _summaries(real_rate), _summaries(sim)
    rows = []
    for c in real:
        rv, sv = real[c].dropna(), simulated[c].dropna()
        lo, hi = np.quantile(sv, [0.01, 0.99])
        rows.append({"statistic": c, **{f"real_p{p}": float(np.quantile(rv, p / 100))
                                        for p in (5, 50, 95)},
                     **{f"sim_p{p}": float(np.quantile(sv, p / 100)) for p in (5, 50, 95)},
                     "outside": float(np.mean((rv < lo) | (rv > hi)))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- everything
def validate(model, n_sbc: int = 1000, n_per_history: int = 500, n_draws: int = 500,
             seed: int = 2026, log=print) -> dict:
    """Run SBC on wells with histories drawn as in training, then coverage and accuracy at each
    fixed history, and the known-answer test. Returns a dict of tables (DataFrames) and
    summaries; seeds differ from training's so every well is new."""
    prior = model.prior
    widths = prior_widths(prior)
    out = {}
    r = posterior_ranks(model, held_out(prior, n_sbc, seed), n_draws, seed=seed)
    out["sbc"] = sbc_table(r)
    out["sbc_ranks"] = pd.DataFrame(r["rank"], columns=list(QUANTITIES))
    cov, accu = [coverage_table(r, None)], [accuracy_table(r, widths, None)]
    accept = [r["acceptance"]]
    if log:
        log(f"SBC on {n_sbc} wells: {int(out['sbc'].passes.sum())}/{len(QUANTITIES)} pass")
    for i, h in enumerate(HISTORIES):
        rh = posterior_ranks(model, held_out(prior, n_per_history, seed + 10 + i, h), n_draws,
                             seed=seed + 100 * (i + 1))
        cov.append(coverage_table(rh, h))
        accu.append(accuracy_table(rh, widths, h))
        accept.append(rh["acceptance"])
        if log:
            log(f"history {h}: done")
    out["coverage"] = pd.concat(cov, ignore_index=True)
    out["accuracy"] = pd.concat(accu, ignore_index=True)
    a = np.concatenate(accept)
    out["acceptance"] = {"median": float(np.median(a)), "p01": float(np.quantile(a, 0.01)),
                         "min": float(a.min())}
    out["known_answer"] = known_answer(model)
    return out
