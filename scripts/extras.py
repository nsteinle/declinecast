"""Checks reported in docs/FINDINGS.md that are not CLI commands: the parameter posteriors on real
development wells (the noise pile-up rule, what b learns at short histories), the terminal-decline
sensitivity, and the two ablations.

    python scripts/extras.py --out results/extras [--noiseless results/ablation_noiseless] \\
        [--no-lateral results/ablation_no_lateral] [--n100k results/ablation_n100k]

Every input is a development well (first produced 2014-2019). About 15 minutes on two CPU
cores with both ablations.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from declinecast import arps
from declinecast.datasets import load_panel, well_arrays
from declinecast.evaluation import evaluate
from declinecast.modelio import load_model
from declinecast.priors import LOWER, UPPER
from declinecast.quantities import capacity_curve
from declinecast.simulator import PARAMS, encode
from declinecast.validation import held_out, posterior_ranks, prior_widths

warnings.filterwarnings("ignore", category=RuntimeWarning)


def real_posteriors(model, arrays, histories=(6, 12, 24), n_draws=400, seed=0):
    """Posterior summaries of the six parameters on real wells, and the draws at the last
    history (for the terminal-decline check)."""
    widths = prior_widths(model.prior)[:6]
    rows, last = [], None
    for h in histories:
        ok = np.flatnonzero(arrays.available >= h)
        x = encode(arrays.rate[ok], np.full(len(ok), h), arrays.lateral_ft[ok])
        draws, acc = model.posterior(x, n_draws, seed=seed + h)
        lo, med, hi = np.quantile(draws, [0.05, 0.5, 0.95], axis=1)
        for j, name in enumerate(PARAMS):
            span = UPPER[j] - LOWER[j]
            rows.append({
                "history": h, "parameter": name, "wells": len(ok),
                "median_of_medians": float(np.median(med[:, j])),
                "p05_of_medians": float(np.quantile(med[:, j], 0.05)),
                "p95_of_medians": float(np.quantile(med[:, j], 0.95)),
                "width90_over_prior": float(np.median(hi[:, j] - lo[:, j]) / widths[j]),
                "share_median_in_top_5pct_of_support":
                    float(np.mean(med[:, j] > UPPER[j] - 0.05 * span)),
                "share_median_in_bottom_5pct_of_support":
                    float(np.mean(med[:, j] < LOWER[j] + 0.05 * span)),
                "median_acceptance": float(np.median(acc))})
        last = draws
    return pd.DataFrame(rows), last


def terminal_decline(draws):
    """EUR to 30 years, and cumulative to 36 and 60 months, at 5% and 10% effective terminal
    decline, from the same posterior draws."""
    d10 = -np.log(1 - 0.10)
    out = {}
    for name, months in (("cum_36", 36), ("cum_60", 60), ("eur_30y", 360)):
        ratio = np.concatenate([  # 25 wells at a time, to keep memory small
            np.median(capacity_curve(d, months, dmin=d10).sum(-1), axis=1)
            / np.median(capacity_curve(d, months).sum(-1), axis=1)
            for d in np.array_split(draws, max(len(draws) // 25, 1))])
        out[name] = {"median_ratio_10_to_5": float(np.median(ratio)),
                     "p05_ratio": float(np.quantile(ratio, 0.05)),
                     "p95_ratio": float(np.quantile(ratio, 0.95))}
    di, b = 10 ** draws[..., 1], draws[..., 2]
    ts = arps.switch_time(di, b) + draws[..., 3]
    out["share_of_draws_switching_before_month_60"] = float(np.mean(ts < 60))
    out["share_of_draws_switching_before_month_36"] = float(np.mean(ts < 36))
    return out


def ablation(model, panel, n_wells, seed):
    """Network-only scores at 12 months of history, 36-month horizon, on a development
    sample, and simulated-well coverage of the derived 36-month cumulative."""
    r = evaluate(model, panel, "development", histories=(12,), horizons=(36,), n_draws=300,
                 seed=seed, max_wells=n_wells, n_boot=200, log=None)
    s = r["scores"]
    s = s[(s.wells == "all") & (s.method == "network_unadjusted")
          if "network_unadjusted" in set(s.method) else
          (s.wells == "all") & (s.method == "network")].iloc[0].to_dict()
    th = model.manifest["training_history"]
    s["best_validation_loss"] = th["best_validation_loss"]
    s["kept_epoch"], s["epochs_run"] = th["kept_epoch"], th["epochs_run"]
    ranks = posterior_ranks(model, held_out(model.prior, 400, seed=91, history=12), 300)
    r36 = ranks["rank"][:, len(PARAMS)]  # capacity_cum_36 is the first derived column
    s["sim_coverage_90_capacity_cum_36_h12"] = float(np.mean(np.abs(r36 - 0.5) <= 0.45))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--noiseless", type=Path)
    ap.add_argument("--no-lateral", type=Path)
    ap.add_argument("--n100k", type=Path, help="the full model trained on 100,000 simulations")
    ap.add_argument("--wells", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    model = load_model(a.model_dir)
    dev = panel.wells[panel.wells.split == "development"]
    dev = dev.sample(min(a.wells, len(dev)), random_state=a.seed)
    arrays = well_arrays(panel, wells=dev)
    table, draws = real_posteriors(model, arrays, seed=a.seed)
    table.to_csv(a.out / "real_posteriors.csv", index=False)
    summary = {"wells": len(dev), "terminal_decline": terminal_decline(draws)}
    abl = {"full": ablation(model, panel, a.wells, a.seed)}
    for key, path in (("noiseless", a.noiseless), ("no_lateral", a.no_lateral),
                      ("n100k", a.n100k)):
        if path is not None:
            abl[key] = ablation(load_model(path), panel, a.wells, a.seed)
    summary["ablations"] = abl
    (a.out / "extras.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(table.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
