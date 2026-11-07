"""The per-well MCMC baseline on the locked test wells, against the network's saved forecasts.

    python scripts/mcmc_baseline.py --test results/test --out results/mcmc

(A development score folder works the same way, for a quick check without the test wells.)

Post-processing only: the network's test forecasts are read from the folder written by the one
test scoring (``declinecast score --split test --unlock-test``), never recomputed. MCMC
(``declinecast.mcmc``: the same curve, prior and noise model as the network's simulator) is run
on the same wells and histories, with the same forecast code, the same seeds for the noise and
uptime draws, and the same scores, so the two differ only in how the posterior is computed.
Also times both posteriors on this machine: MCMC vectorized over 64 wells at a time (as in the
run) and one well at a time. About 40 minutes on two CPU cores.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from declinecast.datasets import load_panel, well_arrays
from declinecast.forecasting import LEVELS_Q, forecast, uptime_table
from declinecast.mcmc import MCMCPosterior
from declinecast.modelio import load_model
from declinecast.scoring import INTERVALS, cluster_bootstrap, covered, score_table, wis
from declinecast.simulator import encode

warnings.filterwarnings("ignore", category=RuntimeWarning)
QCOLS = [f"q{lv:g}" for lv in LEVELS_Q]
COMPARED = ("network", "network_unadjusted", "mcmc", "arps_ls", "multiplier")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, required=True,
                    help="a score folder: the test folder for the documented run, or any "
                         "development folder for a quick check")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--max-wells", type=int, help="a random subset (for a quick run)")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((a.test / "score.json").read_text())
    if meta["as_reported"]:
        raise SystemExit("--test must be a final-record score folder")
    split = meta["split"]                       # "test" for the documented run
    histories, horizons = meta["histories"], meta["horizons"]
    n_draws, seed = meta["n_draws"], meta["seed"]
    panel, model = load_panel(), load_model(a.model_dir)
    wells = panel.wells
    dev = well_arrays(panel, wells=wells[wells.split == "development"])
    test = well_arrays(panel, wells=wells[wells.split == split])
    keep = np.flatnonzero(test.available >= min(horizons))   # as in evaluation.evaluate
    if meta.get("max_wells") and len(keep) > meta["max_wells"]:   # the folder's own sample
        keep = np.sort(np.random.default_rng(seed).choice(keep, meta["max_wells"],
                                                          replace=False))
    if a.max_wells and len(keep) > a.max_wells:
        keep = np.sort(np.random.default_rng(seed).choice(keep, a.max_wells, replace=False))
    test = test.take(keep)
    uptime = uptime_table(dev, [(h, t) for h in histories for t in horizons])

    mc = MCMCPosterior(model.prior)
    records = []
    for h in histories:
        t0 = time.time()
        # the same seed as the test scoring, so the noise and uptime draws are shared
        f = forecast(mc, test, h, horizons, uptime, n_draws=n_draws, seed=seed + h)
        for t in horizons:
            ok = test.available >= t
            rec = pd.DataFrame(f["quantiles"][t][ok], columns=QCOLS)
            rec.insert(0, "api", test.api[ok])
            rec["history"], rec["horizon"], rec["method"] = h, t, "mcmc"
            records.append(rec)
        print(f"history {h}: {len(test.api):,} wells, {time.time() - t0:.0f} s", flush=True)
    mcmc = pd.concat(records, ignore_index=True)
    mcmc.to_csv(a.out / "mcmc_forecasts.csv", index=False)

    # time the network's posterior on the same machine, same draws, a sample of the same wells
    rows = np.random.default_rng(1).choice(len(test.api), min(256, len(test.api)), replace=False)
    x = encode(test.rate[rows], np.full(len(rows), 12), test.lateral_ft[rows])
    t0 = time.perf_counter()
    model.posterior(x, n_draws, seed=1)
    nn_seconds = (time.perf_counter() - t0) / len(rows)
    # and MCMC one well at a time, the way a per-well fit is usually run
    one = MCMCPosterior(model.prior)
    for i in range(16):
        one.posterior(x[i:i + 1], n_draws, seed=i)
    timing = {"mcmc_seconds_per_well": mc.seconds / mc.wells,
              "mcmc_seconds_per_well_one_at_a_time": one.seconds / one.wells,
              "network_seconds_per_well": nn_seconds,
              "mcmc_fits": mc.wells, "draws_per_well": n_draws,
              "mcmc_settings": mc.settings.__dict__,
              "mcmc_median_acceptance": float(np.median(mc.acceptance)),
              "network_training_seconds": model.manifest["training_history"]["seconds"],
              "machine": f"{platform.processor() or platform.machine()}, "
                         f"{__import__('os').cpu_count()} CPU threads"}
    (a.out / "mcmc_timing.json").write_text(json.dumps(timing, indent=2) + "\n")

    # score on the wells where every compared method has a forecast
    saved = pd.read_csv(a.test / "forecasts.csv", dtype={"api": str})
    saved = saved[saved.method.isin(COMPARED)]
    allf = pd.concat([saved[["api", "history", "horizon", "method", "actual_bbl"] + QCOLS],
                      mcmc], ignore_index=True)
    actual = saved.groupby(["api", "history", "horizon"]).actual_bbl.first()
    op = wells.set_index("api").first_operator
    scores, boots = [], []
    for (h, t), g in allf.groupby(["history", "horizon"]):
        piv = {m: g[g.method == m].set_index("api")[QCOLS] for m in COMPARED}
        apis = sorted(set.intersection(*(set(p.index) for p in piv.values())))
        y = actual.loc[[(api, h, t) for api in apis]].to_numpy()
        fc = {m: piv[m].loc[apis].to_numpy() for m in COMPARED}
        s = score_table(fc, y, history=h, horizon=t)
        scores.append(s)
        ok = np.isfinite(y) & np.all([np.all(np.isfinite(q) & (q > 0), 1) for q in fc.values()], 0)
        med = list(np.round(LEVELS_Q, 4)).index(0.5)
        per = {"wis": {m: wis(q[ok], y[ok]) for m, q in fc.items()},
               "abs_log_error": {m: np.abs(np.log10(q[ok, med] / y[ok])) for m, q in fc.items()}}
        for lv in INTERVALS:
            per[f"coverage_{int(lv * 100)}"] = {m: covered(q[ok], y[ok], lv).astype(float)
                                               for m, q in fc.items()}
        cl = op.loc[np.asarray(apis)[ok]].to_numpy()
        for metric in ("wis", "abs_log_error", "coverage_80", "coverage_90"):
            b = cluster_bootstrap(per[metric], cl, reference="mcmc", n_boot=a.n_boot, seed=seed)
            boots.append(b.assign(history=h, horizon=t, metric=metric))
    scores = pd.concat(scores, ignore_index=True)
    boots = pd.concat(boots, ignore_index=True)
    scores.to_csv(a.out / "mcmc_scores.csv", index=False)
    boots.to_csv(a.out / "mcmc_bootstrap.csv", index=False)
    cols = ["history", "horizon", "method", "n", "wis", "median_abs_pct_error",
            "median_log_bias", "coverage_80", "coverage_90"]
    print(scores[cols].round(4).to_string(index=False))
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
