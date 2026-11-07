"""Fit the forecast recalibration on development wells and write it into a model directory.

    python scripts/recalibrate.py --dev results/dev --model-dir src/declinecast/data/model

For each history H and horizon T scored on development wells, it finds the factor f that,
applied to the network's predicted oil (months H to T; the oil already produced is not touched),
makes the median log10 error of the forecast median zero. This is a location-only correction:
the spread of each forecast is scaled by the same f. It writes the factors into the model's
manifest (``forecast_adjustment``), regenerates the model card, and prints the development
scores before and after (after is in-sample, by construction; the locked test is the check).
Only development wells are read.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from declinecast.forecasting import LEVELS_Q
from declinecast.modelio import update_manifest, write_model_card
from declinecast.scoring import INTERVALS, covered, score_table, wis

QCOLS = [f"q{lv:g}" for lv in LEVELS_Q]


def fit_factor(seen, median, actual):
    future = median - seen
    def gap(f):
        return np.median(np.log10((seen + f * future) / actual))
    return float(brentq(gap, 0.3, 3.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=Path, required=True, help="a development score folder")
    ap.add_argument("--model-dir", type=Path, required=True)
    a = ap.parse_args()
    meta = json.loads((a.dev / "score.json").read_text())
    if meta["split"] != "development" or meta["as_reported"]:
        raise SystemExit("fit the recalibration on a development score folder (final record)")
    f = pd.read_csv(a.dev / "forecasts.csv", dtype={"api": str})
    # a model that already carries a recalibration also writes its forecasts before it
    # (network_unadjusted); the fit always uses those, so rerunning gives the same factors
    base = "network_unadjusted" if "network_unadjusted" in set(f.method) else "network"
    net = f[f.method == base]
    factors, rows = {}, []
    for (h, t), g in net.groupby(["history", "horizon"]):
        ok = np.isfinite(g["q0.5"]) & (g.actual_bbl > 0)
        g = g[ok]
        fac = fit_factor(g.observed_to_date_bbl.to_numpy(), g["q0.5"].to_numpy(),
                         g.actual_bbl.to_numpy())
        factors[f"{h}-{t}"] = round(fac, 4)
        q = g[QCOLS].to_numpy()
        seen = g.observed_to_date_bbl.to_numpy()[:, None]
        adj_q = seen + fac * (q - seen)
        rows.append(score_table({"network_unadjusted": q, "network_recalibrated": adj_q},
                                g.actual_bbl.to_numpy(), history=h, horizon=t)
                    .assign(factor=fac))
    table = pd.concat(rows, ignore_index=True)
    # the folder's forecasts: the original network rows become network_unadjusted, and
    # recalibrated network rows are added, so Tableau shows the same methods as the test
    adj = net.copy()
    seen = adj.observed_to_date_bbl.to_numpy()[:, None]
    fac = np.array([factors[f"{h}-{t}"] for h, t in zip(adj.history, adj.horizon,
                                                         strict=True)])[:, None]
    q = seen + fac * (adj[QCOLS].to_numpy() - seen)
    adj[QCOLS] = q
    y = adj.actual_bbl.to_numpy()
    for lv in INTERVALS:
        adj[f"covered_{int(lv * 100)}"] = covered(q, y, lv)
    adj["wis"] = wis(q, y)
    adj["method"] = "network"
    f = f[f.method != "network"].copy() if base == "network_unadjusted" else f
    f.loc[f.method == "network", "method"] = "network_unadjusted"
    pd.concat([f, adj], ignore_index=True).to_csv(a.dev / "forecasts.csv", index=False)
    update_manifest(a.model_dir, forecast_adjustment={
        "factors": factors,
        "method": "predicted oil scaled so the median log10 error on development wells is 0",
        "fit_on": {"split": "development", "wells": int(net.api.nunique()),
                   "score_folder_settings": {k: meta[k] for k in ("n_draws", "seed",
                                                                 "max_wells")}},
        "fitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    write_model_card(a.model_dir)
    table.to_csv(a.dev / "recalibration.csv", index=False)
    cols = ["history", "horizon", "method", "n", "factor", "wis", "median_log_bias",
            "median_abs_pct_error", "coverage_80", "coverage_90"]
    print(table[cols].round(4).to_string(index=False))
    print(json.dumps(factors))


if __name__ == "__main__":
    main()
