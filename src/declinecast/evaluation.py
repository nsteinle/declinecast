"""Score the network and the baselines on real wells.

For each history H (months seen) and horizon T (months forecast), every well with at least T
months of record gets a forecast of its cumulative oil to T from each method, and the forecasts
are scored against what the well actually produced (``declinecast.scoring``).

* **Development wells** (first produced 2014-2019). The network's prior and the uptime table are
  set from these wells, so its development scores are mildly in-sample; the empirical
  baselines are fit by five-fold cross-validation blocked by township (or by operator), so
  they are not scored on wells they were fit on.
* **Locked test wells** (first produced 2020 and later). Everything is fit on development wells
  only; the test wells are scored once (FINDINGS section 8).

``as_reported=True`` gives every method the production as first filed rather than as finally
revised (the record an analyst would have had at the time), while the outcome stays the final
record.
"""

from __future__ import annotations

import time
import zlib

import numpy as np
import pandas as pd

from declinecast import baselines, ood
from declinecast.datasets import well_arrays
from declinecast.forecasting import LEVELS_Q, forecast, uptime_table
from declinecast.scoring import INTERVALS, cluster_bootstrap, covered, score_table, wis

__all__ = ["HISTORIES_SCORED", "HORIZONS", "METHODS", "evaluate"]

HISTORIES_SCORED = (6, 12, 18, 24)
HORIZONS = (36, 60)
METHODS = ("network", "multiplier", "type_curve", "arps_ls")
N_FOLDS = 5


def _folds(keys, n_folds=N_FOLDS):
    return np.array([zlib.crc32(str(k).encode()) % n_folds for k in keys])


def _empirical(ref, target, fold_of, fold_target, h, t):
    """Multiplier and type-curve quantiles; with folds, each target fold is forecast from the
    reference wells outside it."""
    if fold_of is None:
        return baselines.multiplier(ref, target, h, t), baselines.type_curve(ref, target, t)
    mult = np.full((len(target.api), len(LEVELS_Q)), np.nan)
    tc = mult.copy()
    for f in range(N_FOLDS):
        r, g = ref.take(fold_of != f), fold_target == f
        if g.any():
            sub = target.take(g)
            mult[g] = baselines.multiplier(r, sub, h, t)
            tc[g] = baselines.type_curve(r, sub, t)
    return mult, tc


def evaluate(model, panel, split: str = "development", histories=HISTORIES_SCORED,
             horizons=HORIZONS, n_draws: int = 500, seed: int = 0, as_reported: bool = False,
             block: str = "township", max_wells: int | None = None, n_boot: int = 2000,
             log=print) -> dict:
    """Returns ``scores`` (one row per history, horizon, method and well group), ``bootstrap``
    (cluster-bootstrap intervals, by operator and by pad, for the mean WIS, the mean absolute
    log10 error of the median, and the 80% and 90% coverage, with each method's difference
    from least-squares Arps),
    ``forecasts`` (one row per well, history, horizon and method) and ``ood`` counts."""
    wells = panel.wells
    dev = well_arrays(panel, wells=wells[wells.split == "development"])
    target_wells = wells[wells.split == split]
    cols = ({"oil_column": "oil_bbl_first", "days_column": "days_first"} if as_reported else {})
    target = well_arrays(panel, wells=target_wells, **cols)
    truth = well_arrays(panel, wells=target_wells)
    keep = np.flatnonzero(target.available >= min(horizons))
    if max_wells is not None and len(keep) > max_wells:
        keep = np.sort(np.random.default_rng(seed).choice(keep, max_wells, replace=False))
    target, truth = target.take(keep), truth.take(keep)
    uptime = uptime_table(dev, [(h, t) for h in histories for t in horizons])
    band = ood.rate_band(model.prior)
    if split == "development":
        key = {"township": dev.township, "operator": dev.first_operator}[block]
        fold_ref = _folds(key)
        fold_target = _folds({"township": target.township,
                              "operator": target.first_operator}[block])
    else:
        fold_ref = fold_target = None
    scores, boots, records, ood_counts = [], [], [], []
    for h in histories:
        t0 = time.time()
        nn = forecast(model, target, h, horizons, uptime, n_draws=n_draws, seed=seed + h)
        flag = ood.flags(target, h, band, nn["acceptance"])
        ood_counts.append({"history": h, **{k: int(v.sum()) for k, v in flag.items()},
                           "wells": len(target.api)})
        for t in horizons:
            ok = target.available >= t
            mult, tc = _empirical(dev, target, fold_ref, fold_target, h, t)
            fc = {"network": nn["quantiles"][t], "multiplier": mult, "type_curve": tc,
                  "arps_ls": baselines.arps_ls(target, h, t, uptime, n_draws=n_draws,
                                               seed=seed + h)}
            if "quantiles_unadjusted" in nn:  # the network before recalibration
                fc["network_unadjusted"] = nn["quantiles_unadjusted"][t]
            y = truth.cumulative(t)
            fc = {m: q[ok] for m, q in fc.items()}
            yy, fl = y[ok], flag["any"][ok]
            labels = {"split": split, "as_reported": as_reported, "history": h, "horizon": t}
            scores.append(score_table(fc, yy, **labels, wells="all"))
            scores.append(score_table({m: q[~fl] for m, q in fc.items()}, yy[~fl], **labels,
                                      wells="not_flagged"))
            if fl.sum() >= 10:
                scores.append(score_table({m: q[fl] for m, q in fc.items()}, yy[fl], **labels,
                                          wells="flagged"))
            common = np.isfinite(yy) & np.all(
                [np.all(np.isfinite(q) & (q > 0), 1) for q in fc.values()], axis=0)
            med = list(np.round(LEVELS_Q, 4)).index(0.5)
            per_metric = {
                "wis": {m: wis(q[common], yy[common]) for m, q in fc.items()},
                "abs_log_error": {m: np.abs(np.log10(q[common, med] / yy[common]))
                                  for m, q in fc.items()},
                "coverage_80": {m: covered(q[common], yy[common], 0.80).astype(float)
                                for m, q in fc.items()},
                "coverage_90": {m: covered(q[common], yy[common], 0.90).astype(float)
                                for m, q in fc.items()},
            }
            for cname, cl in (("operator", target.first_operator[ok][common]),
                              ("pad", target.loc_id[ok][common])):
                for metric, per in per_metric.items():
                    b = cluster_bootstrap(per, cl, reference="arps_ls", n_boot=n_boot,
                                          seed=seed)
                    boots.append(b.assign(cluster=cname, metric=metric, **labels))
            api = target.api[ok]
            for m, q in fc.items():
                rec = pd.DataFrame(q, columns=[f"q{lv:g}" for lv in LEVELS_Q])
                rec.insert(0, "method", m)
                rec.insert(0, "api", api)
                rec["actual_bbl"] = yy
                rec["observed_to_date_bbl"] = target.oil[ok, :h].sum(1)
                rec["ood_flag"] = fl
                for lv in INTERVALS:
                    rec[f"covered_{int(lv * 100)}"] = covered(q, yy, lv)
                rec["wis"] = np.where(np.all(np.isfinite(q), 1), wis(np.where(q > 0, q, np.nan),
                                                                     yy), np.nan)
                records.append(rec.assign(**labels))
        if log:
            log(f"{split}: history {h} on {len(target.api):,} wells ({time.time() - t0:.0f} s)")
    return {"scores": pd.concat(scores, ignore_index=True),
            "bootstrap": pd.concat(boots, ignore_index=True),
            "forecasts": pd.concat(records, ignore_index=True),
            "ood": pd.DataFrame(ood_counts),
            "uptime": uptime.to_dict(), "rate_band": band.to_dict()}
