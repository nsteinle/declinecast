"""The steps behind ``declinecast validate``, ``score`` and ``predict``, as Python functions that
write a results folder.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from declinecast._version import __version__
from declinecast.datasets import WellArrays, well_arrays
from declinecast.evaluation import HISTORIES_SCORED, HORIZONS, evaluate
from declinecast.forecasting import LEVELS_Q, forecast, uptime_table
from declinecast.simulator import MIN_DAYS
from declinecast.validation import QUANTITIES, validate

__all__ = ["run_validation", "run_scoring", "wells_from_table", "predict_table",
           "example_fans", "check_out"]


def check_out(out: Path, overwrite: bool) -> Path:
    out = Path(out)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} is not empty; use --overwrite to replace its files")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _model_ref(model) -> dict:
    return {"path": str(model.path), "weights_sha256": model.manifest["files"]["weights.pt"],
            "created_utc": model.manifest["created_utc"]}


def run_validation(model, out, n_sbc=1000, n_per_history=500, n_draws=500, seed=2026,
                   overwrite=False, log=print) -> dict:
    out = check_out(out, overwrite)
    res = validate(model, n_sbc=n_sbc, n_per_history=n_per_history, n_draws=n_draws,
                   seed=seed, log=log)
    res["sbc"].to_csv(out / "sbc.csv", index=False)
    res["sbc_ranks"].to_csv(out / "sbc_ranks.csv", index=False)
    res["coverage"].to_csv(out / "coverage.csv", index=False)
    res["accuracy"].to_csv(out / "accuracy.csv", index=False)
    cov = res["coverage"]
    summary = {
        "declinecast": __version__, "model": _model_ref(model),
        "settings": {"n_sbc": n_sbc, "n_per_history": n_per_history, "n_draws": n_draws,
                     "seed": seed},
        "sbc_band": res["sbc"].attrs["band"],
        "sbc_pass": {r.quantity: bool(r.passes) for r in res["sbc"].itertuples()},
        "coverage_consistent": f"{int(cov.consistent.sum())}/{len(cov)}",
        "worst_coverage_gap": float((cov.coverage - cov.level).abs().max()),
        "acceptance": res["acceptance"], "known_answer": res["known_answer"],
    }
    (out / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def example_fans(model, panel, split="test", history=12, horizon=60, n_wells=8, n_draws=500,
                 seed=11) -> pd.DataFrame:
    """Monthly cumulative forecast quantiles for a few wells with ``horizon`` months of record,
    and the actual cumulative, for the dashboard's fan chart."""
    wells = panel.wells
    dev = well_arrays(panel, wells=wells[wells.split == "development"])
    a = well_arrays(panel, wells=wells[wells.split == split])
    pool = np.flatnonzero(a.available >= horizon)
    if not len(pool):
        return pd.DataFrame()
    rows = np.sort(np.random.default_rng(seed).choice(pool, min(n_wells, len(pool)),
                                                      replace=False))
    months = list(range(history + 1, horizon + 1))
    up = uptime_table(dev, [(history, t) for t in months])
    f = forecast(model, a, history, months, up, n_draws=n_draws, rows=rows, seed=seed)
    recs = []
    for i, r in enumerate(rows):
        actual = np.cumsum(a.oil[r, :horizon])
        for k in range(1, history + 1):
            recs.append({"api": a.api[r], "month": k, "actual_bbl": actual[k - 1],
                         "phase": "history"})
        for t in months:
            q = f["quantiles"][t][i]
            recs.append({"api": a.api[r], "month": t, "actual_bbl": actual[t - 1],
                         "phase": "forecast",
                         **{f"q{lv:g}": v for lv, v in zip(LEVELS_Q, q, strict=True)}})
    return pd.DataFrame(recs).assign(split=split, history=history)


def run_scoring(model, panel, split, out, histories=HISTORIES_SCORED, horizons=HORIZONS,
                n_draws=500, seed=0, as_reported=False, block="township", max_wells=None,
                n_boot=2000, examples=True, overwrite=False, log=print) -> dict:
    out = check_out(out, overwrite)
    res = evaluate(model, panel, split=split, histories=histories, horizons=horizons,
                   n_draws=n_draws, seed=seed, as_reported=as_reported, block=block,
                   max_wells=max_wells, n_boot=n_boot, log=log)
    res["forecasts"].to_csv(out / "forecasts.csv", index=False)
    res["scores"].to_csv(out / "scores.csv", index=False)
    res["bootstrap"].to_csv(out / "bootstrap.csv", index=False)
    res["ood"].to_csv(out / "ood.csv", index=False)
    if examples and not as_reported:
        example_fans(model, panel, split=split, n_draws=n_draws).to_csv(out / "examples.csv",
                                                                        index=False)
    summary = {"declinecast": __version__, "model": _model_ref(model), "split": split,
               "as_reported": as_reported, "block": block if split == "development" else None,
               "histories": list(histories), "horizons": list(horizons), "n_draws": n_draws,
               "seed": seed, "max_wells": max_wells, "panel_built_utc": panel.meta["built_utc"],
               "uptime_quantiles": res["uptime"], "rate_band": res["rate_band"]}
    (out / "score.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {"summary": summary, "scores": res["scores"], "bootstrap": res["bootstrap"]}


# ---------------------------------------------------------------------- your own wells
def wells_from_table(df: pd.DataFrame, n_months: int = 120) -> WellArrays:
    """Well arrays from a table with one row per well and producing month.

    Columns: ``well_id``, ``month`` (``YYYY-MM``), ``oil_bbl``, ``days`` (producing days) and
    ``lateral_ft`` (constant per well). Month 0 is each well's first month with oil; months with
    no row count as shut in. Record availability runs to the last month in the table."""
    need = {"well_id", "month", "oil_bbl", "days", "lateral_ft"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    d = df.copy()
    ym = pd.PeriodIndex(d["month"].astype(str), freq="M")
    d["ym"] = ym.year * 12 + ym.month - 1
    d["well_id"] = d["well_id"].astype(str)
    first = d[d.oil_bbl > 0].groupby("well_id").ym.min()
    ids = first.index.to_numpy()
    end = int(d.ym.max())
    d = d[d.well_id.isin(first.index)]
    k = (d.ym - d.well_id.map(first)).to_numpy()
    row = pd.Series(np.arange(len(ids)), index=ids)[d.well_id].to_numpy()
    ok = (k >= 0) & (k < n_months)
    oil, days = np.zeros((len(ids), n_months)), np.zeros((len(ids), n_months))
    np.add.at(oil, (row[ok], k[ok]), d.oil_bbl.to_numpy(float)[ok])
    np.add.at(days, (row[ok], k[ok]), d.days.to_numpy(float)[ok])
    rate = np.where((oil > 0) & (days >= MIN_DAYS), oil / np.maximum(days, 1), np.nan)
    lat = d.groupby("well_id").lateral_ft.first().reindex(ids).to_numpy(float)
    blank = np.array([""] * len(ids), dtype=object)
    return WellArrays(api=ids, rate=rate, oil=oil, days=days,
                      available=np.minimum(end - first.to_numpy() + 1, n_months),
                      lateral_ft=lat, vintage=first.to_numpy() // 12, first_operator=blank,
                      loc_id=blank, township=blank)


def predict_table(model, arrays: WellArrays, panel, history: int | None = None,
                  horizons=(36, 60), n_draws=500, seed=0) -> pd.DataFrame:
    """Forecast every well from its own months of record (or the first ``history`` months).
    Wells with fewer than 3 months, or with more months than the shortest horizon, are
    skipped with a note. Uptime comes from the bundled development wells."""
    wells = panel.wells
    dev = well_arrays(panel, wells=wells[wells.split == "development"])
    hist = np.minimum(arrays.available, 60) if history is None else np.full(
        len(arrays.api), int(history))
    recs = []
    for h in np.unique(hist):
        rows = np.flatnonzero(hist == h)
        hz = [t for t in horizons if t > h]
        if h < 3 or not hz:
            for r in rows:
                recs.append({"well_id": arrays.api[r], "history_months": int(h),
                             "note": "needs at least 3 months and a horizon beyond them"})
            continue
        up = uptime_table(dev, [(int(h), t) for t in hz])
        f = forecast(model, arrays, int(h), hz, up, n_draws=n_draws, rows=rows, seed=seed)
        for t in hz:
            for i, r in enumerate(rows):
                q = f["quantiles"][t][i]
                recs.append({"well_id": arrays.api[r], "history_months": int(h),
                             "horizon_months": t, "observed_to_date_bbl": f["observed"][i],
                             **{f"q{lv:g}": v for lv, v in zip(LEVELS_Q, q, strict=True)},
                             "posterior_acceptance": f["acceptance"][i], "note": ""})
    return pd.DataFrame(recs)


def parameter_summary(model, arrays: WellArrays, history: int, n_draws=500, seed=0):
    """Posterior medians and 90% intervals of the parameters and derived quantities."""
    from declinecast.quantities import derived
    from declinecast.simulator import encode

    x = encode(arrays.rate, np.full(len(arrays.api), history), arrays.lateral_ft)
    draws, _ = model.posterior(x, n_draws, seed=seed)
    d = derived(draws)
    full = np.concatenate([draws] + [d[k][..., None] for k in d], axis=-1)
    lo, med, hi = np.quantile(full, [0.05, 0.5, 0.95], axis=1)
    rows = []
    for i, w in enumerate(arrays.api):
        for j, q in enumerate(QUANTITIES):
            rows.append({"well_id": w, "quantity": q, "p05": lo[i, j], "median": med[i, j],
                         "p95": hi[i, j]})
    return pd.DataFrame(rows)
