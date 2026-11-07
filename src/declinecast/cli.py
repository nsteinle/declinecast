"""declinecast command line.

  declinecast datasets                                  # the bundled extract and its source
  declinecast quality                                   # the extract's data-quality table
  declinecast quality --raw data/raw                    # the full report on the raw files
  declinecast fetch --dest data/raw                     # opt-in: the published ECMC files
  declinecast panel --raw data/raw --out data/panel     # rebuild the extract in DuckDB SQL
  declinecast simulate --out results/ppc                # prior predictive check
  declinecast train --out results/model                 # ~1.5-3 h on a laptop CPU
  declinecast validate --out results/validation         # SBC and coverage, simulated wells
  declinecast score --split development --out results/dev
  declinecast score --split test --unlock-test --out results/test     # once
  declinecast predict --csv my_wells.csv --out forecasts.csv
  declinecast predict --api 050010984800 --history 12 --out one_well.csv
  declinecast tableau --score results/dev results/test --validation results/validation \\
      --out results/tableau
  declinecast sql --query 'SELECT vintage, count(*) FROM wells GROUP BY ALL ORDER BY ALL'

Every command that uses a model reads the pretrained one bundled with the package unless
``--model-dir`` names another.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd


def _model(a):
    from declinecast.modelio import load_model

    return load_model(a.model_dir)


def _datasets(a) -> None:
    from declinecast.datasets import load_panel

    p = load_panel()
    w = p.wells
    print(f"DJ Basin horizontal Niobrara and Codell cohort, {len(w):,} wells, "
          f"{len(p.months):,} well-months")
    print(f"source: {p.meta['source']}\n        {p.meta['download_page']}")
    print(f"built: {p.meta['built_utc']} with {p.meta['built_with']}")
    print("split by first-production year: "
          + ", ".join(f"{s} {n:,}" for s, n in w.split.value_counts().sort_index().items()))
    print("license: see DATA_LICENSE.md (public record; the extract is derived)")


def _quality(a) -> None:
    if a.raw:
        from declinecast import sql
        from declinecast.ecmc import build_tables
        from declinecast.quality import raw_quality_report

        with sql.connect_files() as con:
            build_tables(con, a.raw)
            print(json.dumps(raw_quality_report(con), indent=2, default=str))
        return
    from declinecast.datasets import load_panel
    from declinecast.quality import extract_quality_report

    p = load_panel()
    print("raw files (from the bundle's panel.json):")
    q = p.meta["quality"]
    print(f"  {q['raw_rows']:,} rows; " + ", ".join(f"{k} {v:,}" for k, v in
                                                   q["versions"].items() if v is not None))
    print("  cohort funnel: " + ", ".join(f"{k} {v:,}" for k, v in q["cohort_funnel"].items()
                                          if v is not None))
    print("\nbundled extract (DuckDB SQL, sql/quality_extract.sql):")
    q = extract_quality_report(p).set_index("split")
    print(q.round(0).astype("int64").T.to_string())


def _fetch(a) -> None:
    from declinecast.ecmc import PRODUCTION_YEARS, fetch

    years = a.years or PRODUCTION_YEARS
    rec = fetch(a.dest, years=years, pause=a.pause, overwrite=a.overwrite)
    print(f"{len(rec['files'])} files in {a.dest}; provenance.json written")


def _panel(a) -> None:
    from declinecast.ecmc import build_panel

    meta = build_panel(a.raw, a.out, threads=a.threads)
    print(f"wrote {a.out}/wells.parquet, months.parquet, panel.json "
          f"({meta['quality']['cohort_funnel'].get('cohort', 'n/a')} wells)")


def _simulate(a) -> None:
    from declinecast.datasets import load_panel, well_arrays
    from declinecast.pipeline import check_out
    from declinecast.training import development_prior
    from declinecast.validation import predictive_check

    p = load_panel()
    out = check_out(a.out, a.overwrite)
    prior = development_prior(p)
    dev = well_arrays(p, wells=p.wells[p.wells.split == "development"])
    table = predictive_check(prior, dev.rate[dev.available >= 36], n_sim=a.n, seed=a.seed)
    table.to_csv(out / "predictive_check.csv", index=False)
    (out / "prior.json").write_text(json.dumps(prior.to_dict(), indent=2) + "\n")
    print(table.round(3).to_string(index=False))


def _train(a) -> None:
    from declinecast.datasets import load_panel
    from declinecast.inference import TrainConfig
    from declinecast.training import train_model

    cfg = TrainConfig(n_simulations=a.n_simulations, max_epochs=a.max_epochs, seed=a.seed,
                      noise=not a.noiseless, use_lateral=not a.no_lateral)
    path = train_model(load_panel(), a.out, cfg, overwrite=a.overwrite)
    print(f"model written to {path}")


def _validate(a) -> None:
    from declinecast.pipeline import run_validation

    s = run_validation(_model(a), a.out, n_sbc=a.n_sbc, n_per_history=a.n_per_history,
                       n_draws=a.draws, seed=a.seed, overwrite=a.overwrite)
    print(f"SBC: {sum(s['sbc_pass'].values())}/{len(s['sbc_pass'])} quantities pass; "
          f"coverage consistent with nominal in {s['coverage_consistent']} cells; "
          f"median acceptance {s['acceptance']['median']:.3f}")


def _score(a) -> None:
    from declinecast.datasets import load_panel
    from declinecast.pipeline import run_scoring

    if a.split == "test" and not a.unlock_test:
        raise SystemExit("the 2020+ wells are the locked test set; pass --unlock-test to score "
                         "them (the pre-registration says once)")
    r = run_scoring(_model(a), load_panel(), a.split, a.out, n_draws=a.draws, seed=a.seed,
                    as_reported=a.as_reported, block=a.block, max_wells=a.max_wells,
                    n_boot=a.bootstrap, overwrite=a.overwrite)
    s = r["scores"]
    s = s[s.wells == "all"]
    cols = ["history", "horizon", "method", "n", "wis", "median_abs_pct_error",
            "median_log_bias", "coverage_80", "coverage_90"]
    print(s[cols].round(3).to_string(index=False))


def _predict(a) -> None:
    from declinecast.datasets import load_panel, well_arrays
    from declinecast.pipeline import predict_table, wells_from_table

    p = load_panel()
    if a.csv:
        arrays = wells_from_table(pd.read_csv(a.csv))
    else:
        w = p.wells[p.wells.api.isin(a.api)]
        if len(w) != len(set(a.api)):
            raise SystemExit("some --api values are not in the bundled extract")
        arrays = well_arrays(p, wells=w)
    table = predict_table(_model(a), arrays, p, history=a.history, horizons=tuple(a.horizons),
                          n_draws=a.draws, seed=a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out, index=False)
    show = table.dropna(subset=["horizon_months"]) if "horizon_months" in table else table
    cols = [c for c in ("well_id", "history_months", "horizon_months", "q0.1", "q0.5", "q0.9")
            if c in show]
    print(show[cols].round(0).to_string(index=False))
    print(f"wrote {a.out} (q0.1 is P90, the low case; q0.9 is P10)")


def _tableau(a) -> None:
    from declinecast.tableau import export

    written = export(a.out, score_dirs=a.score or (), validation_dir=a.validation)
    for name, t in written.items():
        print(f"{name:18s} {t['rows']:>8,} rows")


def _sql(a) -> None:
    from declinecast import sql
    from declinecast.datasets import load_panel

    p = load_panel()
    wells = p.wells.drop(columns=["spud_date"])
    print(sql.query(a.query, wells=wells, months=p.months).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="declinecast", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    from declinecast._version import __version__

    ap.add_argument("--version", action="version", version=f"declinecast {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def model_arg(p):
        p.add_argument("--model-dir", type=Path, default=None,
                       help="a model directory (default: the bundled pretrained model)")

    def out_args(p):
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--overwrite", action="store_true")

    sub.add_parser("datasets", help="the bundled extract, its source and license")
    p = sub.add_parser("quality", help="data-quality report")
    p.add_argument("--raw", type=Path, help="a folder of raw ECMC files (full report)")
    p = sub.add_parser("fetch", help="download the published ECMC files (opt-in, no crawling)")
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--years", type=int, nargs="*")
    p.add_argument("--pause", type=float, default=2.0, help="seconds between requests")
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("panel", help="build the cohort extract from raw files (DuckDB SQL)")
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int)
    p = sub.add_parser("simulate", help="prior predictive check against development wells")
    out_args(p)
    p.add_argument("--n", type=int, default=30000)
    p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("train", help="simulate a training set and train the network")
    out_args(p)
    p.add_argument("--n-simulations", type=int, default=500_000)
    p.add_argument("--max-epochs", type=int, default=45)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noiseless", action="store_true", help="ablation: no noise or downtime")
    p.add_argument("--no-lateral", action="store_true", help="ablation: hide the lateral")
    p = sub.add_parser("validate", help="calibration on held-out simulated wells")
    model_arg(p)
    out_args(p)
    p.add_argument("--n-sbc", type=int, default=1000)
    p.add_argument("--n-per-history", type=int, default=500)
    p.add_argument("--draws", type=int, default=500)
    p.add_argument("--seed", type=int, default=2026)
    p = sub.add_parser("score", help="forecast and score real wells against baselines")
    model_arg(p)
    out_args(p)
    p.add_argument("--split", choices=["development", "test"], required=True)
    p.add_argument("--unlock-test", action="store_true")
    p.add_argument("--as-reported", action="store_true",
                   help="inputs as first filed (outcomes stay final)")
    p.add_argument("--block", choices=["township", "operator"], default="township",
                   help="cross-validation blocks for the empirical baselines (development)")
    p.add_argument("--max-wells", type=int)
    p.add_argument("--draws", type=int, default=500)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("predict", help="forecast cumulative oil for your wells")
    model_arg(p)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path,
                     help="columns well_id, month (YYYY-MM), oil_bbl, days, lateral_ft")
    src.add_argument("--api", nargs="+", help="12-digit API numbers of bundled wells")
    p.add_argument("--history", type=int, help="months to use (default: all, up to 60)")
    p.add_argument("--horizons", type=int, nargs="+", default=[36, 60])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--draws", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("tableau", help="Tableau Public tables (CSV) from results folders")
    p.add_argument("--score", type=Path, nargs="*")
    p.add_argument("--validation", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("sql", help="a read-only DuckDB query over the tables wells and months")
    p.add_argument("--query", required=True)

    a = ap.parse_args(argv)
    handler = {"datasets": _datasets, "quality": _quality, "fetch": _fetch, "panel": _panel,
               "simulate": _simulate, "train": _train, "validate": _validate,
               "score": _score, "predict": _predict, "tableau": _tableau, "sql": _sql}[a.cmd]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            handler(a)
        except FileExistsError as e:
            raise SystemExit(str(e)) from None


if __name__ == "__main__":
    main()
