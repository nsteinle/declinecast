"""Tableau Public extract: a validation folder and score folders as flat CSV tables.

Tableau Public reads text files and not Parquet, so the extract is CSV. Tables are built by
DuckDB queries in ``declinecast/sql`` (``tableau_*.sql``) or copied unchanged. ``docs/TABLEAU.md``
describes the dashboard; its first panel is the calibration on real wells.

    reliability.csv   real wells: share of actual cumulatives inside each central interval, by
                      method, history and horizon, against the nominal level
    forecasts.csv     one row per well, history, horizon and method: P90/P50/P10, actual,
                      error, interval hits, WIS, out-of-distribution flag, operator, vintage,
                      lateral, latitude and longitude
    scores.csv        WIS, median error, bias and coverage per method, history and horizon
    intervals.csv     cluster-bootstrap 95% intervals for WIS and 80% coverage, and for each
                      method's difference from least-squares Arps
    examples.csv      monthly cumulative forecast fans for a few example wells, with actuals
    coverage.csv      simulated wells: coverage of the posterior's credible intervals
    pp.csv            simulated wells: SBC PP-plot points with the KS band
    accuracy.csv      simulated wells: correlation, bias, error and contraction per quantity
    tables.json       what each file holds, its columns and row count, and its source
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from declinecast import sql
from declinecast._version import __version__
from declinecast.datasets import load_panel

__all__ = ["export", "DESCRIPTIONS"]

DESCRIPTIONS = {
    "reliability.csv": "real wells: share of actual cumulatives inside each central interval, "
                       "by method, history and horizon, against the nominal level",
    "forecasts.csv": "one row per well, history, horizon and method: P90, P50, P10, actual, "
                     "error, interval hits, WIS, out-of-distribution flag and well descriptors",
    "scores.csv": "WIS, median error, bias and coverage per method, history and horizon",
    "intervals.csv": "cluster-bootstrap 95% intervals for WIS and 80% coverage and for each "
                     "method's difference from least-squares Arps",
    "examples.csv": "monthly cumulative forecast fans for example wells, with actuals",
    "coverage.csv": "simulated wells: coverage of the posterior's central credible intervals",
    "pp.csv": "simulated wells: simulation-based-calibration PP plot with the KS band",
    "accuracy.csv": "simulated wells: Spearman, bias, error and contraction per quantity",
}


def export(out: str | Path, score_dirs=(), validation_dir: str | Path | None = None,
           panel=None) -> dict:
    """Write the Tableau tables to ``out``. ``score_dirs`` are folders written by
    ``declinecast score`` (development, test, as-reported); ``validation_dir`` one written by
    ``declinecast validate``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    panel = panel or load_panel()
    written = {}

    def put(name, frame, source):
        frame.to_csv(out / name, index=False)
        written[name] = {"description": DESCRIPTIONS[name], "rows": len(frame),
                         "columns": list(frame.columns), "source": source}

    score_dirs = [Path(d) for d in score_dirs]
    if score_dirs:
        forecasts = pd.concat([pd.read_csv(d / "forecasts.csv", dtype={"api": str})
                               for d in score_dirs], ignore_index=True)
        wells = panel.wells.drop(columns=["spud_date"])
        with sql.connect(forecasts=forecasts, wells=wells) as con:
            put("forecasts.csv", con.execute(sql.load("tableau_forecasts")).df(),
                "sql/tableau_forecasts.sql")
            put("reliability.csv", con.execute(sql.load("tableau_reliability")).df(),
                "sql/tableau_reliability.sql")
        for name, src in (("scores.csv", "scores.csv"), ("intervals.csv", "bootstrap.csv"),
                          ("examples.csv", "examples.csv")):
            parts = [pd.read_csv(d / src, dtype={"api": str}) for d in score_dirs
                     if (d / src).exists()]
            if parts:
                put(name, pd.concat(parts, ignore_index=True), f"declinecast score ({src})")
    if validation_dir is not None:
        v = Path(validation_dir)
        meta = json.loads((v / "validation.json").read_text())
        coverage = pd.read_csv(v / "coverage.csv")
        ranks = pd.read_csv(v / "sbc_ranks.csv")
        with sql.connect(coverage=coverage, ranks=ranks) as con:
            put("coverage.csv", con.execute(sql.load("tableau_coverage")).df(),
                "sql/tableau_coverage.sql")
            band = f"{float(meta['sbc_band']):.6f}"  # a number, formatted here
            put("pp.csv", con.execute(sql.load("tableau_pp", band=band)).df(),
                "sql/tableau_pp.sql")
        put("accuracy.csv", pd.read_csv(v / "accuracy.csv"), "declinecast validate")
    (out / "tables.json").write_text(json.dumps(
        {"declinecast": __version__, "tables": written}, indent=2) + "\n")
    return written
