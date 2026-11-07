"""The figures in README.md and docs/, from the results folders written by scripts/reproduce.sh.

    python scripts/figures.py --results results --out docs/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from declinecast import sql
from declinecast.plotting import fan_plot, pp_plot, reliability_plot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    v = a.results / "validation"
    band = json.loads((v / "validation.json").read_text())["sbc_band"]
    pp_plot(pd.read_csv(v / "sbc_ranks.csv"), band, a.out / "sbc_pp.png",
            title="Simulation-based calibration, held-out simulated wells")
    for split in ("test", "dev"):
        f = a.results / split / "forecasts.csv"
        if not f.exists():
            continue
        forecasts = pd.read_csv(f, dtype={"api": str})
        rel = sql.query(sql.load("tableau_reliability"), forecasts=forecasts)
        rel["method"] = rel["method"].map({"network": "declinecast network",
                                           "multiplier": "history multiplier",
                                           "type_curve": "type curve (lateral only)",
                                           "arps_ls": "least-squares Arps"})
        name = "locked test wells (first produced 2020-2022)" if split == "test" else \
            "development wells (2014-2019)"
        reliability_plot(rel, a.out / f"reliability_{split}.png", horizon=36,
                         title=f"Interval coverage of 36-month cumulative oil, {name}")
        ex = a.results / split / "examples.csv"
        if ex.exists() and split == "test":
            fan_plot(pd.read_csv(ex, dtype={"api": str}), a.out / "fans_test.png")
    print("figures in", a.out)


if __name__ == "__main__":
    main()
