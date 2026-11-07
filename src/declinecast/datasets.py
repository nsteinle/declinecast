"""The bundled DJ Basin cohort extract, and the arrays built from it.

``load_panel()`` reads the two Parquet tables shipped in ``declinecast/data`` (built by
``declinecast panel`` from the raw ECMC files), checking their SHA-256 against ``panel.json``.
``well_arrays`` turns the monthly table into the per-well matrices the model and the scores use.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from declinecast import sql
from declinecast.simulator import MIN_DAYS

__all__ = ["Panel", "load_panel", "well_arrays", "WellArrays", "RECORD_END_YM", "DEV_VINTAGES",
           "TEST_FIRST_VINTAGE", "split_of", "ym_label"]

#: The last month whose filings are complete. Cohort wells filing a month drop from about
#: 8,200 to about 6,000 from July 2025 (late filings), so the record is used through June 2025.
RECORD_END_YM = 2025 * 12 + 5
#: Development wells first produced 2014-2019; the locked test set is 2020 and later
#: (decided 7 November 2026, before any scoring).
DEV_VINTAGES = (2014, 2019)
TEST_FIRST_VINTAGE = 2020


def ym_label(ym: int) -> str:
    return f"{ym // 12}-{ym % 12 + 1:02d}"


def split_of(vintage) -> np.ndarray:
    v = np.asarray(vintage)
    return np.where(v >= TEST_FIRST_VINTAGE, "test", "development")


@dataclass
class Panel:
    wells: pd.DataFrame
    months: pd.DataFrame
    meta: dict


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_panel(path: str | Path | None = None, verify: bool = True) -> Panel:
    """The cohort extract: ``wells`` (one row per well) and ``months`` (one row per reported
    month). ``path`` is a directory holding ``wells.parquet``, ``months.parquet`` and
    ``panel.json``; by default the bundled copy."""
    if path is None:
        base = resources.files("declinecast") / "data"
        with resources.as_file(base) as p:
            return load_panel(Path(p), verify)
    path = Path(path)
    meta = json.loads((path / "panel.json").read_text())
    frames = {}
    for name in ("wells", "months"):
        f = path / f"{name}.parquet"
        if verify and _sha256(f) != meta["files"][name]["sha256"]:
            raise ValueError(f"{f} does not match the checksum in panel.json")
        with sql.connect_files() as con:
            frames[name] = con.execute(
                f"SELECT * FROM read_parquet({sql.literal(f.as_posix())})").df()
    wells = frames["wells"]
    wells["split"] = split_of(wells["vintage"])
    return Panel(wells, frames["months"], meta)


@dataclass
class WellArrays:
    """Per-well monthly matrices, by calendar month since first production (column k)."""

    api: np.ndarray            # (n,)
    rate: np.ndarray           # (n, K) oil per producing day; NaN where unobserved
    oil: np.ndarray            # (n, K) barrels in the month (0 if no row)
    days: np.ndarray           # (n, K) producing days in the month (0 if no row)
    available: np.ndarray      # (n,) months of record available (through RECORD_END_YM)
    lateral_ft: np.ndarray
    vintage: np.ndarray
    first_operator: np.ndarray
    loc_id: np.ndarray
    township: np.ndarray

    def take(self, rows) -> WellArrays:
        """The wells at positions ``rows``."""
        return WellArrays(**{f: getattr(self, f)[rows] for f in self.__dataclass_fields__})

    def cumulative(self, months: int) -> np.ndarray:
        """Barrels produced in calendar months 0 .. months-1; NaN if not all available."""
        cum = self.oil[:, :months].sum(1)
        return np.where(self.available >= months, cum, np.nan)


def well_arrays(panel: Panel, n_months: int = 120, oil_column: str = "oil_bbl",
                days_column: str = "days", wells: pd.DataFrame | None = None) -> WellArrays:
    """Build the matrices for ``wells`` (default: all cohort wells). ``oil_column`` and
    ``days_column`` choose the final (``oil_bbl``, ``days``) or as-first-reported
    (``oil_bbl_first``, ``days_first``) panel."""
    w = (panel.wells if wells is None else wells).reset_index(drop=True)
    index = pd.Series(np.arange(len(w)), index=w.api)
    m = panel.months[panel.months.api.isin(index.index)]
    m = m.merge(w[["api", "first_ym"]], on="api")
    m = m[(m.ym <= RECORD_END_YM)]
    k = (m.ym - m.first_ym).to_numpy()
    keep = (k >= 0) & (k < n_months)
    m, k = m[keep], k[keep]
    rows = index[m.api].to_numpy()
    oil = np.zeros((len(w), n_months))
    rate = np.full((len(w), n_months), np.nan)
    o = m[oil_column].fillna(0).to_numpy(float)
    d = m[days_column].fillna(0).to_numpy(float)
    oil[rows, k] = o
    days = np.zeros((len(w), n_months))
    days[rows, k] = d
    good = (o > 0) & (d >= MIN_DAYS)
    rate[rows[good], k[good]] = o[good] / d[good]
    available = np.minimum(RECORD_END_YM - w.first_ym.to_numpy() + 1, n_months)
    return WellArrays(api=w.api.to_numpy(), rate=rate, oil=oil, days=days,
                      available=available,
                      lateral_ft=w.lateral_ft.to_numpy(float), vintage=w.vintage.to_numpy(),
                      first_operator=w.first_operator.to_numpy(),
                      loc_id=w.loc_id.to_numpy(),
                      township=(w.township + "-" + w["range"]).to_numpy())
