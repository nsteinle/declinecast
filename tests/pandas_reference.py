"""An independent pandas implementation of the panel build, for checking the DuckDB SQL.

It follows the rules in FINDINGS section 1, written separately from the SQL: read by column
name, treat empty and "NULL" as missing, parse both date formats, upper-case codes, keep one
version of each (well, month, formation) record (latest accepted for the final panel, earliest
for the first-reported one), then sum formation rows into well-months.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["DocNum", "ReportMonth", "ReportYear", "DaysProduced", "AcceptedDate", "Revised",
        "OpName", "ApiCountyCode", "ApiSequenceNumber", "ApiSidetrack", "WellStatus",
        "FormationCode", "OilProduced", "GasProduced", "WaterProduced"]


def read_raw(paths, filter_api=None, chunksize=None) -> pd.DataFrame:
    frames = []
    for p in paths:
        year = int(re.search(r"(\d{4})", Path(p).name).group(1))
        reader = pd.read_csv(p, dtype=str, usecols=COLS, keep_default_na=False,
                             chunksize=chunksize or 10**9)
        for df in reader:
            df = df.replace({"": np.nan, "NULL": np.nan})
            side = df.ApiSidetrack.fillna("00").str.strip().str.zfill(2)
            df["api"] = ("05" + df.ApiCountyCode.str.strip().str.zfill(3)
                         + df.ApiSequenceNumber.str.strip().str.zfill(5) + side)
            if filter_api is not None:
                df = df[filter_api(df["api"])]
            a = pd.to_datetime(df.AcceptedDate, format="%m/%d/%Y %H:%M:%S", errors="coerce")
            b = pd.to_datetime(df.AcceptedDate, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
            frames.append(pd.DataFrame({
                "api": df.api,
                "year": df.ReportYear.astype(int), "month": df.ReportMonth.astype(int),
                "days": pd.to_numeric(df.DaysProduced, errors="coerce"),
                "oil_bbl": pd.to_numeric(df.OilProduced, errors="coerce"),
                "gas_mcf": pd.to_numeric(df.GasProduced, errors="coerce"),
                "water_bbl": pd.to_numeric(df.WaterProduced, errors="coerce"),
                "formation": df.FormationCode.str.strip().str.upper(),
                "status": df.WellStatus.str.strip().str.upper(),
                "operator": df.OpName.str.strip(),
                "accepted": a.fillna(b),
                "revised": df.Revised.fillna("").str.strip().str.upper().isin(["Y", "TRUE", "1"]),
                "doc_num": df.DocNum, "file_year": year,
            }))
    return pd.concat(frames, ignore_index=True)


def records(raw: pd.DataFrame, final: bool = True) -> pd.DataFrame:
    key = ["api", "year", "month", "formation"]
    r = raw.copy()
    r["versions"] = r.groupby(key, dropna=False).api.transform("size")
    r["_acc"] = r.accepted.isna()  # missing accepted dates sort last either way
    r = r.sort_values(["_acc", "accepted", "file_year", "doc_num"],
                      ascending=[True, not final, not final, not final], kind="stable")
    return r.drop_duplicates(key, keep="first").drop(columns="_acc")


def well_months(rec: pd.DataFrame) -> pd.DataFrame:
    r = rec.copy()
    r["boe"] = r.oil_bbl.fillna(0) + r.gas_mcf.fillna(0) / 6
    g = r.groupby(["api", "year", "month"])
    out = g.agg(oil_bbl=("oil_bbl", lambda s: s.fillna(0).sum()),
                gas_mcf=("gas_mcf", lambda s: s.fillna(0).sum()),
                water_bbl=("water_bbl", lambda s: s.fillna(0).sum()),
                days=("days", "max"), formation_rows=("api", "size"),
                revised=("revised", "any"), versions=("versions", "max")).reset_index()
    top = r.sort_values("boe", ascending=False, kind="stable").drop_duplicates(
        ["api", "year", "month"])
    out = out.merge(top[["api", "year", "month", "status", "operator"]],
                    on=["api", "year", "month"])
    out["ym"] = out.year * 12 + out.month - 1
    return out.sort_values(["api", "ym"]).reset_index(drop=True)


def wells(rec: pd.DataFrame, wm: pd.DataFrame) -> pd.DataFrame:
    r = rec.copy()
    r["boe"] = r.oil_bbl.fillna(0) + r.gas_mcf.fillna(0) / 6
    prim = (r.groupby(["api", "formation"]).boe.sum().reset_index()
            .sort_values("boe", ascending=False, kind="stable").drop_duplicates("api")
            .set_index("api").formation)
    p = wm[(wm.oil_bbl > 0) | (wm.gas_mcf > 0)]
    g = p.sort_values("ym").groupby("api")
    out = pd.DataFrame({"first_ym": g.ym.min(), "last_ym": g.ym.max(),
                        "producing_months": g.size(),
                        "first_operator": g.operator.first(),
                        "last_operator": g.operator.last()})
    out["primary_formation"] = prim.reindex(out.index)
    return out.reset_index().sort_values("api").reset_index(drop=True)
