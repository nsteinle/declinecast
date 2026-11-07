"""The Colorado ECMC production files: where they are, how to fetch them, and how the panel is
built from them in DuckDB SQL.

The Energy & Carbon Management Commission publishes one CSV per year of per-well, per-month,
per-formation production, a data dictionary, and a well master and directional surveys as
shapefiles. ``fetch`` downloads only those published files, one at a time; it never crawls.
``build_panel`` turns the raw files into the bundled cohort extract.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from declinecast import sql
from declinecast._version import __version__
from declinecast.dbf import read_dbf

__all__ = ["BASE_URL", "PRODUCTION_YEARS", "file_urls", "fetch", "read_wellmaster",
           "build_tables", "build_panel", "COHORT_RULES"]

BASE_URL = "https://ecmc.state.co.us/documents/data/downloads"
DOWNLOAD_PAGE = "https://ecmc.colorado.gov/data-maps-reports/downloadable-data-documents"
DATA_DICTIONARY = f"{BASE_URL}/production/production_record_data_dictionary.htm"
PRODUCTION_YEARS = tuple(range(2013, 2026))

#: The cohort screens (FINDINGS section 1). Vintages before 2014 are excluded because the
#: record opens in January 2013, so a well that "starts" in 2013 may have started earlier.
COHORT_RULES = {"first_year": 2014, "min_months": 12, "min_lateral_ft": 3000,
                "min_spud_year": 2012}

WELLMASTER_FIELDS = ["API_County", "API_Seq", "Spud_Date", "Max_MD", "Max_TVD", "Field_Name",
                     "Well_Class", "Latitude", "Longitude", "Township", "Range", "Section",
                     "Loc_ID"]


def file_urls(years=PRODUCTION_YEARS) -> dict[str, str]:
    """The published file names and their URLs."""
    urls = {f"{y}_prod_reports.zip": f"{BASE_URL}/production/{y}_prod_reports.zip"
            for y in years}
    urls["WELLS_SHP.ZIP"] = f"{BASE_URL}/gis/WELLS_SHP.ZIP"
    return urls


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(dest: str | Path, years=PRODUCTION_YEARS, pause: float = 2.0,
          overwrite: bool = False) -> dict:
    """Download the published production files and the well master into ``dest``.

    Requests go one at a time to the exact URLs listed by ``file_urls``, with a descriptive
    User-Agent and a pause between them. ECMC's robots.txt disallows ``/documents/`` to
    crawlers; this is not a crawler (it follows no links and requests nothing it was not told
    to), but run it rarely and only when you need the raw files. Returns the provenance record
    that is also written to ``dest/provenance.json``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    agent = f"declinecast/{__version__} (+https://github.com/nsteinle/declinecast)"
    files = []
    for i, (name, url) in enumerate(file_urls(years).items()):
        target = dest / name
        if overwrite or not target.exists():
            if i:
                time.sleep(pause)
            req = urllib.request.Request(url, headers={"User-Agent": agent})
            with urllib.request.urlopen(req, timeout=600) as resp, open(target, "wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
        files.append({"name": name, "url": url, "bytes": target.stat().st_size,
                      "sha256": _sha256(target)})
    record = {
        "source": "Colorado Energy & Carbon Management Commission (ECMC)",
        "download_page": DOWNLOAD_PAGE,
        "data_dictionary": DATA_DICTIONARY,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files,
    }
    (dest / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def read_wellmaster(zip_path: str | Path) -> pd.DataFrame:
    """The well master's attributes, one row per well (API-10: state, county, sequence).

    Where an API-10 appears more than once, the record with the deepest measured depth is kept,
    since that is the wellbore whose lateral the production comes from."""
    raw = read_dbf(zip_path, "Wells.dbf", WELLMASTER_FIELDS)
    num = lambda s: pd.to_numeric(s, errors="coerce")  # noqa: E731
    df = pd.DataFrame({
        "api10": "05" + raw.API_County.str.zfill(3) + raw.API_Seq.str.zfill(5),
        "spud_date": pd.to_datetime(raw.Spud_Date, format="%Y%m%d", errors="coerce"),
        "max_md": num(raw.Max_MD),
        "max_tvd": num(raw.Max_TVD),
        "field_name": raw.Field_Name,
        "well_class": raw.Well_Class,
        "latitude": num(raw.Latitude),
        "longitude": num(raw.Longitude),
        "township": raw.Township,
        "range": raw.Range,
        "section": raw.Section,
        "loc_id": num(raw.Loc_ID).astype("Int64"),
    })
    df = (df.sort_values(["api10", "max_md"], ascending=[True, False], kind="stable")
            .drop_duplicates("api10").reset_index(drop=True))
    df["spud_date"] = df["spud_date"].astype("datetime64[us]")
    return df


def _production_files(raw_dir: Path, years) -> list[Path]:
    """The yearly CSVs, extracted from their zips next to them if needed."""
    out = []
    for y in years:
        csv = raw_dir / f"{y}_prod_reports.csv"
        if not csv.exists():
            zpath = raw_dir / f"{y}_prod_reports.zip"
            with zipfile.ZipFile(zpath) as zf:
                members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
                if len(members) != 1:
                    raise ValueError(f"{zpath.name}: expected one CSV, found {members}")
                with zf.open(members[0]) as src, open(csv, "wb") as dst:
                    while chunk := src.read(1 << 22):
                        dst.write(chunk)
        out.append(csv)
    return out


def build_tables(con, raw_dir: str | Path, years=PRODUCTION_YEARS, rules=None) -> None:
    """Create the tables ``raw``, ``records`` (final versions), ``records_first``,
    ``well_months``, ``well_months_first``, ``wells``, ``wellmaster`` and ``cohort`` on the
    connection ``con`` (from ``sql.connect_files``)."""
    raw_dir = Path(raw_dir)
    rules = {**COHORT_RULES, **(rules or {})}
    files = ", ".join(sql.literal(p.as_posix()) for p in _production_files(raw_dir, years))
    con.execute("CREATE OR REPLACE TABLE raw AS " + sql.load("raw_rows", files=files))
    con.execute("CREATE OR REPLACE TABLE records AS "
                + sql.load("records", direction="DESC"))
    con.execute("CREATE OR REPLACE TABLE well_months AS " + sql.load("well_months"))
    con.execute("CREATE OR REPLACE TABLE wells AS " + sql.load("wells"))
    # the as-first-reported panel: the earliest accepted version of each record
    con.execute("CREATE OR REPLACE TABLE records_first AS "
                + sql.load("records", direction="ASC"))
    con.execute("CREATE OR REPLACE TABLE well_months_first AS "
                + sql.load("well_months").replace("FROM records", "FROM records_first"))
    con.register("wellmaster_df", read_wellmaster(raw_dir / "WELLS_SHP.ZIP"))
    con.execute("CREATE OR REPLACE TABLE wellmaster AS SELECT * FROM wellmaster_df")
    con.unregister("wellmaster_df")
    con.execute("CREATE OR REPLACE TABLE cohort AS "
                + sql.load("cohort", **{k: str(int(v)) for k, v in rules.items()}))


def build_panel(raw_dir: str | Path, out_dir: str | Path, years=PRODUCTION_YEARS,
                threads: int | None = None) -> dict:
    """Build the cohort extract from the raw files: ``wells.parquet``, ``months.parquet`` and
    ``panel.json`` (row counts, checksums, the quality report and the source files' checksums).
    """
    from declinecast.quality import raw_quality_report

    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with sql.connect_files(threads) as con:
        build_tables(con, raw_dir, years)
        report = raw_quality_report(con)
        for name in ("wells", "months"):
            target = out_dir / f"{name}.parquet"
            con.execute(f"COPY ({sql.load('extract_' + name)}) TO {sql.literal(target.as_posix())}"
                        " (FORMAT parquet, COMPRESSION zstd)")
    meta = {
        "source": "Colorado Energy & Carbon Management Commission (ECMC), per-well monthly "
                  "production files and well master",
        "download_page": DOWNLOAD_PAGE,
        "data_dictionary": DATA_DICTIONARY,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "built_with": f"declinecast {__version__}",
        "cohort_rules": COHORT_RULES,
        "files": {name: {"sha256": _sha256(out_dir / f"{name}.parquet"),
                         "bytes": (out_dir / f"{name}.parquet").stat().st_size}
                  for name in ("wells", "months")},
        "sources": [{"name": p.name, "sha256": _sha256(p)}
                    for p in sorted(raw_dir.glob("*.zip")) + sorted(raw_dir.glob("*.ZIP"))],
        "quality": report,
    }
    (out_dir / "panel.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return meta
