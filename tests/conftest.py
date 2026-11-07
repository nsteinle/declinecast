"""Shared fixtures: a miniature copy of the raw ECMC files with every defect the panel build
handles, and the bundled data."""

from __future__ import annotations

import struct
import warnings
import zipfile
from pathlib import Path

import pytest

warnings.filterwarnings("ignore", category=RuntimeWarning)

OLD_HEADER = ("DocNum,ReportMonth,ReportYear,DaysProduced,AcceptedDate,Revised,OpName,OpNumber,"
              "FacilityId,ApiCountyCode,ApiSequenceNumber,ApiSidetrack,Well,WellStatus,"
              "FormationCode,OilProduced,OilSales,OilAdjustment,OilGravity,GasProduced,GasSales,"
              "GasBtuSales,GasUsedOnLease,GasShrinkage,GasPressureTubing,GasPressureCasing,"
              "WaterProduced,WaterPressureTubing,WaterPressureCasing,FlaredVented,BomInvent,"
              "EomInvent")
# 2025: three renamed columns, FlaredVented moved ahead of WaterProduced
NEW_HEADER = ("DocNum,ReportMonth,ReportYear,DaysProduced,AcceptedDate,Revised,OpName,OpNumber,"
              "FacilityId,ApiCountyCode,ApiSequenceNumber,ApiSidetrack,Well,WellStatus,"
              "FormationCode,OilProduced,OilSales,OilAdjustment,OilGravity,GasProduced,GasSales,"
              "GasBtuSales,GasUsedOnLease,GasSrinkage,GasPressureTubing,GasPressureCasing,"
              "FlaredVented,WaterProduced,WaterPressureTubing,WaterPressureCasing,BOMInvent,"
              "EOMInvent")


def _old(doc, month, year, days, accepted, revised, op, county, seq, side, status, fm, oil, gas,
         water, flared=""):
    return (f"{doc},{month},{year},{days},{accepted},{revised},{op},1,1,{county},{seq},{side},"
            f"W,{status},\"{fm}\",{oil},,,,{gas},,,,,,,{water},,,{flared},,")


def _new(doc, month, year, days, accepted, revised, op, county, seq, side, status, fm, oil, gas,
         water, flared="NULL"):
    return (f"{doc},{month},{year},{days},{accepted},{revised},{op},1,1,{county},{seq},{side},"
            f"W,{status},\"{fm}\",{oil},NULL,NULL,NULL,{gas},NULL,NULL,NULL,NULL,NULL,NULL,"
            f"{flared},{water},NULL,NULL,NULL,NULL")


def _dbf(records: list[dict], fields: list[tuple[str, int]]) -> bytes:
    """A dBASE III table with character fields (enough for ``read_dbf``)."""
    n, rec_len = len(records), 1 + sum(w for _, w in fields)
    header_len = 32 + 32 * len(fields) + 1
    out = bytearray(struct.pack("<BBBBIHH20x", 3, 126, 1, 1, n, header_len, rec_len))
    for name, width in fields:
        out += struct.pack("<11sc4xBB14x", name.encode(), b"C", width, 0)
    out += b"\r"
    for r in records:
        out += b" " + b"".join(str(r.get(name, "")).ljust(w)[:w].encode("latin-1")
                               for name, w in fields)
    out += b"\x1a"
    return bytes(out)


WELLMASTER = [
    # county, seq, spud, md, tvd, lat, lon, twp, rng, sec, loc
    ("123", "40001", "20220601", "17000", "7000", "40.30", "-104.70", "5N", "65W", "10", "900"),
    ("123", "40002", "20220601", "16500", "7000", "40.30", "-104.70", "5N", "65W", "10", "900"),
    ("123", "40003", "20220701", "9000", "7100", "40.40", "-104.60", "6N", "64W", "3", "901"),
    ("123", "40004", "20220701", "7000", "7100", "40.40", "-104.60", "6N", "64W", "3", "902"),
    ("001", "05242", "19800101", "6000", "6000", "39.90", "-104.90", "1S", "67W", "1", "903"),
]
FIELDS = [("API_County", 3), ("API_Seq", 5), ("Spud_Date", 8), ("Max_MD", 8), ("Max_TVD", 8),
          ("Field_Name", 12), ("Well_Class", 4), ("Latitude", 8), ("Longitude", 9),
          ("Township", 4), ("Range", 4), ("Section", 3), ("Loc_ID", 6)]


def write_raw(dest: Path) -> Path:
    """Three yearly files and a well master. Wells:

    * 40001: horizontal Niobrara, first production 2023-01, commingled with Codell in two
      months, one record revised and re-exported in the 2025 file with a later accepted date;
    * 40002: horizontal, first production 2023-03, a shut-in month (0 days) and a missing month;
    * 40003: a short lateral (1,900 ft), so screened out;
    * 40004: impossible geometry (TVD > MD), screened out;
    * 05242: an old vertical well in another formation, screened out.
    """
    dest.mkdir(parents=True, exist_ok=True)
    y23 = [OLD_HEADER]
    for m in range(1, 13):
        y23.append(_old(1000 + m, m, 2023, 30, f"{m:02d}/20/2023 10:00:00", "", "OPCO A",
                        "123", "40001", "00", "PR", "NBRR", 3000 - 150 * m, 4000, 900, "7"))
        if m in (2, 3):
            y23.append(_old(1100 + m, m, 2023, 28, f"{m:02d}/20/2023 10:00:00", "", "OPCO A",
                            "123", "40001", "00", "PR", "CODL", 400, 500, 100))
        if m >= 3 and m != 6 and m != 8:
            days = 0 if m == 7 else 31
            oil = 0 if m == 7 else 2500 - 100 * m
            y23.append(_old(1200 + m, m, 2023, days, f"{m:02d}/21/2023 10:00:00", "",
                            "OPCO B", "123", "40002", "00", "pr" if m == 4 else "PR",
                            "nbrr" if m == 5 else "NBRR", oil, 3000, 800))
        y23.append(_old(1300 + m, m, 2023, 30, f"{m:02d}/22/2023 10:00:00", "", "OPCO C",
                        "123", "40003", "00", "PR", "NBRR", 1500, 2000, 500))
        y23.append(_old(1400 + m, m, 2023, 30, f"{m:02d}/22/2023 10:00:00", "", "OPCO C",
                        "123", "40004", "00", "PR", "CODL", 1200, 1500, 400))
        y23.append(_old(1500 + m, m, 2023, 31, f"{m:02d}/23/2023 10:00:00", "", "TINY CO",
                        "001", "05242", "00", "PR", "DSND", 100, "", 100))
    y24 = [OLD_HEADER]
    for m in range(1, 13):
        y24.append(_old(2000 + m, m, 2024, 31, f"{m:02d}/20/2024 10:00:00", "", "OPCO A",
                        "123", "40001", "00", "PR", "NBRR", 1100 - 30 * m, 2500, 600))
        y24.append(_old(2200 + m, m, 2024, 30, f"{m:02d}/21/2024 10:00:00", "", "OPCO B",
                        "123", "40002", "00", "PR", "NBRR", 900 - 20 * m, 2000, 500))
    y25 = [NEW_HEADER]
    # a re-export of 40001's September 2024 record, revised, accepted later: the final panel
    # takes it, the first-reported panel keeps the original
    y25.append(_new(3001, "09", 2024, 30, "2025-02-01 00:00:00.000", "Y", "OPCO A", "123",
                    "40001", "00", "PR", "NBRR", 999, 2400, 650, "55"))
    # an unchanged re-export of 40002's October 2024 record (identical values)
    y25.append(_new(2210, "10", 2024, 30, "2024-10-21 10:00:00.000", "NULL", "OPCO B", "123",
                    "40002", "00", "PR", "NBRR", 700, 2000, 500))
    for m in range(1, 7):
        y25.append(_new(4000 + m, f"{m:02d}", 2025, 31, f"2025-{m:02d}-20 00:00:00.000", "NULL",
                        "OPCO A", "123", "40001", "NULL", "PR", "NBRR", 700 - 10 * m, 2000,
                        "NULL", "12"))
    (dest / "2023_prod_reports.csv").write_text("\n".join(y23) + "\n")
    (dest / "2024_prod_reports.csv").write_text("\n".join(y24) + "\n")
    (dest / "2025_prod_reports.csv").write_text("\n".join(y25) + "\n")
    recs = [dict(zip([f for f, _ in FIELDS],
                     [c, s, sp, md, tvd, "WATTENBERG", "OW", la, lo, tw, rg, se, loc],
                     strict=True))
            for c, s, sp, md, tvd, la, lo, tw, rg, se, loc in WELLMASTER]
    with zipfile.ZipFile(dest / "WELLS_SHP.ZIP", "w") as zf:
        zf.writestr("Wells.dbf", _dbf(recs, FIELDS))
    return dest


#: rules for the miniature files (record opens January 2023)
MINI_RULES = {"first_year": 2023, "min_months": 12, "min_lateral_ft": 3000,
              "min_spud_year": 2012}
MINI_YEARS = (2023, 2024, 2025)


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory) -> Path:
    return write_raw(tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="session")
def panel():
    from declinecast.datasets import load_panel

    return load_panel()


@pytest.fixture(scope="session")
def model():
    from declinecast.modelio import load_model

    return load_model()
