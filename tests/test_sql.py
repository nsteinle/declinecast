"""The DuckDB panel build against the independent pandas reference, on the miniature raw files
and, when the real ECMC files are present, on a 1% sample of wells from all 9.4 million rows."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pandas_reference as ref
import pytest
from conftest import MINI_RULES, MINI_YEARS

from declinecast import sql
from declinecast.dbf import read_dbf
from declinecast.ecmc import build_panel, build_tables, read_wellmaster


@pytest.fixture(scope="module")
def built(raw_dir):
    con = sql.connect_files()
    build_tables(con, raw_dir, years=MINI_YEARS, rules=MINI_RULES)
    yield con
    con.close()


def _frame(con, q):
    return con.execute(q).df()


def test_raw_rows_read_by_name(built, raw_dir):
    raw = _frame(built, "SELECT * FROM raw")
    assert len(raw) == sum(len(p.read_text().splitlines()) - 1
                           for p in sorted(raw_dir.glob("*.csv")))
    # the 2025 file moved FlaredVented ahead of WaterProduced: water must not read as flared
    r25 = raw[(raw.file_year == 2025) & (raw.year == 2024)]
    assert set(r25.water_bbl) == {650.0, 500.0}
    assert raw.water_bbl[(raw.file_year == 2025) & (raw.year == 2025)].isna().all()  # "NULL"
    assert raw.accepted.notna().all()                      # both date formats parsed
    assert (raw.api.str.len() == 12).all()                 # a NULL sidetrack becomes 00
    assert raw.formation.str.isupper().all()
    assert raw.status_not_upper.sum() == 1


def test_well_months_match_pandas(built, raw_dir):
    raw = ref.read_raw(sorted(raw_dir.glob("*.csv")))
    for table, final in (("well_months", True), ("well_months_first", False)):
        got = _frame(built, f"SELECT * FROM {table} ORDER BY api, ym").reset_index(drop=True)
        want = ref.well_months(ref.records(raw, final=final))
        assert len(got) == len(want)
        for c in ("api", "ym", "formation_rows", "versions", "status", "operator"):
            assert got[c].astype(str).tolist() == want[c].astype(str).tolist(), c
        for c in ("oil_bbl", "gas_mcf", "water_bbl", "days"):
            np.testing.assert_allclose(got[c].astype(float), want[c].astype(float), err_msg=c)
        assert got.revised.tolist() == want.revised.tolist()


def test_dedupe_and_commingling(built):
    wm = _frame(built, "SELECT * FROM well_months WHERE api = '051234000100'")
    first = _frame(built, "SELECT * FROM well_months_first WHERE api = '051234000100'")
    sep = wm[(wm.year == 2024) & (wm.month == 9)].iloc[0]
    assert sep.oil_bbl == 999 and sep.versions == 2 and sep.revised   # revision kept
    assert first[(first.year == 2024) & (first.month == 9)].oil_bbl.iloc[0] == 1100 - 30 * 9
    feb = wm[(wm.year == 2023) & (wm.month == 2)].iloc[0]
    assert feb.formation_rows == 2 and feb.oil_bbl == 3000 - 300 + 400  # summed, not deduped
    assert feb.days == 30                                                # longest row


def test_wells_and_cohort(built, raw_dir):
    raw = ref.read_raw(sorted(raw_dir.glob("*.csv")))
    rec = ref.records(raw)
    want = ref.wells(rec, ref.well_months(rec))
    got = _frame(built, "SELECT * FROM wells ORDER BY api").reset_index(drop=True)
    for c in ("api", "first_ym", "last_ym", "producing_months", "first_operator",
              "primary_formation"):
        assert got[c].astype(str).tolist() == want[c].astype(str).tolist(), c
    cohort = _frame(built, "SELECT * FROM cohort ORDER BY api").set_index("api")
    assert cohort.in_cohort.sum() == 2
    assert cohort.loc["051234000300", "horizontal"] == False  # noqa: E712 (1,900 ft)
    assert cohort.loc["051234000400", "impossible_geometry"] == True  # noqa: E712
    assert cohort.loc["050010524200", "dj_formation"] == False  # noqa: E712


def test_dbf_reader_and_wellmaster(raw_dir):
    df = read_dbf(raw_dir / "WELLS_SHP.ZIP", "Wells.dbf", ["API_Seq", "Max_MD"])
    assert df.API_Seq.tolist()[:2] == ["40001", "40002"]
    with pytest.raises(KeyError):
        read_dbf(raw_dir / "WELLS_SHP.ZIP", "Wells.dbf", ["NotAField"])
    wm = read_wellmaster(raw_dir / "WELLS_SHP.ZIP")
    assert wm.api10.str.len().eq(10).all() and wm.max_md.notna().all()


def test_build_panel_writes_checksummed_extract(raw_dir, tmp_path, monkeypatch):
    import declinecast.ecmc as ecmc

    monkeypatch.setattr(ecmc, "COHORT_RULES", MINI_RULES)
    meta = build_panel(raw_dir, tmp_path, years=MINI_YEARS)
    from declinecast.datasets import load_panel

    p = load_panel(tmp_path)
    assert len(p.wells) == meta["quality"]["cohort_funnel"]["cohort"] == 2
    assert {"oil_bbl_first", "days_first"} <= set(p.months.columns)


def test_connect_blocks_file_access(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a\n1\n")
    with pytest.raises(duckdb.Error):
        sql.query(f"SELECT * FROM read_csv({sql.literal(f.as_posix())})")
    assert sql.query("SELECT 41 + 1 AS x").x.iloc[0] == 42


def test_ident_and_literal_quote():
    assert sql.ident('a"b') == '"a""b"'
    assert sql.literal("o'k") == "'o''k'"


RAW = os.environ.get("DECLINECAST_RAW")


@pytest.mark.slow
@pytest.mark.skipif(not RAW, reason="set DECLINECAST_RAW to a folder of the raw ECMC files")
def test_real_files_match_pandas_on_a_sample():
    raw_dir = Path(RAW)
    con = sql.connect_files()
    build_tables(con, raw_dir)
    assert con.execute("SELECT count(*) FROM raw").fetchone()[0] == 9_379_877
    assert con.execute("SELECT count(*) FROM cohort WHERE in_cohort").fetchone()[0] == 8_344
    sample = lambda api: api.str[8:10] == "17"  # noqa: E731 (about 1% of wells)
    raw = ref.read_raw(sorted(raw_dir.glob("*_prod_reports.csv")), filter_api=sample,
                       chunksize=500_000)
    want = ref.well_months(ref.records(raw))
    got = con.execute("SELECT * FROM well_months WHERE substr(api, 9, 2) = '17' "
                      "ORDER BY api, ym").df()
    assert len(got) == len(want) > 10_000
    for c in ("oil_bbl", "gas_mcf", "days", "formation_rows"):
        np.testing.assert_allclose(got[c].astype(float), want[c].astype(float), err_msg=c)
    pd.testing.assert_series_equal(got.api, want.api, check_names=False)
