"""Data-quality reports: on the raw ECMC files (``raw_quality_report``, written into the
bundle's ``panel.json``), and on the bundled cohort extract (``extract_quality_report``)."""

from __future__ import annotations

from declinecast import sql

__all__ = ["raw_quality_report", "extract_quality_report"]


def _rows(con, name: str, **fields) -> list[dict]:
    return con.execute(sql.load(name, **fields)).df().to_dict("records")


def raw_quality_report(con) -> dict:
    """Counts of every defect handled by the panel build, from the tables ``build_tables``
    created on ``con``."""
    from declinecast.ecmc import COHORT_RULES

    files = _rows(con, "quality_files")
    versions = _rows(con, "quality_versions")[0]
    well_months = _rows(con, "quality_well_months")[0]
    cohort = _rows(con, "quality_cohort", window_start=str(min(f["file_year"] for f in files)))[0]
    status = con.execute(
        "SELECT SUM(status_not_upper::INTEGER) FROM raw").fetchone()[0]
    first_month = con.execute("SELECT MIN(year * 12 + month - 1), MAX(year * 12 + month - 1) "
                              "FROM raw WHERE file_year = year").fetchone()
    as_int = lambda d: {k: (int(v) if v is not None else None) for k, v in d.items()}  # noqa: E731
    return {
        "files": [as_int(f) for f in files],
        "raw_rows": int(sum(f["rows"] for f in files)),
        "record_months": {"first": int(first_month[0]), "last": int(first_month[1])},
        "versions": as_int(versions),
        "status_codes_not_upper_case": int(status or 0),
        "well_months": as_int(well_months),
        "cohort_funnel": as_int(cohort),
        "cohort_rules": COHORT_RULES,
    }


def extract_quality_report(panel=None):
    """The bundled extract's quality table (one row per split), from DuckDB SQL."""
    from declinecast.datasets import RECORD_END_YM, TEST_FIRST_VINTAGE, load_panel
    from declinecast.simulator import MIN_DAYS

    panel = panel or load_panel()
    q = sql.load("quality_extract", test_first=str(TEST_FIRST_VINTAGE),
                 record_end=str(RECORD_END_YM), min_days=str(MIN_DAYS))
    wells = panel.wells.drop(columns=["spud_date", "split"])
    return sql.query(q, months=panel.months, wells=wells)
