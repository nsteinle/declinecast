"""The bundled extract: checksums, counts, splits and the quality report."""

from __future__ import annotations

import shutil
from importlib import resources

import numpy as np
import pytest

from declinecast.datasets import RECORD_END_YM, load_panel, well_arrays
from declinecast.quality import extract_quality_report


def test_counts_and_splits(panel):
    assert len(panel.wells) == 8344
    assert panel.wells.api.is_unique
    counts = panel.wells.split.value_counts()
    assert counts["development"] == 5415 and counts["test"] == 2929
    dev = panel.wells[panel.wells.split == "development"]
    assert dev.vintage.between(2014, 2019).all()
    assert (panel.wells[panel.wells.split == "test"].vintage >= 2020).all()
    assert (panel.wells.lateral_ft >= 3000).all()


def test_checksum_is_checked(tmp_path):
    with resources.as_file(resources.files("declinecast") / "data") as d:
        for name in ("wells.parquet", "months.parquet", "panel.json"):
            shutil.copy(d / name, tmp_path / name)
    load_panel(tmp_path)
    with open(tmp_path / "wells.parquet", "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(ValueError, match="checksum"):
        load_panel(tmp_path)


def test_well_arrays(panel):
    w = panel.wells.head(50)
    a = well_arrays(panel, wells=w)
    assert a.rate.shape == (50, 120)
    assert np.all(a.available == np.minimum(RECORD_END_YM - w.first_ym.to_numpy() + 1, 120))
    ok = np.isfinite(a.rate)
    np.testing.assert_allclose(a.rate[ok] * a.days[ok], a.oil[ok])
    assert np.all(a.days[ok] >= 3)


def test_extract_quality_report(panel):
    q = extract_quality_report(panel).set_index("split")
    assert q.loc["development", "wells"] == 5415
    assert q.loc["test", "wells"] == 2929
    assert (q.months_first_oil_differs < q.well_months * 0.01).all()


def test_raw_report_is_in_panel_json(panel):
    q = panel.meta["quality"]
    assert q["raw_rows"] == 9_379_877
    assert q["cohort_funnel"]["cohort"] == 8344
