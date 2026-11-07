"""The command line, end to end on small runs with the bundled model and data."""

from __future__ import annotations

import json

import duckdb
import pandas as pd
import pytest

from declinecast.cli import main


def test_datasets_quality_sql(capsys):
    main(["datasets"])
    assert "8,344 wells" in capsys.readouterr().out
    main(["quality"])
    out = capsys.readouterr().out
    assert "9,379,877 rows" in out and "development" in out
    main(["sql", "--query", "SELECT count(*) AS n FROM wells WHERE vintage >= 2020"])
    assert "2929" in capsys.readouterr().out


def test_sql_cannot_read_files(tmp_path):
    with pytest.raises(duckdb.Error):
        main(["sql", "--query", f"SELECT * FROM read_csv('{tmp_path}/x.csv')"])


def test_test_split_is_locked(tmp_path):
    with pytest.raises(SystemExit, match="locked"):
        main(["score", "--split", "test", "--out", str(tmp_path / "t")])


def test_predict_from_csv(tmp_path, panel):
    w = panel.wells[panel.wells.split == "development"].iloc[0]
    m = panel.months[panel.months.api == w.api].sort_values("ym").head(12)
    df = pd.DataFrame({"well_id": "mine",
                       "month": [f"{y // 12}-{y % 12 + 1:02d}" for y in m.ym],
                       "oil_bbl": m.oil_bbl, "days": m.days, "lateral_ft": w.lateral_ft})
    df.to_csv(tmp_path / "wells.csv", index=False)
    main(["predict", "--csv", str(tmp_path / "wells.csv"), "--out", str(tmp_path / "f.csv"),
          "--draws", "100"])
    f = pd.read_csv(tmp_path / "f.csv")
    assert set(f.horizon_months) == {36, 60}
    assert (f["q0.1"] <= f["q0.5"]).all() and (f["q0.5"] <= f["q0.9"]).all()
    assert (f["q0.1"] >= f.observed_to_date_bbl).all()


def test_validate_score_tableau(tmp_path):
    v, s, t = tmp_path / "v", tmp_path / "s", tmp_path / "t"
    main(["validate", "--out", str(v), "--n-sbc", "40", "--n-per-history", "20",
          "--draws", "50"])
    assert json.loads((v / "validation.json").read_text())["settings"]["n_sbc"] == 40
    main(["score", "--split", "development", "--out", str(s), "--max-wells", "30",
          "--draws", "50", "--bootstrap", "50"])
    scores = pd.read_csv(s / "scores.csv")
    assert {"network", "multiplier", "type_curve", "arps_ls"} <= set(scores.method)
    with pytest.raises(SystemExit, match="not empty"):
        main(["score", "--split", "development", "--out", str(s), "--max-wells", "5"])
    main(["tableau", "--score", str(s), "--validation", str(v), "--out", str(t)])
    tables = json.loads((t / "tables.json").read_text())["tables"]
    for name in ("reliability.csv", "forecasts.csv", "pp.csv", "coverage.csv", "scores.csv"):
        assert tables[name]["rows"] > 0, name
    rel = pd.read_csv(t / "reliability.csv")
    assert rel.observed.between(0, 1).all()


def test_predict_creates_the_output_folder(tmp_path, panel):
    api = panel.wells[panel.wells.split == "test"].api.iloc[0]
    out = tmp_path / "new" / "folder" / "one.csv"
    main(["predict", "--api", api, "--history", "12", "--horizons", "36", "--out", str(out),
          "--draws", "50"])
    assert out.exists()
