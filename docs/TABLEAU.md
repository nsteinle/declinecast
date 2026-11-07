# A Tableau Public dashboard from the results

`declinecast tableau` writes the validation and scoring results as flat CSV tables, built by the
package's DuckDB SQL layer. This page lists the tables and describes a five-panel dashboard
built from them in Tableau Public, the free desktop app (now called Tableau Desktop Public
Edition). Tableau Public opens text files but not Parquet, and allows up to 15 million rows per
data source, so the tables here are well within its limits.

The tables are tested (`tests/test_cli.py::test_validate_score_tableau`); the steps in Tableau
are not, because they are done by hand in the app. Menu names can differ slightly between
versions.

## 1. Write the tables

```bash
declinecast validate --out results/validation
declinecast score --split development --out results/dev --max-wells 2000
declinecast score --split test --unlock-test --out results/test
declinecast tableau --score results/dev results/test --validation results/validation \
    --out results/tableau
```

| file | grain and use |
|---|---|
| `reliability.csv` | per split, method, history, horizon and nominal level: the share of actual cumulatives inside that central interval (`observed`) and the well count. **The first panel.** |
| `forecasts.csv` | one row per well, history, horizon and method: `p90_low_bbl`, `p50_bbl`, `p10_high_bbl` (reserves convention: P90 is the low case), `actual_bbl`, `p50_error_fraction`, `covered_80`, `covered_90`, `wis`, `ood_flag`, and the well's operator, vintage, formation, lateral, latitude and longitude |
| `scores.csv` | per split, history, horizon, method and well group (`all`, `not_flagged`, `flagged`): WIS, median absolute error (log10 and percent), median bias, coverage at 50, 68, 80, 90 and 95%, and the median width of the 80% range |
| `intervals.csv` | cluster-bootstrap 95% intervals (by operator and by pad) for the mean WIS, the mean absolute log error, and 80% and 90% coverage, with each method's difference from least-squares Arps |
| `examples.csv` | monthly cumulative-oil forecast fans for eight test wells seen for 12 months, with what they produced |
| `coverage.csv` | simulated wells: coverage of the posterior's credible intervals per quantity, history and level, with Clopper-Pearson intervals |
| `pp.csv` | simulated wells: the SBC PP plot, 101 points per quantity, with the KS band |
| `accuracy.csv` | simulated wells: Spearman correlation, bias, error and contraction (posterior width over prior width) per quantity and history |
| `tables.json` | every file's description, columns and row count, and the query or command it came from |

The files have different grains, so connect each one as its own data source rather than
joining them.

## 2. The dashboard

In Tableau Public: **Connect → To a File → Text file**, choose `reliability.csv`, then add the
other files with **Data → New Data Source**.

1. **Do the ranges hold up on real wells?** (`reliability.csv`). Filter `split` to `test` and
   `horizon_months` to 36. `nominal` on Columns, `observed` on Rows, `method` on Color,
   `history_months` on Columns before `nominal` (one small panel per history), mark type Line
   with markers. Add a diagonal reference (a calculated field equal to `nominal` on a
   synchronized dual axis). A method on the diagonal says what it means; one below it is
   overconfident. This is the panel to lead with, because it is the business question.
2. **Forecast against outcome** (`forecasts.csv`). Filter to the network, test split, 12 months
   of history, 36-month horizon. `actual_bbl` on Columns, `p50_bbl` on Rows, both on log axes,
   `covered_80` on Color, and error bars from `p90_low_bbl` to `p10_high_bbl` (a range band or
   a Gantt bar of size `p10_high_bbl - p90_low_bbl`). Points off the diagonal with a colored
   bar that misses it are the wells that surprised the model.
3. **Scores against the baselines** (`scores.csv` and `intervals.csv`). `history` on Columns,
   `wis` on Rows, `method` on Color, mark type Line, filtered to `wells = all` and one split and
   horizon. Add the operator-bootstrap interval from `intervals.csv` as a reference band.
4. **A few wells, month by month** (`examples.csv`). `month` on Columns; `q0.05`, `q0.1`,
   `q0.5`, `q0.9`, `q0.95` and `actual_bbl` on Rows (Measure Values), `api` on Rows, so each
   well gets its own row. The shaded fans are the 90% and 80% ranges, the line is the
   median, and the black line is what the well produced.
5. **Where the forecasts miss** (`forecasts.csv`). A map: `longitude` and `latitude`, filtered
   to the network at 12 months of history, `p50_error_fraction` on Color (diverging, centered
   at 0), `ood_flag` on Shape. A cluster of one color is a geological or operational effect the
   model does not know about.

For the calibration on simulated wells, which is the statistical check behind panel 1, add
`pp.csv` (`expected` on Columns, `observed` on Rows, `quantity` on Color, with `band_low` and
`band_high` as a reference band) and `coverage.csv` on a second dashboard.

Put the five sheets on one dashboard, and add a text box with the data credit: production and
well data courtesy of the Colorado Energy & Carbon Management Commission (ECMC); the extract is
derived and not an official ECMC product. Say also that the figures are forecasts of
cumulative oil for DJ Basin horizontal Niobrara and Codell wells, not reserves estimates.

## 3. Publish and update

**File → Save to Tableau Public As** saves the workbook to your Tableau Public profile. It needs
a free account, and everything saved there, including the data, is public. The tables hold only
public-record production, the package's forecasts and well locations from the public well
master; do not publish your own wells this way unless they can be public.

After a new scoring run, write the tables again into the same folder (they replace the old
files), then refresh the data sources in Tableau and save to Tableau Public again.
