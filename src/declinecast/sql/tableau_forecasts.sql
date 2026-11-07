-- One row per well, history, horizon and method: the forecast's P10, P50 and P90 (reserves
-- convention: P90 is the low case), the actual cumulative, and the well's descriptors for maps
-- and filters.
SELECT
    f.split, f.api, w.first_operator AS operator, w.vintage, w.primary_formation AS formation,
    round(w.lateral_ft) AS lateral_ft, w.latitude, w.longitude,
    f.history AS history_months, f.horizon AS horizon_months,
    CASE f.method WHEN 'network' THEN 'Network (declinecast)'
                  WHEN 'network_unadjusted' THEN 'Network before recalibration'
                  WHEN 'multiplier' THEN 'History multiplier'
                  WHEN 'type_curve' THEN 'Type curve (lateral only)'
                  WHEN 'arps_ls' THEN 'Least-squares Arps' END AS method,
    f.observed_to_date_bbl,
    f."q0.9" AS p10_high_bbl, f."q0.5" AS p50_bbl, f."q0.1" AS p90_low_bbl,
    f.actual_bbl,
    f."q0.5" / f.actual_bbl - 1 AS p50_error_fraction,
    f.covered_80, f.covered_90, f.wis, f.ood_flag, f.as_reported
FROM forecasts f
JOIN wells w USING (api)
ORDER BY f.history, f.horizon, f.method, f.api
