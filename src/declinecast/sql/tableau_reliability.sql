-- Reliability on real wells: for each method, history and horizon, the share of actual
-- cumulatives inside each central interval, against the interval's nominal level. A calibrated
-- method sits on the diagonal.
WITH long AS (
    UNPIVOT forecasts
    ON covered_50, covered_68, covered_80, covered_90, covered_95
    INTO NAME interval VALUE inside
)
SELECT
    split, as_reported, history AS history_months, horizon AS horizon_months, method,
    CAST(replace(interval, 'covered_', '') AS INTEGER) / 100.0 AS nominal,
    avg(CAST(inside AS DOUBLE)) AS observed,
    count(*) AS wells
FROM long
WHERE NOT isnan(wis)
GROUP BY ALL
ORDER BY ALL
