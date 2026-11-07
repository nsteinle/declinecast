-- PP plot for simulation-based calibration: the empirical CDF of the true values' normalized
-- ranks, on a grid of 101 points, per quantity. A calibrated posterior gives a diagonal line,
-- inside the KS band of half-width {band}.
WITH long AS (
    UNPIVOT ranks ON COLUMNS(*) INTO NAME quantity VALUE rank
),
grid AS (SELECT range / 100.0 AS level FROM range(0, 101))
SELECT
    quantity, level AS expected,
    avg(CAST(rank <= level AS DOUBLE)) AS observed,
    greatest(level - {band}, 0) AS band_low,
    least(level + {band}, 1) AS band_high
FROM long CROSS JOIN grid
GROUP BY ALL
ORDER BY quantity, expected
