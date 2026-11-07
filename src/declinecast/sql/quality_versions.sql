-- Records that occur more than once across files, and whether the copies disagree.
WITH k AS (
    SELECT api, year, month, formation,
           COUNT(*) AS n,
           COUNT(DISTINCT coalesce(oil_bbl, -1)) AS oil_values,
           COUNT(DISTINCT coalesce(days, -1)) AS day_values,
           COUNT(DISTINCT file_year) AS files
    FROM raw
    GROUP BY ALL
)
SELECT COUNT(*)                                   AS records,
       SUM((n > 1)::INTEGER)                      AS records_in_two_files,
       SUM((n > 1 AND oil_values > 1)::INTEGER)   AS copies_with_different_oil,
       SUM((n > 1 AND day_values > 1)::INTEGER)   AS copies_with_different_days,
       SUM(n) - COUNT(*)                          AS duplicate_rows_removed
FROM k
