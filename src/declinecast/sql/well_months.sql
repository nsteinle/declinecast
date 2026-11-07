-- One row per (well, calendar month). Colorado reports a commingled well once per producing
-- formation, so the formation rows are summed, never de-duplicated. Days produced is the
-- longest of the formation rows (the well, not each zone, was on line), and the status is the
-- status of the formation row with the most oil-equivalent.
SELECT
    api,
    year,
    month,
    year * 12 + month - 1                                             AS ym,
    coalesce(SUM(oil_bbl), 0)                                         AS oil_bbl,
    coalesce(SUM(gas_mcf), 0)                                         AS gas_mcf,
    coalesce(SUM(water_bbl), 0)                                       AS water_bbl,
    MAX(days)                                                         AS days,
    COUNT(*)                                                          AS formation_rows,
    bool_or(revised)                                                  AS revised,
    MAX(versions)                                                     AS versions,
    arg_max(status, coalesce(oil_bbl, 0) + coalesce(gas_mcf, 0) / 6) AS status,
    arg_max(operator, coalesce(oil_bbl, 0) + coalesce(gas_mcf, 0) / 6) AS operator
FROM records
GROUP BY api, year, month
