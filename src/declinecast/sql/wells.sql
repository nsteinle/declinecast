-- One row per well: its producing span, its primary formation (the formation with the most
-- oil-equivalent over the record), and the operator that filed its first producing month.
WITH formation_boe AS (
    SELECT api, formation, SUM(coalesce(oil_bbl, 0) + coalesce(gas_mcf, 0) / 6) AS boe
    FROM records
    GROUP BY api, formation
),
producing AS (
    SELECT * FROM well_months WHERE oil_bbl > 0 OR gas_mcf > 0
)
SELECT
    p.api,
    MIN(p.ym)                         AS first_ym,
    MAX(p.ym)                         AS last_ym,
    COUNT(*)                          AS producing_months,
    arg_min(p.operator, p.ym)         AS first_operator,
    arg_max(p.operator, p.ym)         AS last_operator,
    any_value(f.primary_formation)    AS primary_formation
FROM producing p
JOIN (SELECT api, arg_max(formation, boe) AS primary_formation
      FROM formation_boe GROUP BY api) f USING (api)
GROUP BY p.api
