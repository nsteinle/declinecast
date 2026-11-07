-- Statewide well-month defects, on the final panel.
SELECT
    COUNT(*)                                                         AS well_months,
    SUM((formation_rows > 1)::INTEGER)                               AS commingled_well_months,
    SUM(CASE WHEN formation_rows > 1 THEN formation_rows ELSE 0 END) AS records_in_commingled,
    SUM((oil_bbl = 0 AND gas_mcf = 0)::INTEGER)                      AS zero_production,
    SUM((coalesce(days, 0) = 0)::INTEGER)                            AS zero_or_missing_days,
    SUM((days IS NULL)::INTEGER)                                     AS missing_days,
    SUM((oil_bbl > 0 AND coalesce(days, 0) = 0)::INTEGER)            AS oil_without_days,
    SUM((days > day(last_day(make_date(year, month, 1))))::INTEGER)  AS days_beyond_month,
    SUM((oil_bbl < 0 OR gas_mcf < 0 OR water_bbl < 0)::INTEGER)      AS negative_volumes,
    SUM(revised::INTEGER)                                            AS revised_well_months
FROM well_months
