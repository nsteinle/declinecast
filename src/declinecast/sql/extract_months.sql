-- The bundled monthly table: every reported month of every cohort well from its first
-- producing month on, from the final panel, with the as-first-reported oil and days beside it.
SELECT m.api, m.ym, m.oil_bbl, m.gas_mcf, m.water_bbl, m.days, m.status, m.revised,
       m.versions, m.formation_rows, f.oil_bbl AS oil_bbl_first, f.days AS days_first
FROM well_months m
JOIN cohort c ON c.api = m.api AND c.in_cohort
LEFT JOIN well_months_first f ON f.api = m.api AND f.ym = m.ym
WHERE m.ym >= c.first_ym
ORDER BY m.api, m.ym
