-- The bundled well table: one row per cohort well, with what the model and the splits need.
SELECT api, first_ym, vintage, last_ym, producing_months, primary_formation,
       first_operator, last_operator, spud_date, lateral_ft, max_md, max_tvd,
       latitude, longitude, township, range, section, loc_id, field_name
FROM cohort
WHERE in_cohort
ORDER BY api
