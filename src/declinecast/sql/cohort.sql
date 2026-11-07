-- The study cohort: horizontal Niobrara and Codell wells in the DJ Basin whose first producing
-- month is inside the record, with at least {min_months} producing months, a modern lateral and
-- a modern spud date. Each screen is a flag, so the quality report can count what it removes.
WITH w AS (
    SELECT
        wells.*,
        m.spud_date, m.max_md, m.max_tvd, m.max_md - m.max_tvd AS lateral_ft,
        m.latitude, m.longitude, m.township, m.range, m.section, m.loc_id, m.field_name,
        m.well_class,
        wells.primary_formation IN ('NBRR', 'NB-CD', 'CODL', 'NIOB', 'CDLL') AS dj_formation,
        wells.first_ym >= {first_year} * 12                              AS starts_in_window,
        wells.producing_months >= {min_months}                           AS enough_months,
        m.api10 IS NOT NULL                                              AS in_well_master,
        m.max_md - m.max_tvd >= {min_lateral_ft}                         AS horizontal,
        m.max_tvd >= m.max_md                                            AS impossible_geometry,
        year(m.spud_date) >= {min_spud_year}                             AS modern_spud
    FROM wells
    LEFT JOIN wellmaster m ON m.api10 = left(wells.api, 10)
)
SELECT
    *,
    first_ym // 12                                                       AS vintage,
    coalesce(dj_formation AND starts_in_window AND enough_months AND in_well_master
             AND horizontal AND NOT impossible_geometry AND modern_spud, false) AS in_cohort
FROM w
