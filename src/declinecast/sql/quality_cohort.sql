-- The cohort screens as a funnel, each count conditional on the screens above it.
SELECT
    COUNT(*)                                                                AS wells_producing,
    SUM(dj_formation::INTEGER)                                              AS dj_formation,
    SUM((dj_formation AND starts_in_window)::INTEGER)                       AS starts_in_window,
    SUM((dj_formation AND starts_in_window AND enough_months)::INTEGER)     AS enough_months,
    SUM((dj_formation AND starts_in_window AND enough_months
         AND in_well_master)::INTEGER)                                      AS in_well_master,
    SUM((dj_formation AND starts_in_window AND enough_months AND in_well_master
         AND coalesce(impossible_geometry, false))::INTEGER)                AS impossible_geometry,
    SUM((dj_formation AND starts_in_window AND enough_months AND in_well_master
         AND NOT coalesce(modern_spud, false))::INTEGER)                    AS old_or_missing_spud,
    SUM((dj_formation AND starts_in_window AND enough_months AND in_well_master
         AND NOT coalesce(horizontal, false))::INTEGER)                     AS short_or_missing_lateral,
    SUM(in_cohort::INTEGER)                                                 AS cohort,
    SUM((in_cohort AND producing_months >= 24)::INTEGER)                    AS cohort_24_months,
    SUM((in_cohort AND producing_months >= 36)::INTEGER)                    AS cohort_36_months,
    SUM((in_cohort AND producing_months >= 60)::INTEGER)                    AS cohort_60_months,
    SUM((first_ym // 12 = {window_start})::INTEGER)                         AS wells_starting_at_window_edge
FROM cohort
