-- Quality of the bundled cohort extract, by split: wells, well-months, and the defects the model
-- and the scores have to live with (shut-in months, months under the 3-day floor, revisions,
-- commingled formations, and the first-filed record differing from the final one).
WITH m AS (
    SELECT months.*, wells.vintage,
           CASE WHEN wells.vintage >= {test_first} THEN 'test' ELSE 'development' END AS split
    FROM months JOIN wells USING (api)
    WHERE months.ym <= {record_end}
)
SELECT
    split,
    count(DISTINCT api)                                         AS wells,
    count(*)                                                    AS well_months,
    sum(CAST(oil_bbl <= 0 OR oil_bbl IS NULL AS INTEGER))       AS months_without_oil,
    sum(CAST(coalesce(days, 0) < {min_days} AS INTEGER))        AS months_under_min_days,
    sum(CAST(revised AS INTEGER))                               AS months_revised,
    sum(CAST(versions > 1 AS INTEGER))                          AS months_filed_more_than_once,
    sum(CAST(formation_rows > 1 AS INTEGER))                    AS months_commingled,
    sum(CAST(oil_bbl_first IS DISTINCT FROM oil_bbl AS INTEGER)) AS months_first_oil_differs,
    sum(CAST(days_first IS DISTINCT FROM days AS INTEGER))      AS months_first_days_differs,
    sum(oil_bbl)                                                AS oil_bbl_final,
    sum(oil_bbl_first)                                          AS oil_bbl_first_filed
FROM m
GROUP BY split
ORDER BY split
