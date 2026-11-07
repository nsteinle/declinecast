-- One row per (well, production month, formation). The 2025 file re-exports late and amended
-- filings for earlier production months, and most of them also appear in their own year's file,
-- so the same record can occur twice. "final" keeps the most recently accepted version, "first"
-- the earliest; both panels are built so that results can be compared on each.
SELECT * EXCLUDE (pick)
FROM (
    SELECT *,
           COUNT(*) OVER (PARTITION BY api, year, month, formation)          AS versions,
           row_number() OVER (PARTITION BY api, year, month, formation
                              ORDER BY accepted {direction} NULLS LAST,
                                       file_year {direction}, doc_num {direction}) AS pick
    FROM raw
)
WHERE pick = 1
