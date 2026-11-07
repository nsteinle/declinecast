-- Rows per raw file, and how many of them belong to another year's production.
SELECT file_year, COUNT(*) AS rows, SUM((year <> file_year)::INTEGER) AS other_year_rows,
       MIN(year) AS first_year, MAX(year) AS last_year
FROM raw
GROUP BY file_year
ORDER BY file_year
