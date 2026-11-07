-- Coverage of the posterior's central credible intervals on held-out simulated wells, per
-- quantity, history and level, with the 95% Clopper-Pearson interval on each coverage.
SELECT
    coalesce(CAST(history AS VARCHAR), 'training mix (3-60)') AS history,
    quantity, level AS nominal, coverage AS observed, ci_low, ci_high, n AS wells,
    consistent AS nominal_inside_interval,
    coverage - level AS gap
FROM coverage
ORDER BY quantity, history, nominal
