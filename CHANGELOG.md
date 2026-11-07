# Changelog

## 0.1.0 (2026-11-07)

First public release.

- A DuckDB SQL panel build from the raw Colorado ECMC production files (9.4 million rows,
  2013-2025): reads columns by name across the 2025 format change, keeps one version of each
  re-exported or revised record (final and as-first-reported), sums commingled formations,
  and screens the DJ Basin horizontal Niobrara and Codell cohort with counted flags. Tested
  against an independent pandas implementation (`tests/pandas_reference.py`).
- A bundled, checksummed extract of 8,344 wells, split by first-production year into
  development wells (2014-2019) and locked test wells (2020 and later).
- A stochastic well simulator: a ramp to a peak, the modified hyperbolic Arps decline with a 5%
  terminal decline, AR(1) log-rate noise with Student-t innovations, and downtime, with the
  noise scale and memory inferred per well.
- Neural posterior estimation with the embedding network and neural spline flow described in
  Steinle et al. (arXiv:2607.12971), written from the paper's text, trained on 500,000
  simulated wells, and bundled as a pretrained model with no pickles.
- Validation on held-out simulated wells: simulation-based calibration with a Bonferroni KS
  test and PP plots, coverage at fixed histories, accuracy and contraction, a known-answer
  test, and a prior predictive check against development wells.
- Forecasts of cumulative oil to 36 and 60 months from 6 to 24 months of history, with the
  AR(1) noise carried forward and future uptime drawn from development wells.
- Scoring against a history multiplier, a lateral-only type curve and per-well least-squares
  Arps (WIS, coverage, median error), with operator- and pad-clustered bootstrap intervals,
  out-of-distribution flags, and an as-first-reported comparison.
- A per-well MCMC baseline on the same model (`declinecast.mcmc`, an ensemble sampler in NumPy)
  and `scripts/mcmc_baseline.py`, which scores it on the locked test wells against the
  network's saved forecasts and times both.
- `declinecast` command line (`datasets`, `quality`, `fetch`, `panel`, `simulate`, `train`,
  `validate`, `score`, `predict`, `tableau`, `sql`), Tableau Public tables, a generated model
  card, CI on three Python versions, a Docker image and PyPI Trusted Publishing.
