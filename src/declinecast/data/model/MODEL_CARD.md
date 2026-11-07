# Model card: declinecast pretrained network

declinecast 0.1.0, created 2026-11-07T16:02:49+00:00.

## What it does

Given the first 3 to 60 months of a horizontal DJ Basin oil well's monthly production (oil and producing days) and its lateral length, it returns a posterior over six decline parameters, and from them a forecast of the well's cumulative oil with intervals. It is a neural posterior estimate, trained only on simulated wells.

## Intended use and limits

* Screening and portfolio work on horizontal Niobrara and Codell oil wells in the DJ Basin, at 6 to 24 months of history, for cumulative oil to 36 (and, with the vintage caveat in FINDINGS, 60) months.
* Not a reserves estimate: no economic limit, no price deck, no SEC or PRMS process.
* Not validated outside the DJ Basin, for gas or water, for refracs, or for wells affected by offset-well interference; flagged wells (`declinecast.ood`) are outside what the simulator was checked against.

## Model

* Flow: neural spline flow, 12 masked autoregressive transformations, 10-bin rational-quadratic splines, conditioner hidden layers [256, 256] (sbi 0.27.0 with zuko 1.6.0).
* Embedding: the two-branch network of Steinle et al. (arXiv:2607.12971, section VI A), written from the paper's text, with three input channels and pooling over observed months (`declinecast/embedding.py`).
* Parameters: log10_qi, log10_di, b, t_peak, log10_sigma, rho; support lower [1.3, -0.699, 0.01, 0.0, -1.398, 0.0], upper [3.7, 1.602, 2.0, 5.0, 0.079, 0.95].
* Terminal decline 5% per year (effective).
* Prior: log10 qi given lateral L ~ N(2.681 + 0.648 log10(L / 10,000 ft), 0.305), truncated; the rest uniform. Lateral and downtime drawn from development wells (first produced 2014-2019).

## Training

* 500,000 simulated wells, 10% held out for early stopping; batch 512, Adam at 0.0005, learning rate halved after 4 flat epochs, gradient norm clipped at 5.0, at most 45 epochs, seed 0.
* Kept epoch 39 of 45 run; best validation loss -3.370.
* Noise: on; lateral input: on.
* Versions: python 3.13.16, torch 2.14.1+cu130, sbi 0.27.0, zuko 1.6.0, numpy 2.5.4, pandas 3.0.6, duckdb 2.0.0.

## Recalibration

The predicted (future) oil is multiplied by a factor per history and horizon, fit so the median log error on 2,000 development wells is zero (`scripts/recalibrate.py`): 6 months to 36 months: 0.7839, 6 months to 60 months: 0.7417, 12 months to 36 months: 0.9441, 12 months to 60 months: 0.9251, 18 months to 36 months: 0.9706, 18 months to 60 months: 0.9644, 24 months to 36 months: 0.9888, 24 months to 60 months: 0.9823. Applied by default (`forecast(..., adjust=False)` turns it off).

## Calibration on held-out simulated wells

* SBC (1,000 wells, 500 draws each, KS at a 5% family-wise level): passes for log10_qi, log10_di, b, t_peak, log10_sigma; fails for rho, capacity_cum_36, capacity_cum_60, eur_30y.
* Coverage cells consistent with nominal (95% Clopper-Pearson): 173/180; largest gap 0.056.
* Posterior draws inside the support: median 0.989, 1st percentile 0.290.

## Scores on real wells

**Locked test wells (first produced 2020-2022), cumulative oil to 36 months**

| history | horizon | method | wells | WIS | median error | 80% coverage |
|---|---|---|---|---|---|---|
| 6 | 36 | network | 1,283 | 0.0430 | 13.1% | 87.7% |
| 6 | 36 | multiplier | 1,283 | 0.0397 | 12.1% | 84.3% |
| 6 | 36 | arps_ls | 1,283 | 0.0629 | 18.6% | 57.2% |
| 12 | 36 | network | 1,461 | 0.0240 | 6.8% | 81.0% |
| 12 | 36 | multiplier | 1,461 | 0.0245 | 6.0% | 84.1% |
| 12 | 36 | arps_ls | 1,461 | 0.0326 | 9.3% | 48.3% |
| 18 | 36 | network | 1,462 | 0.0127 | 3.2% | 77.3% |
| 18 | 36 | multiplier | 1,462 | 0.0128 | 3.7% | 83.0% |
| 18 | 36 | arps_ls | 1,462 | 0.0143 | 3.7% | 48.6% |
| 24 | 36 | network | 1,462 | 0.0059 | 1.4% | 78.1% |
| 24 | 36 | multiplier | 1,462 | 0.0073 | 2.2% | 85.2% |
| 24 | 36 | arps_ls | 1,462 | 0.0072 | 1.5% | 55.0% |


## Files

| file | SHA-256 |
|---|---|
| weights.pt | `c5caea9c1259d813f0d70d92a1852cb0aabfe7e827aae5684bbf7a0346253870` |
