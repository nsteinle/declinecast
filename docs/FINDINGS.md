# Findings

Everything here was measured on the bundled extract and the bundled model, and is reproduced by
`scripts/reproduce.sh`. Development wells first produced 2014-2019; locked test wells 2020 and
later. Unless a line says otherwise, intervals in brackets are 95% bootstrap intervals over
whole operators (29 among the scored development wells).

## 1. The data

**Source.** ECMC per-well, per-month, per-formation production files for 2013 to 2025, and the
well master (DATA_LICENSE.md). 9,379,877 rows.

**Defects found and handled** (`declinecast quality --raw`, in `panel.json`):

| defect | count | handling |
|---|---|---|
| records filed in two files (the 2025 file re-exports earlier months) | 239,845 | keep one version per (well, month, formation): the latest accepted (final) and the earliest (as-reported) |
| re-exported copies whose oil differs from the original | 1,981 (304 differ in days) | as above; both panels are kept |
| 2025 format change: renamed and moved columns, the text `NULL`, ISO dates, zero-padded months | whole 2025 file | read by column name; both null tokens and both date formats |
| commingled well-months (more than one formation row) | 1,117,247 | rows summed; producing days = longest row |
| well-months with zero production | 1,746,662 | kept; not a rate observation |
| well-months with zero or missing days | 1,714,149 | rate is oil per producing day; under 3 days = unobserved |
| oil with no producing days | 274 | unobserved |
| negative volumes | 400 | kept as filed; none in a cohort well's rate |
| revised well-months | 283,663 | flagged |
| status codes not upper case | 2,053 | upper-cased |
| wells starting in January 2013 (the left edge of the record) | 46,877 | excluded: their first month is unknown |
| incomplete recent filings | wells reporting fall from about 8,200 to about 6,000 from July 2025 | record used through June 2025 |

**Correction to phase 0.** Phase 0 (17 October 2026) summed every row per well-month, so the
239,845 duplicated records were counted twice. The cohort is 8,344 wells, not 8,438, and the
phase-0 counts in the project plan are superseded by these.

**Cohort funnel.** 61,375 producing wells → 30,373 in the Niobrara and Codell → 9,341 first
producing in 2014 or later → 8,600 with at least 12 producing months, all in the well master →
minus 20 with impossible geometry, 107 with an old or missing spud date and 236 with a lateral
under 3,000 ft → **8,344**: 5,415 development wells and 2,929 test wells. Of these, 7,670 have
at least 24 producing months, 6,945 at least 36 and 5,582 at least 60.

**The SQL layer.** The panel build is DuckDB SQL (`src/declinecast/sql/`): 9.4 million rows to
the extract in about 45 s on two cores. `tests/test_sql.py` checks every table against an
independent pandas implementation on a miniature copy of the raw files with each defect above.
The same comparison on all 9.4 million rows did not fit in the build machine's memory (8 GB)
while a model was training; it runs with `DECLINECAST_RAW` set and is in the runbook.

**Revisions are rare in this cohort.** In the extract, the first-filed oil differs from the
final oil in 861 of 554,092 development well-months (0.16%) and 548 of 105,499 test well-months
(0.52%). Total oil differs by under 0.01%.

## 2. What the development wells look like

Measured on development wells only (they also set the prior):

- **The peak is not month 0.** The share of wells peaking in months 0 to 5 is 0.135, 0.332,
  0.283, 0.129, 0.076 and 0.045. Month 0 is usually partial (median 17 producing days).
- **Downtime.** Per-well uptime over months 12 to 35 has median 0.982, 10th percentile 0.786
  and 5th percentile 0.679. Early and later uptime correlate weakly (0.23).
- **Decline.** Fits from the peak on 595 wells: qi median 406 bbl/d, nominal Di median 3.24 per
  year (2nd to 98th percentile 0.6 to 17.5), b median 0.77 (0.01 to 1.57), with 24% above 1, so
  the terminal decline is needed for a finite EUR.
- **Noise.** Residual sd median 0.25 (5th to 95th percentile 0.09 to 0.76), lag-one
  autocorrelation median 0.47 (10th to 90th 0.15 to 0.74), kurtosis 46, and the residual sd
  rises from 0.29 to 0.47 at months 35 to 59. Independent Gaussian noise would be wrong on three
  counts, which is why the simulator has AR(1) noise with Student-t innovations, and why sigma
  and rho are inferred per well.
- **Lateral.** log10 peak rate = 2.681 + 0.648 log10(lateral / 10,000 ft), which explains 22.5%
  of its variance (residual sd 0.179 dex). The prior uses this with the sd widened 1.7 times.

**What a short history can tell** (Cramér-Rao bounds, independent noise at sigma 0.25): the sd
of b is 5.6 at 6 months, 1.0 at 12, 0.3 at 24 and 0.17 at 36, while qi is known to 0.06 to 0.08
dex at every history. So b is unknowable at 6 months and the cumulatives, which combine the
parameters, are what a short history determines. That is why the derived quantities are
calibrated and reported, as the chirp mass was in the paper.

**Prior predictive check** (`declinecast simulate`): for nine summaries of the first 36 months
(peak month, peak rate, drops to months 12 and 33, early and late slopes, roughness, lag-one
correlation, unobserved share), at most 1.6% of real development wells fall outside the
simulated 1 to 99% range. The simulator can produce what is observed. It produces a wider
spread than observed on most summaries, which is what a prior should do.

## 3. The model

- Network: the embedding and flow of the paper (12 rational-quadratic spline transformations,
  10 bins, a 2 x 256 conditioner, about 1.8 million parameters), trained on 500,000 simulated
  wells, history drawn uniformly from 3 to 60 months.
- Training: 45 epochs (the budget; about 1 h 50 min on two CPU cores). The kept epoch is 39,
  with validation loss −3.370 (from +0.137 after epoch 0). Training loss kept falling
  (−3.656 at epoch 44) while the validation loss levelled off from about epoch 30 and wandered
  between −3.1 and −3.37, so more epochs would mostly have widened the gap.
- Posterior draws inside the prior's support: median 98.9% on held-out simulated wells, but
  below 29% for the worst 1% of them.

## 4. Calibration on simulated wells

1,000 held-out wells with histories drawn as in training, then 500 at each of 6, 12, 24 and 48
months, 500 posterior draws each (`declinecast validate`).

**Coverage.** 173 of 180 cells (9 quantities x 4 levels x 5 history settings) contain their
nominal level inside the 95% Clopper-Pearson interval. The largest gap is 5.6 points. For the
36-month capacity cumulative, the 90% interval covers 89.3% (training mix), 88.2% (6 months),
90.8% (12), 91.6% (24) and 88.6% (48).

**SBC (KS test, 5% family-wise, Bonferroni over 9 quantities).**

| quantity | KS statistic | Bonferroni p | passes | mean rank |
|---|---|---|---|---|
| log10 qi | 0.030 | 1.0 | yes | 0.489 |
| log10 Di | 0.028 | 1.0 | yes | 0.506 |
| b | 0.017 | 1.0 | yes | 0.504 |
| t_peak | 0.049 | 0.16 | yes | 0.476 |
| log10 sigma | 0.050 | 0.11 | yes | 0.526 |
| rho | 0.072 | 0.0005 | **no** | 0.539 |
| capacity cumulative, 36 months | 0.068 | 0.002 | **no** | 0.465 |
| capacity cumulative, 60 months | 0.067 | 0.002 | **no** | 0.459 |
| EUR, 30 years | 0.064 | 0.005 | **no** | 0.464 |

A mean rank below 0.5 means the truth sits low in the posterior: the cumulatives are
overestimated. The bias of the posterior median is +0.009 dex (+2%) for the 36-month cumulative
over the training mix, +0.021 dex (+5%) at 6 months of history, +0.005 at 12, +0.006 at 24 and
+0.002 at 48. With 1,000 wells, SBC detects a shift this small even though every coverage cell
for the cumulatives is consistent with nominal.

**Against the pre-registration** (17 October 2026): "all parameters pass SBC with Bonferroni;
derived 36-mo cum passes too" is **falsified** on two counts, rho and the derived cumulatives.
The failures are small and located, but they are failures, and section 5 shows the bias is
larger on real wells.

**Accuracy and contraction.** Spearman correlation of the posterior median with the truth for
the 36-month cumulative: 0.895 at 6 months, 0.964 at 12, 0.989 at 24, 0.991 at 48. The 90%
interval's width relative to the prior's:

| quantity | 6 months | 12 | 24 | 48 |
|---|---|---|---|---|
| b | 0.98 | 0.95 | 0.79 | 0.45 |
| rho | 0.97 | 0.93 | 0.80 | 0.59 |
| log10 Di | 0.67 | 0.38 | 0.30 | 0.25 |
| capacity cumulative, 36 months | 0.36 | 0.16 | 0.11 | 0.09 |
| EUR, 30 years | 0.52 | 0.30 | 0.19 | 0.11 |

The pre-registration said b at 6 months would stay close to its prior. It does (0.98).

**Known answer.** One well with qi = 10^2.6 bbl/d, Di = 10^0.4 per year, b = 0.8, peak at 1.5
months, sigma = 0.15, rho = 0.4 and 48 months of history: the 90% interval contains the truth
for 8 of 9 quantities, all but rho (truth 0.40, interval 0.01 to 0.37). The 36-month cumulative
is 10^5.104 against a truth of 10^5.099 (+1.2%).

## 5. Development wells

2,000 development wells drawn at random (seed 0), all with at least 60 months of record; 500
posterior draws per well; the empirical baselines fit by five-fold cross-validation blocked by
township (`declinecast score --split development --max-wells 2000`).

**Before recalibration, 36-month horizon:**

| history | method | WIS | median error | median bias | 80% coverage | 90% coverage |
|---|---|---|---|---|---|---|
| 6 | network | 0.049 | 17.0% | +0.057 dex | 85.1% | 93.8% |
| 6 | multiplier | 0.037 | 12.8% | +0.002 | 82.6% | 91.6% |
| 6 | least-squares Arps | 0.051 | 19.2% | +0.002 | 63.4% | 74.1% |
| 6 | type curve | 0.073 | 29.0% | +0.001 | 79.5% | 89.4% |
| 12 | network | 0.022 | 6.6% | +0.009 | 83.4% | 91.6% |
| 12 | multiplier | 0.022 | 7.2% | −0.001 | 79.9% | 89.8% |
| 12 | least-squares Arps | 0.025 | 8.3% | −0.015 | 58.7% | 68.2% |
| 18 | network | 0.012 | 3.0% | +0.003 | 81.2% | 90.8% |
| 18 | multiplier | 0.013 | 4.4% | −0.001 | 79.8% | 89.5% |
| 18 | least-squares Arps | 0.013 | 3.7% | −0.005 | 59.0% | 68.9% |
| 24 | network | 0.006 | 1.5% | +0.001 | 80.8% | 89.9% |
| 24 | multiplier | 0.008 | 2.7% | −0.001 | 79.5% | 88.7% |
| 24 | least-squares Arps | 0.007 | 1.6% | −0.002 | 59.9% | 73.0% |

(Wells where every method has a forecast: 1,912 at 6 months, 1,997 to 2,000 later. The type
curve does not use the history, so its rows at 12 to 24 months repeat the 6-month ones and are
omitted.)

- **Coverage holds up on real wells.** The network's 80% interval covers 81 to 85% and its 90%
  interval 90 to 94%, slightly conservative at short histories. Least-squares Arps covers 59 to
  63% at 80%, because a single fit's interval ignores how wells behave after the fit.
- **Against least-squares Arps** (pre-registered: a tie on median accuracy, a clear win on
  coverage): mean absolute log error differs by +0.0075 [−0.0068, +0.0266] dex at 6 months and
  −0.0024 [−0.0053, +0.0011] at 12, both intervals containing zero: a tie, as expected. 80%
  coverage is higher by 21.7 [17.5, 24.6] points at 6 months and 24.7 [18.2, 28.8] at 12: the
  win, as expected. WIS is lower at 12, 18 and 24 months, with intervals excluding zero
  (−0.0030 [−0.0045, −0.0007] at 12).
- **The history multiplier is the strongest baseline.** It beats the network at 6 months (WIS
  0.037 against 0.049, better on 64% of wells), ties at 12 (0.022 each), and loses at 18 and 24,
  where the network is better on 64% and 73% of wells.
- **Bias.** The network's median forecast is high by +0.057 dex (+14%) at 6 months and +0.009
  (+2%) at 12, and unbiased from 18 months. The simulated-well bias at 6 months is +5%, so most
  of the real-well bias at 6 months is misspecification: the simulator's early declines are
  steeper or later than real ones.

**Recalibration.** Because the bias is material at 6 months, I added a location-only
recalibration before scoring the test: for each history and horizon, the predicted oil (not the
oil already produced) is multiplied by the factor that makes the median log error on these
development wells zero (`scripts/recalibrate.py`, stored in the model's manifest as
`forecast_adjustment`). The factors are 0.784 (6 months, 36-month horizon), 0.944 (12), 0.971
(18) and 0.989 (24); for the 60-month horizon 0.742, 0.925, 0.964 and 0.982. Between scored
histories the factor is interpolated, and it is held at its end values outside 6 to 24 months.

On the same development wells (in-sample, so this shows what the correction does, not that it
works): at 6 months WIS falls from 0.051 to 0.043, the median error from 18.3% to 14.6%, and
80% coverage moves from 84.8% to 86.3%; at 12 months WIS goes from 0.0223 to 0.0214 and 80%
coverage from 83.4% to 82.0%. At 18 and 24 months nothing changes materially. The multiplier
still has the lower WIS at 6 months (0.037). The locked test (section 8) is the check, and it
reports the network both with and without the correction.

**Out-of-distribution flags.** 132 to 155 of the 2,000 wells are flagged (48 for the lateral, 84
for a rate outside the simulated band, 5 to 24 for leakage). Flagged wells have higher WIS (0.034
against 0.021 at 12 months) and lower 80% coverage (80% against 84%), so the flag picks out the
harder wells without them being far off.

## 6. Ablations and sensitivity

Each ablation changes one thing and is scored the same way: 1,000 development wells at 12
months of history, 36-month horizon, 300 draws, without the recalibration; and coverage of the
90% interval of the 36-month capacity cumulative on 400 simulated wells at 12 months
(`scripts/extras.py`).

| model | training | best validation loss | real wells: WIS | median error | median bias | 80% coverage | simulated: 90% coverage |
|---|---|---|---|---|---|---|---|
| as shipped | 500,000 simulations, noise on | −3.370 | 0.0220 | 6.4% | +0.009 | 81.7% | 91.3% |
| 100,000 simulations | same, 45 epochs | −1.869 | 0.0239 | 7.8% | +0.019 | 83.9% | 88.0% |
| no lateral | 100,000, lateral hidden | −1.005 | 0.0257 | 8.3% | +0.031 | 87.1% | 89.8% |
| noiseless | 100,000, no noise or downtime | (not comparable) | 0.0257 | 6.9% | +0.016 | 82.8% | 59.5% |

- **Training-set size matters.** On held-out simulations the 500,000-simulation network's loss
  is 1.5 nats per well lower than the 100,000-simulation one's with the same architecture and
  epoch budget, and on real wells its WIS is 8% lower and its median error 1.4 points lower. So
  the paper's N = 5 x 10^5 is justified here, and I did not test whether more would help.
- **The lateral earns its place.** Hiding it raises WIS by 17% and the median error from 6.4%
  to 8.3%, and the ranges become too wide (87% at 80%). (This ablation also has a fifth of the
  simulations; against the 100,000-simulation network, the lateral is worth 8% of WIS.)
- **The noise model is what makes the posterior honest.** The noiseless network, the paper's
  comparison, is badly overconfident on noisy simulated wells (59.5% at 90%) and 17% worse on
  real wells by WIS. Its real-well coverage looks fine only because its posterior for the noise
  scale is uninformed, so the forecast adds noise from the whole prior range. For some real
  wells almost all of its posterior draws fall outside the prior's support, which stopped the
  first run of this check and is why the sampler now falls back to clipped draws and a low
  acceptance rate instead of an error.

**Terminal decline.** On the posterior draws of 1,000 development wells at 24 months, no draw
reaches its terminal-decline switch before month 60, so the 36- and 60-month forecasts do not
depend on the 5% choice at all. The 30-year EUR does: with a 10% terminal decline it is 3.7%
lower at the median well (5th percentile 13% lower). The EUR is not a headline figure for this
reason, and the docs do not quote one.

**The noise pile-up rule** (from the plan: a noise posterior piling up at its upper bound means
the model is wrong). On real development wells, the share of wells whose posterior median for
sigma is in the top 5% of its range is 0.4% at 6 months, 0.2% at 12 and 0 at 24. The rule is
not triggered. The posterior median of sigma is about 0.14 in natural-log units (10^−0.86),
lower than the least-squares residual sd in section 2 (0.25), probably because those residuals
also hold the model error of a fit through the early months; I have not checked this.

**b on real wells.** The posterior's 90% width relative to the prior's is 0.97 at 6 months,
0.81 at 12 and 0.54 at 24, and the median of the posterior medians is 1.06, 1.00 and 0.92: at 6
months b is the prior's centre, as on simulated wells.

## 7. Why the 60-month numbers carry a caveat

All locked test wells with 60 months of record first produced from January to June 2020 (375
wells), and those were the wells curtailed or shut in during the 2020 price collapse in their
early months. The 36-month test set reaches mid-2022 (1,463 wells). So the 36-month horizon is
the headline, and 60-month test numbers are quoted only with this vintage note.

## 8. The locked test

2,929 wells first produced in 2020 or later, set aside on 7 November 2026 when the split was
chosen, before any model existed; 1,463 have 36 months of record and are scored at that
horizon, 375 at 60 months. Scored once (`declinecast score --split test --unlock-test`), after
the recalibration had been fit on development wells. 500 posterior draws per well. Brackets:
operator bootstrap (20 to 22 operators).

**36-month horizon** (the network as shipped, with the recalibration; "unadjusted" is the same
network without it):

| history | method | wells | WIS | median error | median bias | 80% coverage | 90% coverage |
|---|---|---|---|---|---|---|---|
| 6 | network | 1,283 | 0.043 | 13.1% | +0.024 dex | 87.7% | 96.1% |
| 6 | network, unadjusted | 1,283 | 0.058 | 25.2% | +0.088 | 78.7% | 91.3% |
| 6 | multiplier | 1,283 | 0.040 | 12.1% | −0.020 | 84.3% | 92.8% |
| 6 | least-squares Arps | 1,283 | 0.063 | 18.6% | +0.020 | 57.2% | 68.9% |
| 6 | type curve | 1,283 | 0.064 | 30.2% | −0.025 | 82.5% | 93.1% |
| 12 | network | 1,461 | 0.024 | 6.8% | −0.007 | 81.0% | 91.9% |
| 12 | network, unadjusted | 1,461 | 0.025 | 6.5% | +0.001 | 82.0% | 91.7% |
| 12 | multiplier | 1,461 | 0.024 | 6.0% | −0.006 | 84.1% | 92.8% |
| 12 | least-squares Arps | 1,461 | 0.033 | 9.3% | −0.026 | 48.3% | 58.2% |
| 18 | network | 1,462 | 0.013 | 3.2% | −0.005 | 77.3% | 88.0% |
| 18 | multiplier | 1,462 | 0.013 | 3.7% | −0.003 | 83.0% | 91.9% |
| 18 | least-squares Arps | 1,462 | 0.014 | 3.7% | −0.011 | 48.6% | 59.2% |
| 24 | network | 1,462 | 0.006 | 1.4% | −0.001 | 78.1% | 88.9% |
| 24 | multiplier | 1,462 | 0.007 | 2.2% | −0.001 | 85.2% | 93.4% |
| 24 | least-squares Arps | 1,462 | 0.007 | 1.5% | −0.003 | 55.0% | 65.3% |

(At 6 months, 180 wells have no forecast from at least one method, mostly because least
squares needs two months after the peak; every method is scored on the same wells. The type
curve does not use the history; at 12 to 24 months its WIS is 0.066 to 0.067 and its 80%
coverage 81.2%, on the slightly larger set of wells, and those rows are omitted.)

**Differences, operator bootstrap:**

| history | network minus least squares, WIS | network minus least squares, mean abs. log error | network minus multiplier, WIS | network minus multiplier, mean abs. log error |
|---|---|---|---|---|
| 6 | −0.0199 [−0.0409, −0.0117] | −0.0199 [−0.0271, −0.0068] | +0.0034 [−0.0025, +0.0104] | +0.0118 [−0.0005, +0.0355] |
| 12 | −0.0086 [−0.0196, −0.0036] | −0.0038 [−0.0072, −0.0000] | −0.0005 [−0.0082, +0.0042] | +0.0084 [+0.0037, +0.0165] |
| 18 | −0.0016 [−0.0025, −0.0008] | −0.0000 [−0.0017, +0.0032] | −0.0001 [−0.0011, +0.0015] | +0.0008 [−0.0016, +0.0056] |
| 24 | −0.0013 [−0.0028, −0.0007] | −0.0011 [−0.0034, −0.0000] | −0.0015 [−0.0033, −0.0000] | −0.0025 [−0.0043, −0.0001] |

**The recalibration carried over.** Without it, the test wells' bias at 6 months is +0.088 dex
(+22%), larger than the development wells' +0.057, and 80% coverage is 78.7%. With the factor
fit on development wells it is +0.024 dex (+6%), the median error halves (25.2% to 13.1%) and
WIS falls from 0.058 to 0.043. From 12 months on the correction changes WIS by under 0.001 and
lowers coverage by 1 to 3 points.

**Out-of-distribution flags.** 174 to 180 of the 1,463 test wells (94 for the lateral, 87 to
89 for a rate outside the simulated band). At 12 months the flagged wells' 80% coverage is
67.6% against 82.8%, and their WIS 0.038 against 0.022. The test wells' laterals are longer
than most development wells', which is why more of them are flagged for the lateral (94 against
48 in 2,000 development wells).

**As first reported.** With every method given the record as first filed (300 draws), the
network's WIS changes by at most 0.0001 and its 80% coverage by at most 0.8 points (87.4%,
80.3%, 76.8% and 78.5% at 6, 12, 18 and 24 months). The first-filed oil differs from the final
figure in 0.5% of test well-months.

**60-month horizon** (375 wells, all first producing January to June 2020, section 7): the
network's 80% coverage is 88.3%, 77.8%, 69.8% and 73.3% at 6, 12, 18 and 24 months, against
79.4%, 81.0%, 79.4% and 79.4% for the multiplier and 36% to 62% for least squares. WIS: network
0.057, 0.040, 0.029, 0.014; multiplier 0.048, 0.042, 0.025, 0.018. The network's ranges are
overconfident for these wells from 18 months on; with one half-year vintage, curtailed in 2020,
I do not quote a 60-month figure without this caveat.

**Against the pre-registration** (17 October 2026; "expected" is quoted as written):

| claim | expected | outcome |
|---|---|---|
| median accuracy vs per-well least squares, 36-month cumulative | roughly a tie; operator-clustered 95% CI on the margin contains zero | **falsified, in the network's favour, at 6 months** (−0.020 [−0.027, −0.007] dex); borderline at 12 (−0.004 [−0.007, −0.000]); a tie at 18 and 24 |
| interval width and coverage at 6 and 12 months | clear win; 90% intervals cover near 90% where least squares offers none | **confirmed**: 90% coverage 96.1% and 91.9% against 68.9% and 58.2% |
| calibration on simulations | all parameters pass SBC with Bonferroni; derived 36-month cum passes too | **falsified**: rho and the derived cumulatives fail (section 4) |
| real-data coverage | worse than simulation coverage, because of the residual | **mixed**: better than nominal at 6 months (87.7% at 80%), 2 to 3 points under at 18 to 24 months, within the bootstrap interval; the simulated 90% coverage of the 36-month cumulative is 88% to 92% |
| b at 6 months of history | close to its prior | **confirmed**: posterior width 0.98 of the prior's on simulated wells, 0.97 on real wells |
| serving memory | about the flu package's 1,366 MB → Spaces, not Render | **not tested**: the web app is deferred (November plan) |

## 9. What this shows, and what it does not

It shows:

- **Amortized inference on real regulator data, with posterior uncertainty validated by
  simulation-based calibration and coverage**, and then checked against outcomes on real wells
  it never saw. The ranges hold up: 81% to 88% coverage of the 80% range at 6 and 12 months on
  the locked test, against 48% to 57% for least squares.
- **An honest comparison.** The network clearly beats current per-well practice on ranges and
  on WIS, but only beats the simplest empirical benchmark (the history multiplier) at 24 months,
  and the multiplier's median is more accurate at 12 months. Both are reported.
- **Pre-registered claims checked**, two of them falsified (one each way), and a bias found on
  development wells, corrected there, and confirmed corrected on the test.
- **The same accuracy as per-well MCMC, for less compute** (section 12): on the same model and
  the same test wells, the network ties per-well MCMC at 12 and 18 months and is within 0.2% of
  it at 24, at 2.5 to 9 times less compute per forecast once trained.
- **SQL on a large real dataset**: the panel build from 9.4 million rows with the defects of a
  real filing system, tested against an independent implementation.

It does not show:

- a reserves estimate, an EUR figure for booking, or anything about economics;
- performance outside DJ Basin horizontal Niobrara and Codell oil wells, for gas or water, for
  refracs, or for wells affected by offset fracs (none are modelled);
- a 60-month figure without the vintage caveat (section 7);
- a comparison with commercial forecasting tools, which I have not run;
- deployment or serving: there is a batch command line and a Python API, and no hosted app.

## 10. Verification log

Run on 7 November 2026 on two CPU cores and 8 GB of memory (Python 3.13, PyTorch 2.14.1, sbi
0.27.0, zuko 1.6.0, NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1, DuckDB 2.0.0).

| step | command | time |
|---|---|---|
| panel from the raw files | `declinecast panel` | about 45 s |
| prior predictive check | `declinecast simulate` | 5 s |
| training | `train` (500,000 simulations, 45 epochs) | 1 h 51 min |
| validation | `declinecast validate` (3,000 simulated wells) | 10 min |
| development scoring | `declinecast score --split development --max-wells 2000` | 23 min |
| recalibration | `scripts/recalibrate.py` | seconds |
| operator-blocked baselines | `score ... --block operator --draws 100` | 5 min |
| ablation training | three runs of 100,000 simulations | 9, 9 and 20 min |
| locked test, final record | `declinecast score --split test --unlock-test` | 16 min |
| locked test, as first filed | `... --as-reported --draws 300` | 11 min |
| ablations on real wells, terminal decline | `scripts/extras.py` | about 30 min |
| per-well MCMC on the test wells (5,852 fits) | `scripts/mcmc_baseline.py` | 40 min |

Things that went wrong and were fixed, in order:

1. **Phase-0 double counting** (section 1): fixed by keeping one version of each record.
2. **The 2025 format change**: positional reading would have read flared gas as water;
   fixed by reading columns by name. Tested on a miniature file with the same change.
3. **The sbi builder trap.** `build_zuko_nsf` with an integer `hidden_features` gives each
   transformation one hidden layer per transformation (twelve), the configuration the paper
   reports as miscalibrated. The package passes `[256, 256]`, and `tests/test_modelio.py`
   checks the layer count.
4. **Out-of-memory runs.** Validation and the extras first ran out of memory (the flow's
   sampler and the 360-month EUR sums on hundreds of thousands of draws). Fixed by capping the
   draws per sampler call at 8,000 and computing derived quantities in blocks. The results are
   unchanged by this (it only changes batch sizes).
5. **The sampler's error on far-out wells** (section 6): replaced by a fallback after all
   scoring had finished. No scored well on the development or test runs reached it (those runs
   would have stopped with the error).
6. **The comparison on all 9.4 million raw rows** against pandas ran out of memory while a
   model was training (8 GB machine). The miniature-file tests pass; the full comparison is in
   the local test guide for a laptop with more memory.

The numbers in README.md, BUSINESS_CASE.md and this file were read from the run outputs listed
above. Every figure in those three documents has been checked against them.

## 11. Limitations and next steps

Limitations:

1. **A residual bias on simulated wells** (+2% for the derived cumulatives, +5% at 6 months),
   found by SBC, and a larger one on real wells at 6 months, removed by a correction fit on
   development wells. Per-well MCMC with the same model shows the same real-well bias at 6
   months (section 12), so that part is the decline model's, not the network's. The correction
   is empirical and assumes the bias is stable across vintages; it was on the test, but that
   is one test.
2. **Histories under 6 months** reuse the 6-month correction, untested.
3. **The 18 to 24 month ranges** are 2 to 3 points overconfident on the test wells (within
   the bootstrap interval), and 7 to 10 points at the 60-month horizon.
4. **Training stopped at its 45-epoch budget**, not at convergence; the validation loss had
   levelled off.
5. **Wells are forecast one at a time.** The sum over a package of wells has a range that
   depends on how wells of one operator or pad move together, which this model does not
   describe; a portfolio range needs the cluster structure on top.
6. **The scoring on development wells used a 2,000-well sample** of the 5,415 (compute).

Next steps, in the order I would do them:

1. Remove the simulation bias at its source: an ensemble of networks, or more simulations,
   then re-run SBC (the derived cumulatives should pass).
2. Add a growth of the noise scale with age to the simulator (the residual sd rises from 0.29
   to 0.47 at months 35 to 59), and re-check the 60-month coverage.
3. A package-level forecast that resamples operators and pads, to give a P90 to P10 for a sum
   of wells.
4. A second basin (a Permian loader), which is new data and not a new method.

## 12. Per-well MCMC on the same model

Probabilistic decline-curve analysis by MCMC, fit well by well, is the published Bayesian
approach (Gong et al. 2011 and the Texas A&M work after it; it reports P90-P10 coverage near 80%
in hindcasts). To put the network's speed in context without changing the comparison's other
terms, `declinecast.mcmc` fits the **same** model per well: the same curve with the ramp, the
same prior, the same AR(1) Student-t noise and unobserved months (one approximation: a
transition across a gap is taken as Student-t), with the ramp's starting fraction as a seventh
parameter. The sampler is the affine-invariant ensemble sampler of Goodman and Weare (the
`emcee` algorithm) in NumPy: 32 walkers, 2,000 burn-in and 2,000 kept steps, started from the
best of 2,000 prior draws. `scripts/mcmc_baseline.py` runs it on the 1,463 locked test wells
at 6, 12, 18 and 24 months (5,852 fits) and forecasts with the same code, seeds, uptime and
noise draws as the network. The network's test forecasts are read from the one test scoring,
not recomputed; nothing in the network was changed after it.

**Convergence.** On 64 simulated wells at 12 months, rerunning with chains four times longer
moved the median well's P50 of the 36-month cumulative by 0.0025 dex (0.6%), and its P10 and
P90 by 0.005 dex; the 90th percentile well moved by 0.015 to 0.021 dex, the worst by 0.05. The
median acceptance rate on the test wells was 0.35.

**Accuracy, 36-month horizon, locked test** (brackets: the network minus MCMC, operator
bootstrap over 20 to 22 operators; negative favours the network):

| history | method | WIS | median error | median bias | 80% coverage | 90% coverage |
|---|---|---|---|---|---|---|
| 6 | network (as shipped) | 0.0430 | 13.1% | +0.024 | 87.7% | 96.1% |
| 6 | network, unadjusted | 0.0580 | 25.2% | +0.088 | 78.7% | 91.3% |
| 6 | MCMC | 0.0547 | 25.8% | +0.086 | 83.6% | 93.4% |
| 12 | network (as shipped) | 0.0240 | 6.8% | −0.007 | 81.0% | 91.9% |
| 12 | network, unadjusted | 0.0249 | 6.5% | +0.001 | 82.0% | 91.7% |
| 12 | MCMC | 0.0239 | 6.5% | −0.003 | 80.0% | 91.0% |
| 18 | network (as shipped) | 0.0127 | 3.2% | −0.005 | 77.3% | 88.0% |
| 18 | MCMC | 0.0122 | 2.9% | −0.006 | 75.2% | 86.4% |
| 24 | network (as shipped) | 0.0059 | 1.4% | −0.001 | 78.1% | 88.9% |
| 24 | MCMC | 0.0055 | 1.2% | −0.002 | 75.3% | 85.6% |

- **12 and 18 months: a tie.** WIS differs by +0.0001 [−0.0020, +0.0012] and +0.0005
  [−0.0007, +0.0013]; mean absolute log error by +0.0019 [−0.0007, +0.0046] and +0.0016
  [−0.0007, +0.0033] dex.
- **24 months: MCMC is slightly more accurate**, by 0.0004 [0.0001, 0.0008] in WIS and 0.0010
  [0.0001, 0.0019] dex (0.2%) in mean absolute log error, and the network's ranges are better
  calibrated (78.1% against 75.3% at 80%; difference +2.8 [+0.8, +4.4] points).
- **6 months: the bias is the model's.** MCMC, with the exact likelihood, runs +0.086 dex high,
  as the unadjusted network does (+0.088). So the 6-month bias found on development wells
  (section 5) comes from the decline model and simulator, not from the network's
  approximation. The network as shipped carries the development-fitted recalibration, which the
  MCMC baseline does not, so the 6-month row compares the two with and without it.
- **60 months** (374 wells, the 2020 caveat of section 7): MCMC's 80% coverage is 81%, 77%,
  63% and 66% at 6 to 24 months, against 88%, 78%, 70% and 73% for the network.

**Compute** (same machine, two CPU threads, 500 posterior draws per well):

| | seconds per well | the 5,852 test forecasts |
|---|---|---|
| network posterior | 0.16 | 16 min |
| MCMC, 64 wells vectorized at a time | 0.40 | 39 min |
| MCMC, one well at a time | 1.41 | 2 h 17 min |
| network training (once) | 6,665 s in total | |

Per forecast, the network is 2.5 times cheaper than vectorized MCMC and 8.6 times cheaper than
one-at-a-time MCMC. Training pays for itself after about 28,200 forecasts against the
vectorized MCMC and about 5,300 against the one-at-a-time version: a basin-wide monthly
re-forecast of 10,000 wells passes either break-even within three months, and one-at-a-time in
the first.

What this does and does not say:

- The advantage here is modest because this likelihood is cheap: seven parameters and 60 months.
  Amortization pays more when each likelihood evaluation is expensive (a reservoir simulator,
  the paper's spectral model), and that case is not tested here.
- Neither side was tuned for speed. The network's time is mostly the flow's sampling (fewer
  draws, a GPU or larger batches would cut it); MCMC could run shorter chains, at a convergence
  cost measured above. Both parallelize across cores.
- MCMC targets the posterior of the model exactly, up to its sampling error; the network
  approximates it, with the +2% bias SBC found. On real wells that difference is smaller than
  the model's own error: the two forecasts are as accurate as each other where both are
  unadjusted.
