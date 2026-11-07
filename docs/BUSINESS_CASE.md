# Business case: forecasting young oil wells with ranges that hold up

This document follows one problem from the business question to a validated forecast, and
explains the reasoning at each step. The figures come from the bundled extract (8,344 DJ Basin
wells) and the bundled model, and are reproduced by `scripts/reproduce.sh`. Ranges in brackets
are 95% intervals from resampling whole operators (section 6), unless they say otherwise.

## 1. The business problem

A horizontal oil well produces most of its oil in its first few years, and its rate falls fast
from the first months. Whoever owns, buys or lends against a young well needs to know how much
oil it will have produced in two or five years, long before those years have passed, and how
sure that number is. The people who ask:

- an **acquisitions (A&D) team** pricing a package of wells that have produced for a year;
- a **mineral and royalty buyer**, whose income is a share of each well's volumes;
- a **reserve-based lender**, who sets a borrowing base from the low case, not the middle;
- an **operator's reservoir or planning team**, deciding whether a new development area is
  performing to its type curve, and which wells to look at first.

The questions they ask:

1. Given this well's first 6, 12, 18 or 24 months, how much oil will it have produced by month
   36 (and month 60)?
2. What is the low case and the high case (P90 and P10), and does an 80% range really contain
   the outcome 80% of the time?
3. Is this better than what we already do: a least-squares Arps fit, a type curve, or a
   multiplier from how similar wells went?
4. Which wells should we not trust the forecast for?
5. Does it matter that the regulator's record is revised after it is first filed?

Question 2 is the one current practice answers least well. A least-squares fit gives one curve,
and the range around it, when there is one, is too narrow (section 5). A range that is too
narrow is worse than no range, because a lender or a buyer will act on it.

## 2. Translating it into a technical problem

| Business element | Technical choice | Reason |
|---|---|---|
| Unit of decision | one well | wells are bought, booked and reviewed one by one, or as sums of ones |
| Outcome | cumulative oil from first production to month 36 (headline) and 60 | volumes, not rates, are what a price or a borrowing base multiplies; 36 months is the headline because the 60-month wells are all 2020 or earlier (section 7) |
| Inputs | monthly oil and producing days for the first H months, and the lateral length | what the public record holds for every well; the lateral explains 22.5% of the variance of peak rate |
| Decline model | modified hyperbolic Arps with a 5% per year terminal decline | the industry's own model, so the parameters mean what an engineer expects |
| Uncertainty | a Bayesian posterior over six parameters, from a network trained on simulated wells (neural posterior estimation) | it gives a full distribution in a fraction of a second per well, and its calibration can be checked on simulations where the truth is known |
| Noise and downtime | inferred per well: lognormal scatter with AR(1) memory and heavy tails, and months without production | real monthly records are noisy and have gaps; treating the noise as known or as independent would make the ranges too narrow |
| What makes a forecast good | the weighted interval score (WIS), the coverage of the 50 to 95% ranges, and the median error, all on the log scale | a proper score rewards honest ranges; coverage is what a lender or buyer relies on; the median error is what an engineer looks at first |
| Benchmarks | a history multiplier, a type curve from the lateral alone, and per-well least-squares Arps | the multiplier is the strongest simple empirical method, the type curve is what is known before production, least squares is current practice |
| How sure we are of the comparison | bootstrap over whole operators and over whole pads | wells of one operator or one pad are not independent; resampling wells would overstate the precision several-fold |
| Out-of-time test | wells that first produced in 2020 and later, locked on 7 November 2026 and scored once | a forecast is used on wells newer than the ones it was built on |

## 3. Data quality first

The public record is a regulator's filing system, not a clean dataset. `declinecast quality
--raw` (on the raw files) and `declinecast quality` (on the bundled extract) count every
defect, and FINDINGS section 1 lists them. The ones that change the answer:

- **The 2025 file re-exports about 240,000 records from earlier years**, and 239,845 records
  appear in two files. Summing every row, as I did in phase 0, double-counts them. The panel
  keeps one version of each record, the latest accepted (and, for the as-reported comparison,
  the earliest). This changed the cohort from 8,438 to 8,344 wells.
- **The 2025 file changed format**: renamed and reordered columns (water would be read as
  flared gas by position), the text `NULL` for missing values, ISO dates and zero-padded months.
  Columns are read by name.
- **Commingled wells** report one row per producing formation (1,117,247 well-months). Rows are
  summed, not de-duplicated.
- **Zero-day months.** 1,714,149 well-months have no producing days. Rate is oil per producing
  day, and a month with fewer than 3 days is treated as unobserved, not as a month at zero rate.
- **The record ends early.** Wells filing a month drop from about 8,200 to about 6,000 from
  July 2025, because filings are late. The record is used through June 2025.
- **Left censoring.** 46,877 wells first appear in January 2013, where the files start, so
  their real first month is unknown. The cohort starts in 2014.

## 4. Solution

- **Data layer** (DuckDB SQL): the panel, the cohort, the quality report and the Tableau tables.
- **Simulator**: each piece measured on development wells (FINDINGS section 2), and checked
  against them with a prior predictive check: at most 1.6% of real wells fall outside the
  simulated 1 to 99% range on any of nine summaries.
- **Network**: trained once on 500,000 simulated wells; afterwards a forecast for a well takes
  about 0.2 s on two CPU cores (500 posterior draws).
- **Validation on simulated wells**: SBC and coverage at 6, 12, 24 and 48 months of history.
- **Forecasts**: observed oil to date plus the posterior predictive future, with the noise
  carried forward from the last month and future uptime drawn from development wells. Every
  method gets the same observed oil and the same uptime, so the comparison is between decline
  models.
- **Recalibration**: on development wells the network's median ran 14% high at 6 months, so
  the predicted oil is scaled, per history, to make the median error on development wells zero
  (section 5). It is applied by default, and the unadjusted network is reported beside it.
- **Benchmarks**, scored on the same wells: the history multiplier, the lateral-only type
  curve and per-well least-squares Arps with delta-method intervals; and, on the test wells,
  per-well MCMC on the same model, to price the network's speed (section 5).
- **The locked test**: 2,929 wells first produced in 2020 or later, 1,463 of them with 36
  months of record, set aside on 7 November 2026 before any scoring, and scored once.

## 5. Outcomes

On the locked test wells, 36-month horizon (ranges in brackets: operator bootstrap, 20 to 22
operators):

| months seen | network: median error | network: 80% range covers | network: 90% range covers | least squares: 80% range covers | multiplier: 80% range covers |
|---|---|---|---|---|---|
| 6 | 13.1% | 87.7% [79.3, 91.2] | 96.1% | 57.2% | 84.3% |
| 12 | 6.8% | 81.0% [76.7, 84.4] | 91.9% | 48.3% | 84.1% |
| 18 | 3.2% | 77.3% [74.0, 81.8] | 88.0% | 48.6% | 83.0% |
| 24 | 1.4% | 78.1% [73.6, 83.4] | 88.9% | 55.0% | 85.2% |

What this answers:

1. **How much oil by month 36, and how sure?** At 12 months of history the median forecast is
   within 6.8% of the outcome for half the wells, and the P90 to P10 range contains the outcome
   for 81% of them. The range is about ±13% wide on average at 12 months (0.111 dex), and
   about ±3% at 24 months.
2. **Does the 80% range mean 80%?** Close to it: 88%, 81%, 77% and 78% at 6, 12, 18 and 24
   months. At 6 months it is conservative; at 18 and 24 months slightly overconfident, by 2 to
   3 points, within the bootstrap interval.
3. **Better than current practice?** Than per-well least squares, clearly: its 80% range
   contains the outcome for only about half the wells, because a single fit's interval ignores
   how wells behave after the fit, and its WIS is worse at every history. Than the history
   multiplier, only at 24 months (WIS −0.0015 [−0.0033, −0.0000]); the two tie at 6, 12 and
   18 months, and the multiplier's median is more accurate at 12 months. Than per-well MCMC on
   the same model (the published Bayesian method), equally accurate at 12 and 18 months and
   within 0.2% at 24, at 2.5 to 9 times less compute per forecast.
4. **When not to trust it?** The out-of-distribution flag marks 174 of the 1,463 test wells. At
   12 months their 80% range covers 68%, against 83% for the rest.
5. **Does the revision of filings matter?** Not here: on the first-filed record every score
   moves by under one point.

### What the network adds over the multiplier

If the multiplier ties the network on most histories, why use the network? Three reasons, and
one honest limit:

- **A well-level answer, not a basin average.** The multiplier gives every well the same
  ratio; the network's posterior is specific to the well's own decline, peak and noise. Its
  range is narrower for regular wells and wider for erratic ones, which is what lets it beat
  the multiplier at 24 months.
- **Parameters an engineer can read.** Posterior Arps parameters, a noise scale and its memory,
  and derived quantities to any horizon (EUR to 30 years included), all from one posterior.
- **Cheaper than fitting each well.** Per-well MCMC on the same model, the published Bayesian
  approach, is as accurate (a tie at 12 and 18 months, 0.2% better at 24) but costs 0.40 to 1.41
  s per forecast against the network's 0.16 s. A team re-forecasting thousands of wells every
  month gets the same answer for 2.5 to 9 times less compute once the network is trained
  (FINDINGS section 12).
- **Any history.** It forecasts from 3 to 60 months with no table of empirical ratios to
  rebuild for each pair of history and horizon.
- **The limit.** At 12 months, for the median alone, the multiplier is the better and simpler
  answer, and a team that only needs a P50 for a portfolio sum should use it.

## 6. Uncertainty in the results

Wells are clustered: one pad shares a schedule and a design, one operator shares practices. The
test wells come from 20 to 22 operators, so every comparison is resampled by whole operators
(and, as a check, by whole pads). A well-level bootstrap would make the intervals several times
too narrow (FINDINGS section 1). Two consequences:

- Differences of a few thousandths in WIS between the network and the multiplier are inside
  their intervals at 6, 12 and 18 months; they are ties, and the docs call them ties.
- The coverage of the network's ranges is known to within about ±4 points on the test wells.
  A coverage of 77% to 78% at 18 to 24 months is consistent with 80%, but also with 74%.

## 7. Monitoring

A forecast made today is checked by production filed months from now:

1. Keep each forecast run's output (`declinecast predict` writes the quantiles and the oil to
   date).
2. When 12 more months have been filed, rebuild the extract (`declinecast fetch` then `panel`)
   and score the wells forecast a year earlier (`declinecast score`).
3. Compare the realized 80% coverage with this document. Below 74% (the lower end of the
   bootstrap interval at 18 to 24 months) means the decline behaviour has moved: retrain the
   prior and the recalibration on the newer wells.
4. Watch the share of wells flagged out of distribution. It rises when laterals, completions or
   operators move outside the training range.

## 8. Governance

- **Pre-registration.** The expected results were written down on 17 October 2026, before any
  model existed. FINDINGS section 8 compares each with the outcome, including the two that were
  falsified.
- **One look at the test.** The test wells were scored once, after every development decision
  (including the recalibration) had been made, and the command refuses to score them without
  `--unlock-test`.
- **A model card** (docs/MODEL_CARD.md) states the intended use, the limits and the measured
  performance.
- **Provenance.** Every source file's SHA-256 is recorded; the extract's and the model's
  checksums are checked on every load; the network is rebuilt from its manifest, not unpickled.

## 9. Glossary

- **Cumulative oil (cum36, cum60):** barrels produced from the first month to month 36 or 60.
- **EUR:** estimated ultimate recovery; here, the noise-free cumulative to 30 years.
- **P90, P50, P10:** reserves convention. P90 is the low case (90% chance of exceeding it), P10
  the high case. The P90 to P10 range is the central 80% interval.
- **Arps decline:** q(t) = qi (1 + b Di t)^(-1/b), with initial rate qi, initial decline Di and
  hyperbolic exponent b; "modified" switches to an exponential at a terminal decline.
- **Neural posterior estimation:** a network trained on simulated (parameters, data) pairs that
  returns the posterior distribution of the parameters for new data.
- **SBC:** simulation-based calibration; the rank of the true parameter among posterior draws
  is uniform if the posterior is calibrated.
- **WIS:** weighted interval score, a proper score for a set of central intervals and the
  median; lower is better.
- **Coverage:** the share of outcomes inside a stated interval.
- **dex:** a factor of 10 in log10 units; 0.01 dex is about 2.3%.
