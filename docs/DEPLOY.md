# Releasing declinecast

Releases go from GitHub Actions to PyPI with **Trusted Publishing**. PyPI trusts a specific
workflow in a specific repository, so there is no API token that could leak. Test on TestPyPI
first, because PyPI never accepts the same version number twice, even after a deletion.

## 0. Before the repository is public

1. Turn on two-factor authentication on GitHub, PyPI and TestPyPI.
2. Check that the name is still free on release day: https://pypi.org/project/declinecast/
   should return "Not Found".
3. Check that every commit is mine: `git log --format='%an %ae'` must print only
   `Nathan Steinle nsastrophysics@gmail.com`.
4. Check that `private/` is not tracked: `git ls-files private` prints nothing.

## 1. GitHub repository

```bash
cd declinecast
git remote add origin git@github.com:nsteinle/declinecast.git   # create the empty repo first
git push -u origin main
```

Check the **Actions** tab. `ci` must pass (tests on Python 3.10, 3.12 and 3.13, the slow checks
on the bundled model, `twine check` and the Docker build with a small validate, score and
tableau run inside the container).

Then in **Settings → Environments**, create `testpypi` and `pypi`. On `pypi`, add yourself as
a required reviewer, so that a release waits for your approval before uploading.

## 2. TestPyPI

1. https://test.pypi.org → Account settings → Publishing → *Add a new pending publisher*:
   project `declinecast`, owner `nsteinle`, repository `declinecast`, workflow `publish.yml`,
   environment `testpypi`.
2. GitHub → Actions → `publish` → *Run workflow* (branch `main`). The `testpypi` job uploads.
3. Verify in a fresh environment (CPU-only PyTorch first, so pip does not pull the 2 GB CUDA
   build):

```bash
python -m venv /tmp/dc && source /tmp/dc/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ "declinecast[plot]==0.1.0"
declinecast datasets
declinecast predict --api 050010984800 --history 12 --out /tmp/dc-one.csv
deactivate
```

`--extra-index-url` lets pip get PyTorch's friends (sbi, zuko, NumPy, DuckDB) from the real
PyPI. To test again, raise the version (for example to `0.1.0rc2`) in
`src/declinecast/_version.py` and `CITATION.cff`.

## 3. PyPI

1. https://pypi.org → Publishing → *Add a new pending publisher* with the same details but
   environment `pypi`.
2. Make sure the version in `src/declinecast/_version.py` is the one you want (`0.1.0`) and is
   committed on `main`.
3. GitHub → Releases → *Draft a new release* → tag `v0.1.0` (it must equal the package
   version, or the workflow stops) → *Publish release*.
4. Approve the `pypi` environment when GitHub asks.
5. Verify: `pip install "declinecast[plot]"` in a fresh environment, then the commands above.

## 4. Every later release

1. Update the version in `src/declinecast/_version.py` and `CITATION.cff` and add a
   `CHANGELOG.md` entry.
2. Push to `main`; wait for `ci`.
3. Optionally, run TestPyPI with a pre-release version (for example `0.1.1rc1`).
4. Publish a GitHub release tagged `v<version>`.

## Rebuilding the bundled extract and model (maintainers only)

```bash
declinecast fetch --dest data/raw            # opt-in: 14 published files, one at a time
declinecast panel --raw data/raw --out data/panel
declinecast train --out results/model        # about 2 h on two CPU cores
declinecast validate --model-dir results/model --out results/validation
```

`panel.json` records the SHA-256 of every source file, and ECMC re-exports and revises
filings, so a later download can give a slightly different extract. A new extract or model is a
new package version, never an edit of the bundled one.

## Using the model on new wells (batch forecasting)

```bash
declinecast predict --csv wells_2026_11.csv --out forecasts_2026_11.csv
```

1. The CSV needs `well_id`, `month` (`YYYY-MM`), `oil_bbl`, `days` and `lateral_ft`. Month 0 is
   each well's first month with oil; a month with no row counts as shut in, and a month with
   fewer than 3 producing days is treated as unobserved.
2. Each well is forecast from all of its months, up to 60, to the horizons asked for (36 and 60
   by default). The output holds P90, P50 and P10 (columns `q0.1`, `q0.5`, `q0.9`) and the
   other quantiles, the oil produced to date, and the share of posterior draws inside the
   prior's support. A share below 0.8 means the well is unlike the training wells.
3. Re-score when 12 more months of production have been filed for the wells forecast
   (`declinecast score` on a rebuilt extract), and compare the realized 80% coverage with
   FINDINGS section 5. Retrain when the coverage falls outside its bootstrap interval, or when
   the wells being forecast move outside the training range (laterals beyond 15,000 ft, other
   formations or basins).

With Docker (the image runs as a non-root user, so pass your user ID to let it write to the
mounted folder):

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" declinecast \
    predict --csv wells.csv --out forecasts.csv
```
