#!/usr/bin/env bash
# Every result in README.md and docs/ from the bundled extract, in order. About 4 hours on two
# CPU cores without the training step (about 2 hours more with it). Stops at the first error.
set -euo pipefail
OUT=${1:-results}

declinecast quality
declinecast simulate --out "$OUT/ppc"                                     # prior predictive
# declinecast train --out "$OUT/model"      # retrain (the bundled model is this command's output)
declinecast validate --out "$OUT/validation"                              # SBC and coverage
declinecast score --split development --out "$OUT/dev" --max-wells 2000   # development wells
# recalibration of the predicted oil, fit on the development wells (FINDINGS section 5); it
# writes into the model directory, so copy the bundled model first if you retrain
python scripts/recalibrate.py --dev "$OUT/dev" --model-dir "$(python -c 'import declinecast, pathlib; print(pathlib.Path(declinecast.__file__).parent / "data" / "model")')"
declinecast score --split development --out "$OUT/dev_operator" --max-wells 2000 \
    --block operator --draws 100 --bootstrap 500                          # operator-blocked
# the locked test, scored once (FINDINGS section 8)
declinecast score --split test --unlock-test --out "$OUT/test"
declinecast score --split test --unlock-test --out "$OUT/test_as_reported" --as-reported \
    --draws 300
# ablations (FINDINGS section 6): 100,000 simulations each, then the extra checks
declinecast train --out "$OUT/ablation_noiseless" --n-simulations 100000 --max-epochs 20 --noiseless
declinecast train --out "$OUT/ablation_no_lateral" --n-simulations 100000 --max-epochs 20 --no-lateral
declinecast train --out "$OUT/ablation_n100k" --n-simulations 100000 --max-epochs 45
python scripts/extras.py --out "$OUT/extras" --noiseless "$OUT/ablation_noiseless" \
    --no-lateral "$OUT/ablation_no_lateral" --n100k "$OUT/ablation_n100k"
# the per-well MCMC baseline on the same test wells (post-processing, FINDINGS section 12)
python scripts/mcmc_baseline.py --test "$OUT/test" --out "$OUT/mcmc"
python scripts/figures.py --results "$OUT" --out docs/figures
declinecast tableau --score "$OUT/dev" "$OUT/test" --validation "$OUT/validation" \
    --out "$OUT/tableau"
