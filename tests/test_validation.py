"""The calibration statistics themselves, on inputs with known answers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from declinecast.validation import (
    QUANTITIES,
    coverage_table,
    held_out,
    known_answer,
    ks_band,
    sbc_table,
    validate,
)


def _ranks(r):
    return {"rank": r, "truth": None, "quantiles": {}, "n_draws": 100}


def test_uniform_ranks_pass_and_skewed_ranks_fail():
    rng = np.random.default_rng(0)
    ok = sbc_table(_ranks(rng.random((2000, len(QUANTITIES)))))
    assert ok.passes.all()
    bad = rng.random((2000, len(QUANTITIES)))
    bad[:, 0] = bad[:, 0] ** 1.5           # truth sits low in the posterior: biased high
    t = sbc_table(_ranks(bad))
    assert not t.passes.iloc[0] and t.passes.iloc[1:].all()


def test_ks_band_matches_the_asymptotic_value():
    assert ks_band(1000) == pytest.approx(1.358 / np.sqrt(1000), rel=0.02)
    assert ks_band(1000, n_tests=9) > ks_band(1000)


def test_coverage_of_uniform_ranks_is_nominal():
    rng = np.random.default_rng(1)
    c = coverage_table(_ranks(rng.random((4000, len(QUANTITIES)))), history=12)
    assert np.allclose(c.coverage, c.level, atol=0.03)
    assert c.consistent.mean() > 0.85
    k, n = 3600, 4000
    lo = stats.beta.ppf(0.025, k, n - k + 1)
    row = coverage_table(_ranks(np.full((n, len(QUANTITIES)), 0.5)), None).iloc[0]
    assert row.coverage == 1.0 and lo < 0.9


def test_held_out_fixes_history(panel):
    from declinecast.training import development_prior

    d = held_out(development_prior(panel), 50, seed=3, history=12)
    assert (d.history == 12).all()
    assert (d.x[:, 3 + 60 + 12:3 + 120] == 0).all()   # mask channel past month 12


@pytest.mark.slow
def test_known_answer_with_the_bundled_model(model):
    ka = known_answer(model)
    names = ka["quantity"]
    for q in ("log10_qi", "capacity_cum_36", "capacity_cum_60"):
        i = names.index(q)
        assert ka["inside_90"][i], q
    assert ka["acceptance"] > 0.8


@pytest.mark.slow
def test_bundled_model_is_calibrated_on_a_small_run(model):
    out = validate(model, n_sbc=300, n_per_history=150, n_draws=200, seed=77, log=None)
    cov = out["coverage"]
    derived = cov[cov.quantity.str.startswith("capacity") & (cov.level == 0.9)]
    assert (derived.coverage > 0.82).all()
