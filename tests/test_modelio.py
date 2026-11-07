"""Saving and loading model directories: no pickles, checksums, refusal of unexpected files."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from declinecast.datasets import well_arrays
from declinecast.inference import FlowConfig, TrainConfig, log_prob, train
from declinecast.modelio import load_model, write_model, write_model_card
from declinecast.priors import Prior, fit_prior
from declinecast.simulator import simulate

TINY = FlowConfig(num_transforms=2, hidden_layers=(16, 16), num_bins=4)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory, panel):
    dev = panel.wells[panel.wells.split == "development"]
    prior = fit_prior(well_arrays(panel, wells=dev.head(400)))
    theta, x = simulate(prior, 3000, np.random.default_rng(0))
    cfg = TrainConfig(n_simulations=3000, max_epochs=2, batch_size=256)
    est, hist = train(theta, x, TINY, cfg, log=None)
    path = tmp_path_factory.mktemp("model")
    write_model(path, est, prior, TINY, cfg.__dict__, hist, overwrite=True)
    write_model_card(path)
    return path, est, theta[:20], x[:20]


def test_round_trip_gives_the_same_density(tiny_model):
    path, est, theta, x = tiny_model
    m = load_model(path)
    np.testing.assert_allclose(log_prob(m.estimator, theta, x), log_prob(est, theta, x),
                               rtol=1e-5, atol=1e-5)
    draws, acc = m.posterior(x[:3], 50)
    assert draws.shape == (3, 50, 6) and np.all(acc > 0)
    assert np.all((draws >= m.lower) & (draws <= m.upper))
    assert "Model card" in (path / "MODEL_CARD.md").read_text()


def test_tampered_weights_are_refused(tiny_model, tmp_path):
    path = tiny_model[0]
    for f in path.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    w = tmp_path / "weights.pt"
    w.write_bytes(w.read_bytes()[:-1] + bytes([w.read_bytes()[-1] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        load_model(tmp_path)


def test_unexpected_files_are_refused(tiny_model, tmp_path):
    path = tiny_model[0]
    for f in path.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    (tmp_path / "extra.pkl").write_bytes(b"not listed")
    with pytest.raises(ValueError, match="does not list"):
        load_model(tmp_path)


def test_weights_file_holds_tensors_only(tiny_model):
    state = torch.load(tiny_model[0] / "weights.pt", weights_only=True)
    assert all(isinstance(v, torch.Tensor) for v in state.values())


def test_manifest_rebuilds_the_prior(tiny_model):
    manifest = json.loads((tiny_model[0] / "manifest.json").read_text())
    prior = Prior.from_dict(manifest["prior"])
    assert len(prior.lateral_quantiles) == 101
    assert manifest["flow"]["hidden_layers"] == [16, 16]


def test_flow_has_the_papers_conditioner(tiny_model):
    """Two hidden layers per transformation, not one per transformation (the sbi trap)."""
    m = load_model(tiny_model[0])
    linear = [mod for mod in m.estimator.net.modules() if isinstance(mod, torch.nn.Linear)]
    assert len(linear) == TINY.num_transforms * (len(TINY.hidden_layers) + 1)


@pytest.mark.slow
def test_bundled_model_loads(model):
    assert model.manifest["flow"] == {"num_transforms": 12, "hidden_layers": [256, 256],
                                      "num_bins": 10, "n_params": 6}
    assert model.manifest["training"]["n_simulations"] == 500_000
