"""Train a model end to end from the bundled panel: set the prior from development wells,
simulate the training set, train the flow, save the model directory.

Only development wells (vintages 2014-2019) inform anything here. The locked test wells
(2020 and later) are not read.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from declinecast.datasets import Panel, well_arrays
from declinecast.inference import FlowConfig, TrainConfig, train
from declinecast.modelio import write_model
from declinecast.priors import Prior, fit_prior
from declinecast.simulator import simulate

__all__ = ["development_prior", "simulate_training_set", "train_model"]


def development_prior(panel: Panel) -> Prior:
    dev = panel.wells[panel.wells.split == "development"]
    return fit_prior(well_arrays(panel, wells=dev))


def simulate_training_set(prior: Prior, cfg: TrainConfig):
    rng = np.random.default_rng(cfg.seed)
    theta, x = simulate(prior, cfg.n_simulations, rng, noise=cfg.noise)
    if not cfg.use_lateral:
        x[:, 1] = 0.0  # the lateral scalar, at the reference 10,000 ft
    return theta, x


def train_model(panel: Panel, out: str | Path, cfg: TrainConfig | None = None,
                flow_cfg: FlowConfig | None = None, overwrite: bool = False, log=print,
                init_state: dict | None = None, checkpoint=None):
    """Returns the model directory's path."""
    cfg, flow_cfg = cfg or TrainConfig(), flow_cfg or FlowConfig()
    import torch

    torch.set_num_threads(max(torch.get_num_threads(), 1))
    t0 = time.time()
    prior = development_prior(panel)
    theta, x = simulate_training_set(prior, cfg)
    if log:
        log(f"simulated {len(theta):,} wells in {time.time() - t0:.0f} s")
    est, history = train(theta, x, flow_cfg, cfg, log=log, init_state=init_state,
                         checkpoint=checkpoint)
    path = write_model(out, est, prior, flow_cfg, asdict(cfg), history,
                       extra={"training_epochs": history["epochs"]}, overwrite=overwrite)
    from declinecast.modelio import write_model_card

    write_model_card(path)
    return path
