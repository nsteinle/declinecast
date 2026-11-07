"""Neural posterior estimation: build the flow, train it on simulations, sample it.

The density estimator follows the paper's description: a neural spline flow of twelve masked
autoregressive transformations, each a monotonic rational-quadratic spline with ten bins, the
feature order reversed between transformations, and spline parameters predicted from the
128-dimensional conditioning vector by a perceptron with two hidden layers of 256 units. It is
built with ``sbi``'s zuko builder.

One trap in that builder, found while building this: an integer ``hidden_features`` is expanded
to ``[hidden_features] * num_transforms``, which zuko reads as the widths of the conditioner's
hidden layers, so ``hidden_features=256, num_transforms=12`` gives every transformation a
twelve-hidden-layer conditioner. The paper reports exactly that configuration as systematically
miscalibrated. ``build_estimator`` therefore passes ``[256, 256]``.

The proposal is the prior, so the single-round NPE objective needs no correction.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from declinecast.embedding import DeclineEmbedding
from declinecast.simulator import x_dim

__all__ = ["FlowConfig", "TrainConfig", "build_estimator", "train", "sample", "log_prob"]


@dataclass
class FlowConfig:
    num_transforms: int = 12
    hidden_layers: tuple = (256, 256)
    num_bins: int = 10
    n_params: int = 6

    def to_dict(self):
        d = asdict(self)
        d["hidden_layers"] = list(self.hidden_layers)
        return d


@dataclass
class TrainConfig:
    n_simulations: int = 500_000     # the paper's N = 5 x 10^5
    batch_size: int = 512
    learning_rate: float = 5e-4
    validation_fraction: float = 0.1
    stop_after_epochs: int = 10      # early stopping patience
    max_epochs: int = 45             # a laptop budget: about 2 h on two CPU cores
    clip_max_norm: float = 5.0
    lr_patience: int = 4             # halve the learning rate after this many flat epochs
    seed: int = 0
    noise: bool = True               # False: the paper's noiseless comparison
    use_lateral: bool = True         # False: the lateral-length ablation


def build_estimator(theta: torch.Tensor, x: torch.Tensor, cfg: FlowConfig | None = None):
    """The conditional flow q(theta | x) with the embedding network. ``theta`` and ``x`` are a
    batch used only for shapes and the z-scoring of theta."""
    cfg = cfg or FlowConfig()
    from sbi.neural_nets.net_builders.flow import build_zuko_nsf

    return build_zuko_nsf(
        batch_x=theta, batch_y=x,
        z_score_x="independent", z_score_y="none",
        hidden_features=list(cfg.hidden_layers),
        num_transforms=cfg.num_transforms, num_bins=cfg.num_bins,
        embedding_net=DeclineEmbedding(),
    )


def train(theta: np.ndarray, x: np.ndarray, flow_cfg: FlowConfig | None = None,
          cfg: TrainConfig | None = None, log=print, init_state: dict | None = None,
          checkpoint=None):
    """Train by maximum likelihood on (theta, x) pairs with early stopping on a held-out
    validation fraction. Returns ``(estimator, history)``; ``history`` holds the per-epoch
    training and validation losses and the epoch kept."""
    flow_cfg, cfg = flow_cfg or FlowConfig(), cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    theta_t = torch.as_tensor(theta, dtype=torch.float32)
    x_t = torch.as_tensor(x, dtype=torch.float32)
    if x_t.shape[1] != x_dim():
        raise ValueError(f"x has {x_t.shape[1]} columns, expected {x_dim()}")
    n = theta_t.shape[0]
    perm = torch.randperm(n, generator=gen)
    n_val = int(round(n * cfg.validation_fraction))
    val, tr = perm[:n_val], perm[n_val:]
    est = build_estimator(theta_t[tr[:5000]], x_t[tr[:5000]], flow_cfg)
    if init_state is not None:  # continue training from saved weights
        est.load_state_dict(init_state)
    opt = torch.optim.Adam(est.parameters(), lr=cfg.learning_rate)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5,
                                                       patience=cfg.lr_patience)
    best, best_state, since, hist = float("inf"), None, 0, []
    t0 = time.time()
    for epoch in range(cfg.max_epochs):
        est.train()
        order = tr[torch.randperm(len(tr), generator=gen)]
        total = 0.0
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i:i + cfg.batch_size]
            loss = est.loss(theta_t[idx], condition=x_t[idx]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(est.parameters(), cfg.clip_max_norm)
            opt.step()
            total += loss.item() * len(idx)
        est.eval()
        with torch.no_grad():
            v = sum(est.loss(theta_t[val[i:i + 4096]], condition=x_t[val[i:i + 4096]]).sum().item()
                    for i in range(0, n_val, 4096)) / n_val
        sched.step(v)
        hist.append({"epoch": epoch, "train_loss": total / len(tr), "validation_loss": v,
                     "learning_rate": opt.param_groups[0]["lr"],
                     "seconds": round(time.time() - t0, 1)})
        if v < best - 1e-4:
            best, since = v, 0
            best_state = {k: t.detach().clone() for k, t in est.state_dict().items()}
            if checkpoint is not None:
                checkpoint(best_state, hist)
        else:
            since += 1
        if log:
            log(f"epoch {epoch:3d}  train {total / len(tr):.4f}  validation {v:.4f}"
                f"  ({time.time() - t0:.0f} s)")
        if since >= cfg.stop_after_epochs:
            break
    est.load_state_dict(best_state)
    est.eval()
    kept = int(np.argmin([h["validation_loss"] for h in hist]))
    return est, {"epochs": hist, "kept_epoch": kept, "best_validation_loss": best}


@torch.no_grad()
def sample(est, x: np.ndarray, n: int, lower: np.ndarray, upper: np.ndarray, seed: int = 0,
           batch: int = 32, max_rounds: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """``n`` posterior draws for each row of ``x``, keeping only draws inside the prior's
    support (the paper's leakage filter). Returns ``(samples, acceptance)`` with samples of
    shape (rows, n, 6) and the fraction of raw draws kept per row. If fewer than ``n`` draws
    land inside the support after ``max_rounds`` rounds, the rest are raw draws clipped to it;
    such a row has a very low acceptance and is flagged by ``declinecast.ood``."""
    torch.manual_seed(seed)
    x_t = torch.as_tensor(np.asarray(x, np.float32))
    lo = torch.as_tensor(lower, dtype=torch.float32)
    hi = torch.as_tensor(upper, dtype=torch.float32)
    rows = x_t.shape[0]
    out = np.empty((rows, n, len(lower)), np.float32)
    acc = np.empty(rows)
    for s in range(0, rows, batch):
        xb = x_t[s:s + batch]
        b = xb.shape[0]
        kept = [[] for _ in range(b)]
        count = np.zeros(b, int)
        drawn = np.zeros(b, int)
        for _ in range(max_rounds):
            need = int(max(n - count.min(), 0))
            if need == 0:
                break
            m = int(min(max(need * 1.3, 64), 2 * n, max(8_000 // b, 16)))
            draws = est.sample((m,), condition=xb)          # (m, b, d)
            ok = ((draws >= lo) & (draws <= hi)).all(-1)    # (m, b)
            drawn += m
            for j in range(b):
                if count[j] < n:
                    good = draws[ok[:, j], j]
                    kept[j].append(good)
                    count[j] += good.shape[0]
        for j in range(b):
            allj = torch.cat(kept[j])[:n]
            if allj.shape[0] < n:
                # almost all of the flow's mass is outside the support (a well far from the
                # training wells): top up with raw draws clipped to the support, so the caller
                # still gets n draws, and the acceptance rate (well under the OOD threshold of
                # 0.8) says what happened
                extra = est.sample((n - allj.shape[0],), condition=xb[j:j + 1])[:, 0]
                allj = torch.cat([allj, torch.maximum(torch.minimum(extra, hi), lo)])
            out[s + j] = allj.numpy()
            acc[s + j] = count[j] / drawn[j]
    return out, acc


@torch.no_grad()
def log_prob(est, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Log density of the flow (not renormalized for the support), row by row."""
    t = torch.as_tensor(np.asarray(theta, np.float32))
    c = torch.as_tensor(np.asarray(x, np.float32))
    return est.log_prob(t[None], condition=c).squeeze(0).numpy()
