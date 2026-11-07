"""The embedding network, written from the description in Steinle et al., arXiv:2607.12971,
section VI A ("Conditional normalizing flow architecture") and its Figure 2.

The paper's network has two branches. A scalar branch, a two-layer perceptron 3 -> 64 -> 32 with
SiLU activations. A spectral branch, three one-dimensional convolutions 1 -> 32 -> 64 -> 64
channels with kernel sizes 9, 5 and 3, each followed by SiLU, then global average pooling to 64
features. The 32 + 64 = 96 features are fused by two fully connected layers 96 -> 128 -> 128, the
first followed by a layer norm and SiLU, the second by tanh, giving the 128-dimensional
conditioning vector for the flow.

Two adaptations for monthly production, and why:

* **Three input channels instead of one.** A spectrum is complete on a fixed grid; a well's
  history is partial and has gaps. The convolution therefore also reads an observed-month mask
  and the month index. Without the index the branch is translation invariant, and the peak's
  position, which is a parameter here, would be invisible to it.
* **Pooling over observed months only.** Global average pooling over all 60 positions would
  average in months the well has not reached yet, so a 6-month history would be diluted ten
  times more than a 60-month one.

The three scalars here are the months of history, the lateral length and the fraction of the
history that is unobserved.
"""

from __future__ import annotations

import torch
from torch import nn

from declinecast.simulator import N_CHANNELS, N_MONTHS, N_SCALARS

__all__ = ["DeclineEmbedding"]


class DeclineEmbedding(nn.Module):
    def __init__(self, n_scalars: int = N_SCALARS, n_channels: int = N_CHANNELS,
                 n_months: int = N_MONTHS, d_embed: int = 128):
        super().__init__()
        self.n_scalars, self.n_channels, self.n_months = n_scalars, n_channels, n_months
        self.scalar = nn.Sequential(nn.Linear(n_scalars, 64), nn.SiLU(),
                                    nn.Linear(64, 32), nn.SiLU())
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=9, padding=4), nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.SiLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(96, 128), nn.LayerNorm(128), nn.SiLU(),
                                    nn.Linear(128, d_embed), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scalars = x[:, :self.n_scalars]
        series = x[:, self.n_scalars:].reshape(-1, self.n_channels, self.n_months)
        mask = series[:, 1:2, :]                       # observed months
        feats = self.conv(series)
        pooled = (feats * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        return self.fusion(torch.cat([self.scalar(scalars), pooled], dim=1))
