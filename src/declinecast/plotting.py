"""Figures for the docs (needs the ``plot`` extra, matplotlib)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["pp_plot", "reliability_plot", "fan_plot"]

_COLORS = {"network": "#1f5f8b", "multiplier": "#c26a1b", "type_curve": "#7a7a7a",
           "arps_ls": "#8b1f3f"}
_NAMES = {"network": "declinecast network", "multiplier": "history multiplier",
          "type_curve": "type curve (lateral only)", "arps_ls": "least-squares Arps"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def pp_plot(ranks: pd.DataFrame, band: float, out: str | Path, title: str = "") -> Path:
    """Empirical CDF of the SBC ranks per quantity, with the KS band."""
    plt = _plt()
    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.fill_between(grid, np.clip(grid - band, 0, 1), np.clip(grid + band, 0, 1),
                    color="0.88", label=f"95% KS band (Bonferroni, n={len(ranks):,})")
    ax.plot([0, 1], [0, 1], color="0.5", lw=0.8)
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(ranks.columns):
        r = np.sort(ranks[c].to_numpy())
        ax.plot(grid, np.searchsorted(r, grid, side="right") / len(r), lw=1.2,
                color=cmap(i % 10), label=c)
    ax.set(xlabel="credible level", ylabel="fraction of true values below it",
           xlim=(0, 1), ylim=(0, 1), title=title)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return Path(out)


def reliability_plot(reliability: pd.DataFrame, out: str | Path, horizon: int = 36,
                     title: str = "") -> Path:
    """Observed against nominal coverage on real wells, one panel per history."""
    plt = _plt()
    r = reliability[reliability.horizon_months == horizon]
    hs = sorted(r.history_months.unique())
    fig, axes = plt.subplots(1, len(hs), figsize=(3.0 * len(hs), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, hs, strict=True):
        ax.plot([0.4, 1], [0.4, 1], color="0.6", lw=0.8)
        for m, g in r[r.history_months == h].groupby("method"):
            key = next((k for k, v in _NAMES.items() if v.lower() in m.lower() or k == m), m)
            g = g.sort_values("nominal")
            ax.plot(g.nominal, g.observed, marker="o", ms=3, lw=1.2,
                    color=_COLORS.get(key, "k"), label=m)
        ax.set(title=f"{h} months seen", xlabel="nominal interval", xlim=(0.45, 1.0),
               ylim=(0.2, 1.0))
    axes[0].set_ylabel(f"share of actual {horizon}-month cumulatives inside")
    axes[-1].legend(fontsize=7, loc="lower right", frameon=False)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return Path(out)


def fan_plot(examples: pd.DataFrame, out: str | Path, n: int = 4) -> Path:
    """Cumulative-oil forecast fans for a few example wells, with what they produced."""
    plt = _plt()
    apis = examples.api.unique()[:n]
    fig, axes = plt.subplots(1, len(apis), figsize=(3.1 * len(apis), 3.0))
    axes = np.atleast_1d(axes)
    for ax, api in zip(axes, apis, strict=True):
        g = examples[examples.api == api]
        f = g[g.phase == "forecast"]
        k = f.month.to_numpy()
        ax.fill_between(k, f["q0.05"] / 1e3, f["q0.95"] / 1e3, color="#1f5f8b", alpha=0.18,
                        lw=0, label="90%")
        ax.fill_between(k, f["q0.1"] / 1e3, f["q0.9"] / 1e3, color="#1f5f8b", alpha=0.3, lw=0,
                        label="80% (P90-P10)")
        ax.plot(k, f["q0.5"] / 1e3, color="#1f5f8b", lw=1.2, label="median")
        ax.plot(g.month, g.actual_bbl / 1e3, color="k", lw=1.0, label="actual")
        ax.axvline(g[g.phase == "history"].month.max(), color="0.6", lw=0.8, ls=":")
        ax.set(title=f"API {api[:10]}", xlabel="months since first production")
    axes[0].set_ylabel("cumulative oil (thousand bbl)")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return Path(out)
