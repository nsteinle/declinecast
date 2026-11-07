"""The model card written into a model directory, from its manifest and, when given, the
validation and scoring summaries."""

from __future__ import annotations

import numpy as np

__all__ = ["model_card"]


def _fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)


def model_card(manifest: dict, validation: dict | None = None,
               scores: dict | None = None) -> str:
    m = manifest
    tr, fl, pr = m["training"], m["flow"], m["prior"]
    th = m.get("training_history", {})
    lines = [
        "# Model card: declinecast pretrained network",
        "",
        f"declinecast {m['package_version']}, created {m['created_utc']}.",
        "",
        "## What it does",
        "",
        "Given the first 3 to 60 months of a horizontal DJ Basin oil well's monthly production "
        "(oil and producing days) and its lateral length, it returns a posterior over six "
        "decline parameters, and from them a forecast of the well's cumulative oil with "
        "intervals. It is a neural posterior estimate, trained only on simulated wells.",
        "",
        "## Intended use and limits",
        "",
        "* Screening and portfolio work on horizontal Niobrara and Codell oil wells in the DJ "
        "Basin, at 6 to 24 months of history, for cumulative oil to 36 (and, with the vintage "
        "caveat in FINDINGS, 60) months.",
        "* Not a reserves estimate: no economic limit, no price deck, no SEC or PRMS process.",
        "* Not validated outside the DJ Basin, for gas or water, for refracs, or for wells "
        "affected by offset-well interference; flagged wells (`declinecast.ood`) are outside "
        "what the simulator was checked against.",
        "",
        "## Model",
        "",
        f"* Flow: neural spline flow, {fl['num_transforms']} masked autoregressive "
        f"transformations, {fl['num_bins']}-bin rational-quadratic splines, conditioner "
        f"hidden layers {fl['hidden_layers']} (sbi {m['versions']['sbi']} with zuko "
        f"{m['versions']['zuko']}).",
        "* Embedding: the two-branch network of Steinle et al. (arXiv:2607.12971, section VI A), "
        "written from the paper's text, with three input channels and pooling over observed "
        "months (`declinecast/embedding.py`).",
        f"* Parameters: {', '.join(m['parameters'])}; support lower "
        f"{np.round(m['support']['lower'], 3).tolist()}, upper "
        f"{np.round(m['support']['upper'], 3).tolist()}.",
        f"* Terminal decline {m['terminal_decline']['effective_per_year']:.0%} per year "
        "(effective).",
        f"* Prior: log10 qi given lateral L ~ N({pr['qi_intercept']:.3f} + "
        f"{pr['qi_slope']:.3f} log10(L / 10,000 ft), {pr['qi_sd']:.3f}), truncated; the rest "
        "uniform. Lateral and downtime drawn from development wells (first produced "
        "2014-2019).",
        "",
        "## Training",
        "",
        f"* {tr['n_simulations']:,} simulated wells, {tr['validation_fraction']:.0%} held out "
        f"for early stopping; batch {tr['batch_size']}, Adam at {tr['learning_rate']:g}, "
        f"learning rate halved after {tr.get('lr_patience', 'n/a')} flat epochs, "
        f"gradient norm clipped at "
        f"{tr['clip_max_norm']}, at most {tr['max_epochs']} epochs, seed {tr['seed']}.",
        f"* Kept epoch {th.get('kept_epoch')} of {th.get('epochs_run')} run; best validation "
        f"loss {_fmt(th.get('best_validation_loss'))}.",
        f"* Noise: {'on' if tr['noise'] else 'off (ablation)'}; lateral input: "
        f"{'on' if tr['use_lateral'] else 'off (ablation)'}.",
        f"* Versions: {', '.join(f'{k} {v}' for k, v in m['versions'].items())}.",
    ]
    adj = m.get("forecast_adjustment")
    if adj:
        f = adj["factors"]
        lines += [
            "",
            "## Recalibration",
            "",
            "The predicted (future) oil is multiplied by a factor per history and horizon, fit so "
            f"the median log error on {adj['fit_on']['wells']:,} development wells is zero "
            "(`scripts/recalibrate.py`): "
            + ", ".join(f"{k.replace('-', ' months to ')} months: {v}" for k, v in f.items())
            + ". Applied by default (`forecast(..., adjust=False)` turns it off).",
        ]
    if validation:
        v = validation
        sbc = v["sbc_pass"]
        lines += [
            "",
            "## Calibration on held-out simulated wells",
            "",
            f"* SBC ({v['settings']['n_sbc']:,} wells, {v['settings']['n_draws']} draws each, "
            f"KS at a 5% family-wise level): passes for "
            f"{', '.join(k for k, ok in sbc.items() if ok) or 'none'}"
            + (f"; fails for {', '.join(k for k, ok in sbc.items() if not ok)}"
               if not all(sbc.values()) else "") + ".",
            f"* Coverage cells consistent with nominal (95% Clopper-Pearson): "
            f"{v['coverage_consistent']}; largest gap {v['worst_coverage_gap']:.3f}.",
            f"* Posterior draws inside the support: median {v['acceptance']['median']:.3f}, "
            f"1st percentile {v['acceptance']['p01']:.3f}.",
        ]
    if scores:
        lines += ["", "## Scores on real wells", ""]
        for label, rows in scores.items():
            lines.append(f"**{label}**")
            lines.append("")
            lines.append("| history | horizon | method | wells | WIS | median error | "
                         "80% coverage |")
            lines.append("|---|---|---|---|---|---|---|")
            for r in rows:
                lines.append(f"| {r['history']} | {r['horizon']} | {r['method']} | {r['n']:,} | "
                             f"{r['wis']:.4f} | {100 * r['median_abs_pct_error']:.1f}% | "
                             f"{100 * r['coverage_80']:.1f}% |")
            lines.append("")
    lines += ["", "## Files", "", "| file | SHA-256 |", "|---|---|"]
    lines += [f"| {k} | `{h}` |" for k, h in m["files"].items() if k != "MODEL_CARD.md"]
    return "\n".join(lines) + "\n"
