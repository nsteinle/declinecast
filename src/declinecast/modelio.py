"""Saving and loading a trained model directory, with no pickles.

A model directory holds exactly:

* ``manifest.json``: format version, package and library versions, the flow and training
  settings, the prior (every number needed to rebuild it), the observation scalings, the
  terminal decline, the training history, and the SHA-256 of every other file;
* ``weights.pt``: the network's ``state_dict`` (tensors only);
* ``MODEL_CARD.md``: the generated model card.

Loading checks every checksum, refuses files the manifest does not list, rebuilds the network
from the settings in the manifest and reads the weights with ``torch.load(weights_only=True)``,
which refuses anything but tensors and plain containers.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from declinecast import arps
from declinecast._version import __version__
from declinecast.inference import FlowConfig, build_estimator
from declinecast.priors import LOWER, UPPER, Prior
from declinecast.simulator import OBS, PARAMS, x_dim

__all__ = ["FORMAT_VERSION", "write_model", "load_model", "LoadedModel", "update_manifest",
           "write_model_card"]

FORMAT_VERSION = 1
_FILES = ("weights.pt", "MODEL_CARD.md")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _versions() -> dict:
    import duckdb
    import numpy
    import pandas
    import sbi
    import zuko

    return {"python": platform.python_version(), "torch": torch.__version__,
            "sbi": sbi.__version__, "zuko": zuko.__version__, "numpy": numpy.__version__,
            "pandas": pandas.__version__, "duckdb": duckdb.__version__}


@dataclass
class LoadedModel:
    estimator: torch.nn.Module
    prior: Prior
    flow_config: FlowConfig
    manifest: dict
    path: Path

    @property
    def lower(self) -> np.ndarray:
        return np.asarray(self.manifest["support"]["lower"], float)

    @property
    def upper(self) -> np.ndarray:
        return np.asarray(self.manifest["support"]["upper"], float)

    def posterior(self, x: np.ndarray, n: int = 500, seed: int = 0):
        """Posterior draws (rows, n, 6) for encoded inputs ``x``, and the acceptance rate. A
        model trained without the lateral (the ablation) gets the reference lateral, as in its
        training."""
        from declinecast.inference import sample

        if not self.manifest["training"].get("use_lateral", True):
            x = np.array(x, copy=True)
            x[:, 1] = 0.0
        return sample(self.estimator, x, n, self.lower, self.upper, seed=seed)


def write_model(path: str | Path, estimator, prior: Prior, flow_config: FlowConfig,
                train_config: dict, history: dict, extra: dict | None = None,
                model_card: str = "", overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} is not empty; use --overwrite to replace its files")
    path.mkdir(parents=True, exist_ok=True)
    for f in path.iterdir():
        if f.is_file():
            f.unlink()
    state = {k: v.detach().cpu().contiguous() for k, v in estimator.state_dict().items()}
    torch.save(state, path / "weights.pt")
    (path / "MODEL_CARD.md").write_text(model_card or "# Model card\n\n(not yet generated)\n")
    manifest = {
        "format_version": FORMAT_VERSION,
        "package": "declinecast",
        "package_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parameters": list(PARAMS),
        "support": {"lower": LOWER.tolist(), "upper": UPPER.tolist()},
        "terminal_decline": {"effective_per_year": arps.DMIN_EFFECTIVE,
                             "nominal_per_year": arps.DMIN_NOMINAL},
        "observation": {"x_dim": x_dim(), **OBS.__dict__},
        "flow": flow_config.to_dict(),
        "training": train_config,
        "training_history": {"kept_epoch": history["kept_epoch"],
                             "best_validation_loss": history["best_validation_loss"],
                             "epochs_run": len(history["epochs"]),
                             "seconds": history["epochs"][-1]["seconds"]},
        "prior": prior.to_dict(),
        "versions": _versions(),
        **(extra or {}),
        "files": {name: _sha256(path / name) for name in _FILES},
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def update_manifest(path: str | Path, **fields) -> None:
    """Add fields to a saved manifest (validation results, the model card) and refresh the
    checksums."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest.update(fields)
    manifest["files"] = {name: _sha256(path / name) for name in _FILES}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_model(path: str | Path | None = None) -> LoadedModel:
    """Load a model directory (default: the pretrained model bundled with the package)."""
    if path is None:
        from importlib import resources

        with resources.as_file(resources.files("declinecast") / "data" / "model") as p:
            return load_model(Path(p))
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported model format {manifest.get('format_version')}")
    expected = set(manifest["files"]) | {"manifest.json"}
    present = {f.name for f in path.iterdir() if f.is_file()}
    unexpected = present - expected
    if unexpected:
        raise ValueError(f"{path} holds files the manifest does not list: {sorted(unexpected)}")
    for name, digest in manifest["files"].items():
        if _sha256(path / name) != digest:
            raise ValueError(f"{name} does not match its checksum in manifest.json")
    flow_cfg = FlowConfig(**{**manifest["flow"],
                             "hidden_layers": tuple(manifest["flow"]["hidden_layers"])})
    prior = Prior.from_dict(manifest["prior"])
    gen = np.random.default_rng(0)
    theta = torch.as_tensor(np.column_stack(
        [gen.uniform(lo, hi, 64) for lo, hi in zip(LOWER, UPPER, strict=True)]),
        dtype=torch.float32)
    x = torch.as_tensor(gen.normal(size=(64, x_dim())), dtype=torch.float32)
    est = build_estimator(theta, x, flow_cfg)
    state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
    est.load_state_dict(state)
    est.eval()
    return LoadedModel(est, prior, flow_cfg, manifest, path)


def write_model_card(path: str | Path, validation: dict | None = None,
                     scores: dict | None = None) -> None:
    """Regenerate ``MODEL_CARD.md`` from the manifest (and results), then the checksums."""
    from declinecast.modelcard import model_card

    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    (path / "MODEL_CARD.md").write_text(model_card(manifest, validation, scores))
    update_manifest(path)
