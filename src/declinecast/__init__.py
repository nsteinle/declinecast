"""declinecast: amortized Bayesian decline-curve analysis on real regulator data.

A network trained on simulated wells returns, for any well's first months of production, a
posterior over its decline parameters and a forecast of its cumulative oil with intervals. The
posterior's calibration is checked on simulated wells (simulation-based calibration and
coverage) and the forecasts are scored on real DJ Basin wells against three baselines. The panel
build, quality checks and Tableau tables are DuckDB SQL (``declinecast.sql``).

The heavy imports (torch, sbi) load on first use of the names that need them.
"""

from declinecast._version import __version__

__all__ = [
    "__version__",
    "load_panel",
    "well_arrays",
    "load_model",
    "forecast",
    "uptime_table",
    "validate",
    "evaluate",
    "predict_table",
    "wells_from_table",
]

_LAZY = {
    "load_panel": "declinecast.datasets",
    "well_arrays": "declinecast.datasets",
    "load_model": "declinecast.modelio",
    "forecast": "declinecast.forecasting",
    "uptime_table": "declinecast.forecasting",
    "validate": "declinecast.validation",
    "evaluate": "declinecast.evaluation",
    "predict_table": "declinecast.pipeline",
    "wells_from_table": "declinecast.pipeline",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'declinecast' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
