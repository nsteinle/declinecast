"""Engineering rules from the plan, checked on the source tree."""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import declinecast

SRC = Path(declinecast.__file__).parent


def _modules():
    for m in pkgutil.walk_packages([str(SRC)], prefix="declinecast."):
        yield m.name, importlib.import_module(m.name)


def test_no_module_shares_a_name_with_a_function_it_exports():
    for name, mod in _modules():
        short = name.rsplit(".", 1)[-1]
        assert short not in getattr(mod, "__all__", []), name


def test_no_public_function_starts_with_test():
    for name, mod in _modules():
        for attr in getattr(mod, "__all__", []):
            assert not attr.startswith("test_"), f"{name}.{attr}"


def test_no_pickle_and_safe_torch_loads():
    for path in SRC.rglob("*.py"):
        text = path.read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                assert not any(n.split(".")[0] in ("pickle", "joblib", "dill", "cloudpickle")
                               for n in names), path
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "load"
                    and getattr(node.func.value, "id", "") == "torch"):
                kw = {k.arg: k.value for k in node.keywords}
                assert "weights_only" in kw and kw["weights_only"].value is True, path


def test_fetch_is_the_only_network_code():
    markers = ("urllib.request", "import requests", "http.client", "import httpx", "socket")
    for path in SRC.rglob("*.py"):
        text = path.read_text()
        if any(m in text for m in markers):
            assert path.name == "ecmc.py", path
