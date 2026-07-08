from __future__ import annotations
import importlib
import inspect
import os
from pathlib import Path
from types import ModuleType
from typing import Dict, Type, Optional

# We keep everything phase-local.
PKG_BASE = "streamline.p2_impute_scale.registry.impute"
FOLDER = Path(__file__).parent.parent / "registry" / "impute"

# Simple cache so we only scan once per process.
__CACHE: Optional[Dict[str, Type]] = None


def _is_imputer_class(cls: Type) -> bool:
    """
    Heuristic check: class has 'id' attr (str), and methods fit/transform/get_params.
    We don't import the Protocol to avoid circular deps; duck-typing is fine here.
    """
    if not inspect.isclass(cls):
        return False
    if not hasattr(cls, "id"):
        return False
    # minimal surface
    return all(hasattr(cls, m) for m in ("fit", "transform", "get_params"))


def _iter_py_modules():
    for entry in os.listdir(FOLDER):
        if not entry.endswith(".py"):
            continue
        if entry == "__init__.py":
            continue
        modname = f"{PKG_BASE}.{entry[:-3]}"
        yield modname


def _load_module(modname: str) -> Optional[ModuleType]:
    try:
        return importlib.import_module(modname)
    except Exception:
        # Swallow import errors so one bad file doesn't stop discovery.
        return None


def _discover() -> Dict[str, Type]:
    found: Dict[str, Type] = {}
    for modname in _iter_py_modules():
        mod = _load_module(modname)
        if not mod:
            continue
        for name in dir(mod):
            obj = getattr(mod, name)
            if _is_imputer_class(obj):
                # prefer classes defined in this module
                if getattr(obj, "__module__", "").startswith(modname):
                    imputer_id = getattr(obj, "id", None)
                    if isinstance(imputer_id, str) and imputer_id:
                        # last-one-wins if duplicate ids; you can warn here if you like
                        found[imputer_id] = obj
    return found


def list_imputers() -> Dict[str, Type]:
    """Return {imputer_id: class} discovered under registry/ (cached)."""
    global __CACHE
    if __CACHE is None:
        __CACHE = _discover()
    return dict(__CACHE)


def load_imputer(imputer_id: str, **params):
    """Instantiate an imputer by id using dynamic discovery."""
    imps = list_imputers()
    if imputer_id not in imps:
        raise ValueError(f"Imputer '{imputer_id}' not found. Available: {', '.join(sorted(imps))}")
    return imps[imputer_id](**params)
