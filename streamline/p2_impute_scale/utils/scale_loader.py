from __future__ import annotations
import importlib
import inspect
import os
from pathlib import Path
from types import ModuleType
from typing import Dict, Type, Optional

# We keep everything phase-local.
PKG_BASE = "streamline.p2_impute_scale.registry.scale"
FOLDER = Path(__file__).parent.parent / "registry" / "scale"

# Simple cache so we only scan once per process.
__CACHE: Optional[Dict[str, Type]] = None


def _is_scaler_class(cls: Type) -> bool:
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
            if _is_scaler_class(obj):
                # prefer classes defined in this module
                if getattr(obj, "__module__", "").startswith(modname):
                    scaler_id = getattr(obj, "id", None)
                    if isinstance(scaler_id, str) and scaler_id:
                        # last-one-wins if duplicate ids; you can warn here if you like
                        found[scaler_id] = obj
    return found


def list_scalers() -> Dict[str, Type]:
    """Return {scaler_id: class} discovered under registry/ (cached)."""
    global __CACHE
    if __CACHE is None:
        __CACHE = _discover()
    return dict(__CACHE)


def load_scaler(scaler_id: str, **params):
    """Instantiate an scaler by id using dynamic discovery."""
    imps = list_scalers()
    if scaler_id not in imps:
        raise ValueError(f"scaler '{scaler_id}' not found. Available: {', '.join(sorted(imps))}")
    return imps[scaler_id](**params)
