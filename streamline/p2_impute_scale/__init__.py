# streamline/phases/p2_impute_scale/registry/__init__.py
from __future__ import annotations
from typing import Dict, Type
import importlib, pkgutil, sys
from pathlib import Path

def _iter_registry_modules():
    pkg = __name__
    pkgpath = Path(__file__).parent
    for m in pkgutil.iter_modules([str(pkgpath)]):
        if not m.ispkg and m.name not in {"__init__"}:
            yield f"{pkg}.{m.name}"

def load_all() -> Dict[str, Type]:
    """
    Import all registry modules and merge their REGISTRY dicts.
    """
    reg: Dict[str, Type] = {}
    for modname in _iter_registry_modules():
        mod = importlib.import_module(modname)
        if hasattr(mod, "REGISTRY"):
            reg.update(getattr(mod, "REGISTRY"))
    return reg

def available_imputers() -> Dict[str, Type]:
    """
    Public helper to list all discovered imputers.
    """
    return load_all()
