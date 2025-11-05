from __future__ import annotations
import importlib, inspect, os
from pathlib import Path
from typing import List, Dict, Type, Optional

PKG_ROOT = "streamline.p7_ensemble.registry"
ROOT = Path(__file__).parent.parent / "registry"

def _iter_modnames():
    for fn in os.listdir(ROOT):
        if fn.endswith(".py") and fn not in {"__init__.py", "loader.py"}:
            yield f"{PKG_ROOT}.{fn[:-3]}"

def _safe_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None

def load_ensemble_classes() -> List[Type]:
    classes: List[Type] = []
    for modname in _iter_modnames():
        mod = _safe_import(modname)
        if not mod:
            continue
        for name in dir(mod):
            obj = getattr(mod, name)
            if inspect.isclass(obj) and getattr(obj, "__module__", "").startswith(mod.__name__):
                required = ("id", "name", "build_model")  # see classes below
                if all(hasattr(obj, k) for k in required):
                    classes.append(obj)
    return sorted(classes, key=lambda c: getattr(c, "name", str(c)))

def get_ensemble_by_id(ens_id: str) -> Type:
    target = (ens_id or "").strip().lower()
    for cls in load_ensemble_classes():
        if getattr(cls, "id").lower() == target:
            return cls
    raise ValueError(f"Unknown ensemble id: {ens_id}")

def list_ensembles() -> List[Dict[str, str]]:
    return [{
        "id": c.id, "name": c.name,
        "module": c.__module__, "qualname": f"{c.__module__}.{c.__name__}"
    } for c in load_ensemble_classes()]
