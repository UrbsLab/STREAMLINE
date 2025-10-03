# streamline/phases/p3_feature_learning/loader.py
from __future__ import annotations
import importlib, inspect, os
from pathlib import Path
from types import ModuleType
from typing import Dict, Type, Optional

PKG_BASE = "streamline.p3_feature_learning.registry"
FOLDER = Path(__file__).parent.parent / "registry"
__CACHE: Optional[Dict[str, Type]] = None

def _is_learner_class(cls: Type) -> bool:
    if not inspect.isclass(cls): return False
    if not hasattr(cls, "id"): return False
    return all(hasattr(cls, m) for m in ("fit", "transform", "get_feature_names", "get_params"))

def _iter_py_modules():
    for entry in os.listdir(FOLDER):
        if entry.endswith(".py") and entry != "__init__.py":
            yield f"{PKG_BASE}.{entry[:-3]}"

def _load_module(modname: str) -> Optional[ModuleType]:
    try: return importlib.import_module(modname)
    except Exception: return None

def _discover() -> Dict[str, Type]:
    found: Dict[str, Type] = {}
    for modname in _iter_py_modules():
        mod = _load_module(modname)
        if not mod: continue
        for name in dir(mod):
            obj = getattr(mod, name)
            if _is_learner_class(obj) and getattr(obj, "__module__", "").startswith(modname):
                lid = getattr(obj, "id", None)
                if isinstance(lid, str) and lid:
                    found[lid] = obj
    return found

def list_learners() -> Dict[str, Type]:
    global __CACHE
    if __CACHE is None: __CACHE = _discover()
    return dict(__CACHE)

def load_learner(learner_id: str, **params):
    learners = list_learners()
    if learner_id not in learners:
        raise ValueError(f"Learner '{learner_id}' not found. Available: {', '.join(sorted(learners))}")
    return learners[learner_id](**params)
