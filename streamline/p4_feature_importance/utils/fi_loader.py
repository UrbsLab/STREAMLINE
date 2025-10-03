# streamline/phases/p4_feature_selection/loader.py
from __future__ import annotations
import importlib, inspect, os
from pathlib import Path
from types import ModuleType
from typing import Dict, Type, Optional

PKG_BASE = "streamline.p4_feature_selection.registry"
FOLDER = Path(__file__).parent.parent / "registry"
__CACHE: Optional[Dict[str, Type]] = None

def _is_selector_class(cls: Type) -> bool:
    need = ("fit","transform","get_support_mask","get_support_names","get_scores","get_params")
    return inspect.isclass(cls) and hasattr(cls,"id") and all(hasattr(cls, m) for m in need)

def _iter_modules():
    for f in os.listdir(FOLDER):
        if f.endswith(".py") and f != "__init__.py":
            yield f"{PKG_BASE}.{f[:-3]}"

def _load(modname: str) -> Optional[ModuleType]:
    try: return importlib.import_module(modname)
    except Exception: return None

def _discover() -> Dict[str, Type]:
    found: Dict[str, Type] = {}
    for modname in _iter_modules():
        mod = _load(modname)
        if not mod: continue
        for name in dir(mod):
            cls = getattr(mod, name)
            if _is_selector_class(cls) and getattr(cls,"__module__","").startswith(modname):
                found[cls.id] = cls
    return found

def list_selectors() -> Dict[str, Type]:
    global __CACHE
    if __CACHE is None: __CACHE = _discover()
    return dict(__CACHE)

def load_selector(selector_id: str, **params):
    sels = list_selectors()
    if selector_id not in sels:
        raise ValueError(f"Selector '{selector_id}' not found. Available: {', '.join(sorted(sels))}")
    return sels[selector_id](**params)
