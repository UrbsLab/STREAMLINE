from __future__ import annotations
import importlib, inspect, os
from pathlib import Path
from types import ModuleType
from typing import Dict, Type, Optional

PKG = "streamline.p5_feature_selection.registry"
FOLDER = Path(__file__).parent.parent  / "registry"
_CACHE: Optional[Dict[str, Type]] = None

def _is_strategy(cls: type) -> bool:
    need = ("select",)  # one public entrypoint
    return inspect.isclass(cls) and hasattr(cls, "id") and all(hasattr(cls, m) for m in need)

def _iter_modules():
    for f in os.listdir(FOLDER):
        if f.endswith(".py") and f not in {"__init__.py", "loader.py"}:
            yield f"{PKG}.{f[:-3]}"

def _try_import(modname: str) -> Optional[ModuleType]:
    try: return importlib.import_module(modname)
    except Exception: return None

def _discover() -> Dict[str, Type]:
    found: Dict[str, Type] = {}
    for mod in map(_try_import, _iter_modules()):
        if not mod: continue
        for name in dir(mod):
            cls = getattr(mod, name)
            if _is_strategy(cls) and getattr(cls, "__module__", "").startswith(mod.__name__):
                found[cls.id] = cls
    return found

def list_strategies() -> Dict[str, Type]:
    global _CACHE
    if _CACHE is None: _CACHE = _discover()
    return dict(_CACHE)

def load_strategy(selector_id: str, **params):
    strategies = list_strategies()
    if selector_id not in strategies:
        raise ValueError(f"Unknown P5 selector '{selector_id}'. Available: {', '.join(sorted(strategies))}")
    return strategies[selector_id](**params)
