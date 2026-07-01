# streamline/p4_feature_importance/utils/fi_loader.py
from __future__ import annotations
import importlib, inspect, os
from pathlib import Path
from types import ModuleType
from typing import Dict, Type, Optional

PKG_BASE = "streamline.p4_feature_importance.registry"
FOLDER = Path(__file__).parent.parent / "registry"
__CACHE: Optional[Dict[str, Type]] = None

def _is_selector_class(cls: type) -> bool:
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
            if _is_selector_class(cls) and getattr(cls, "__module__", "").startswith(modname):
                found[cls.id] = cls
    return found

def list_importances() -> Dict[str, Type]:
    global __CACHE
    if __CACHE is None: __CACHE = _discover()
    return dict(__CACHE)

def normalize_importance_key(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("*", "star")
    )

def resolve_importance_id(model_id: str) -> Optional[str]:
    models = list_importances()
    if model_id in models:
        return model_id

    aliases: Dict[str, str] = {}
    for mid, cls in models.items():
        for key in filter(None, [
            mid,
            getattr(cls, "path_name", ""),
            getattr(cls, "small_name", ""),
            getattr(cls, "model_name", ""),
            getattr(cls, "__name__", ""),
        ]):
            aliases[str(key).strip().lower()] = mid
            aliases[normalize_importance_key(str(key))] = mid
    return aliases.get(str(model_id).strip().lower()) or aliases.get(normalize_importance_key(model_id))

def load_importance(model_id: str, **params):
    models = list_importances()
    resolved = resolve_importance_id(model_id)
    if resolved not in models:
        raise ValueError(f"Feature-importance model '{model_id}' not found. Available: {', '.join(sorted(models))}")
    return models[resolved](**params)
