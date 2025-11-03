from __future__ import annotations
import importlib, inspect, os
from pathlib import Path
from types import ModuleType
from typing import List, Optional, Type, Dict

PKG_ROOT = "streamline.p6_modeling.models"
ROOT = Path(__file__).parent.parent / "models"

SUBDIR = {
    "BinaryClassification": "binary_classification",
    "MulticlassClassification": "multiclass_classification",
    "Regression": "regression",
}

def _iter_modnames(folder: Path, package: str):
    for fn in os.listdir(folder):
        if fn.endswith(".py") and fn != "__init__.py":
            yield f"{package}.{fn[:-3]}"

def _try_import(modname: str) -> Optional[ModuleType]:
    try:
        return importlib.import_module(modname)
    except Exception:
        return None

def load_model_classes(model_type: str) -> List[Type]:
    sub = SUBDIR.get(model_type)
    if sub is None:
        raise ValueError(f"Unknown model_type '{model_type}'")
    folder = ROOT / sub
    package = f"{PKG_ROOT}.{sub}"
    classes: List[Type] = []
    if not folder.exists():
        return classes
    for modname in _iter_modnames(folder, package):
        mod = _try_import(modname)
        if not mod:
            continue
        for name in dir(mod):
            obj = getattr(mod, name)
            if inspect.isclass(obj) and getattr(obj, "__module__", "").startswith(mod.__name__):
                # Must expose these attrs (used by Phase 6)
                required = ("small_name", "model_name", "model_type")
                if all(hasattr(obj, k) for k in required):
                    classes.append(obj)
    return sorted(classes, key=lambda c: getattr(c, "model_name", str(c)))

def get_model_by_id(model_type: str, model_id: str) -> Type:
    mid = (model_id or "").strip().lower()
    for cls in load_model_classes(model_type):
        aliases = {
            getattr(cls, "small_name", "").lower(),
            getattr(cls, "model_name", "").lower().replace(" ", "_"),
        }
        if mid in aliases:
            return cls
    raise ValueError(f"Model '{model_id}' not found for type '{model_type}'")

# -----------------------------
# NEW: listing helpers
# -----------------------------
def _class_to_entry(cls: Type) -> Dict[str, str]:
    """Return a human/CLI-friendly entry for a discovered model class."""
    return {
        "id": getattr(cls, "small_name", ""),
        "alt_id": getattr(cls, "model_name", "").replace(" ", "_"),
        "name": getattr(cls, "model_name", ""),
        "type": getattr(cls, "model_type", ""),
        "module": getattr(cls, "__module__", ""),
        "qualname": f"{cls.__module__}.{cls.__name__}",
    }

def list_models(model_type: str) -> List[Dict[str, str]]:
    """
    List available models for a given model_type.
    Returns a list of dict entries with:
      {id, alt_id, name, type, module, qualname}
    """
    return [_class_to_entry(c) for c in load_model_classes(model_type)]

def list_all_models() -> List[Dict[str, str]]:
    """
    List all models across BinaryClassification, MulticlassClassification, Regression.
    """
    out: List[Dict[str, str]] = []
    for mt in SUBDIR.keys():
        out.extend(list_models(mt))
    return out
