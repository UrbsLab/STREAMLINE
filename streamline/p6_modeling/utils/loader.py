from __future__ import annotations
import importlib, inspect, os
from pathlib import Path
from types import ModuleType
from typing import List, Optional, Type

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
    for modname in _iter_modnames(folder, package):
        mod = _try_import(modname)
        if not mod:
            continue
        for name in dir(mod):
            obj = getattr(mod, name)
            if inspect.isclass(obj) and getattr(obj, "__module__", "").startswith(mod.__name__):
                # Must expose legacy fields used by ModelJob
                if all(hasattr(obj, k) for k in ("small_name", "model_name", "model_type", "param_grid", "model_evaluation")):
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
