from __future__ import annotations
import importlib
import inspect
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Optional, Type, Dict


if __name__ == "__main__" and __package__ is None:
    print("Adjusting sys.path for standalone execution...")
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    print(f"  Repo root: {repo_root}")
    sys.path.insert(0, str(repo_root))

PKG_ROOT = "streamline.p6_modeling.models"

SUBDIR = {
    "Binary": "binary_classification",
    "Multiclass": "multiclass_classification",
    "Regression": "regression",
}


def _get_models_root() -> Path:
    """
    Get the filesystem path corresponding to PKG_ROOT.
    This avoids relying on __file__ of the current module.
    """
    pkg = importlib.import_module(PKG_ROOT)
    pkg_file = getattr(pkg, "__file__", None)
    if not pkg_file:
        raise RuntimeError(f"Cannot determine filesystem path for package {PKG_ROOT!r}")
    return Path(pkg_file).parent


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

    root = _get_models_root()
    folder = root / sub
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
            getattr(cls, "model_name", "").lower().replace("_", " "),
        }
        # print(aliases, mid)
        if mid in aliases:
            return cls
    raise ValueError(f"Model '{model_id}' not found for type '{model_type}'")


# -----------------------------
# NEW: listing helpers
# -----------------------------
def _class_to_entry(cls: Type) -> Dict[str, str]:
    """Return a human/CLI-friendly entry for a discovered model class."""
    return {
        "small_name": getattr(cls, "small_name", ""),
        "alt_id": getattr(cls, "model_name", "").replace("_", " "),
        "model_name": getattr(cls, "model_name", ""),
        "model_type": getattr(cls, "model_type", ""),
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
    List all models across Binary, Multiclass, Regression.
    """
    out: List[Dict[str, str]] = []
    for mt in SUBDIR.keys():
        out.extend(list_models(mt))
    return out


if __name__ == "__main__":
    """
    Simple self-test / debug runner.

    Usage:
        python -m streamline.p6_modeling.model_registry

    or, from the repo root (if this file is model_registry.py):
        python -m streamline.p6_modeling.model_registry
    """
    import pprint
    import sys

    print("=== Model registry self-test ===")
    print(f"PKG_ROOT = {PKG_ROOT!r}")

    # 1) Sanity: list all models
    try:
        all_models = list_all_models()
    except Exception as e:
        print("ERROR: failed to list models:", repr(e))
        sys.exit(1)

    print(f"Discovered {len(all_models)} models in total.")
    if not all_models:
        print("No models were discovered. Check that:")
        print("  - The package", PKG_ROOT, "exists and is importable.")
        print("  - Subdirectories binary_classification, multiclass_classification, regression exist.")
        print("  - Each model file defines a class with small_name, model_name, model_type.")
        sys.exit(0)

    print("\nDiscovered models:")
    pprint.pprint(all_models)

    # 2) Round-trip: make sure get_model_by_id works for each entry with an id
    print("\nRunning round-trip get_model_by_id checks...")
    ok_count = 0
    for entry in all_models:
        mt = entry.get("model_type") or ""
        mid = entry.get("model_name") or ""
        
        
        if not mt or not mid:
            # skip entries without required info
            continue
        try:
            cls = get_model_by_id(mt, mid)
        except Exception as e:
            print(f"  [FAIL] type={mt!r} id={mid!r}: {e!r}")
            continue

        if getattr(cls, "small_name", None) != mid and getattr(cls, "model_name", None) != mid:
            print(
                f"  [FAIL] type={mt!r} id={mid!r}: "
                f"resolved class small_name={getattr(cls, 'small_name', None)!r}"
            )
            continue

        ok_count += 1
        print(f"  [OK] type={mt!r} id={mid!r} -> {cls.__module__}.{cls.__name__}")

    print(f"\nRound-trip checks completed. Successful lookups: {ok_count}")
    print("=== Self-test done ===")

