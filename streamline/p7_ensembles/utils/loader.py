from __future__ import annotations
import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import List, Dict, Type, Optional


# Allow running this file directly for debugging:
if __name__ == "__main__" and __package__ is None:
    print("Adjusting sys.path for standalone execution...")
    # Heuristic: repo root = 4 levels up from this file
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    print(f"  Repo root: {repo_root}")
    sys.path.insert(0, str(repo_root))

PKG_ROOT = "streamline.p7_ensembles.registry"

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


ROOT = _get_models_root()


def _iter_modnames() -> List[str]:
    """
    Yield fully-qualified module names under the ensemble registry package.
    """
    if not ROOT.exists():
        return
    for fn in os.listdir(ROOT):
        if fn.endswith(".py") and fn not in {"__init__.py", "loader.py"}:
            yield f"{PKG_ROOT}.{fn[:-3]}"


def _safe_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def load_ensemble_classes() -> List[Type]:
    """
    Discover ensemble classes under streamline.p7_ensemble.registry.

    A valid ensemble class must:
      - Live in a module under PKG_ROOT
      - Define attributes: id, name, build_model
        (or adjust if you standardize to .build instead)
    """
    classes: List[Type] = []
    for modname in _iter_modnames() or []:
        mod = _safe_import(modname)
        if not mod:
            continue
        for name in dir(mod):
            obj = getattr(mod, name)
            if inspect.isclass(obj) and getattr(obj, "__module__", "").startswith(mod.__name__):
                required = ("id", "model_name")
                if all(hasattr(obj, k) for k in required) and name not in {"_StackingBase"}:
                    classes.append(obj)
    return sorted(classes, key=lambda c: getattr(c, "name", str(c)))


def get_ensemble_by_id(ens_id: str) -> Type:
    """
    Resolve an ensemble class by its string id (case-insensitive).
    """
    target = (ens_id or "").strip().lower()
    for cls in load_ensemble_classes():
        cid = getattr(cls, "id", "").strip().lower()
        if cid == target:
            return cls
    raise ValueError(f"Unknown ensemble id: {ens_id!r}")


def list_ensembles() -> List[Dict[str, str]]:
    """
    List discovered ensembles, returning CLI / debug-friendly entries:
      {id, name, module, qualname}
    """
    out: List[Dict[str, str]] = []
    for c in load_ensemble_classes():
        out.append({
            "id": getattr(c, "id", ""),
            "model_name": getattr(c, "model_name", ""),
            "small_name": getattr(c, "small_name", ""),
            "module": getattr(c, "__module__", ""),
            "qualname": f"{c.__module__}.{c.__name__}",
        })
    return out


# ---------------------------------------------------------------------
# Self-test / debug runner
# ---------------------------------------------------------------------
if __name__ == "__main__":
    """
    Simple self-test for the ensemble registry.

    Examples (from repo root):
        python -m streamline.p7_ensemble.utils.loader
    or:
        python streamline/p7_ensemble/utils/loader.py
    """
    import pprint

    print("=== Ensemble registry self-test ===")
    print(f"PKG_ROOT = {PKG_ROOT!r}")
    print(f"ROOT     = {ROOT!r}")

    # 1) Discover ensembles
    try:
        ensembles = list_ensembles()
    except Exception as e:
        print("ERROR: failed to list ensembles:", repr(e))
        sys.exit(1)

    print(f"Discovered {len(ensembles)} ensembles in total.")
    if not ensembles:
        print("No ensembles were discovered. Check that:")
        print(f"  - The package {PKG_ROOT} exists and is importable")
        print(f"  - The directory {ROOT} exists and contains *.py modules")
        print("  - Each ensemble class defines: id, name, build_model")
        sys.exit(0)

    print("\nDiscovered ensembles:")
    pprint.pprint(ensembles)

    # 2) Round-trip: ensure get_ensemble_by_id works for each discovered id
    print("\nRunning round-trip get_ensemble_by_id checks...")
    ok = 0
    for entry in ensembles:
        eid = entry.get("id") or ""
        if not eid:
            continue
        try:
            cls = get_ensemble_by_id(eid)
        except Exception as e:
            print(f"  [FAIL] id={eid!r}: {e!r}")
            continue

        resolved_id = getattr(cls, "id", None)
        if (resolved_id or "").lower() != eid.lower():
            print(
                f"  [FAIL] id={eid!r}: resolved class id={resolved_id!r} "
                f"({cls.__module__}.{cls.__name__})"
            )
            continue

        ok += 1
        print(f"  [OK]  id={eid!r} -> {cls.__module__}.{cls.__name__}")

    print(f"\nRound-trip checks completed. Successful lookups: {ok}")
    print("=== Self-test done ===")
