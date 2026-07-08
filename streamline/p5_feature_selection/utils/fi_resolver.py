# streamline/phases/p5_feature_selection/utils.py
from __future__ import annotations
import os
from typing import Dict, List, Tuple, Optional

def normalize_algorithm_key(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("*", "star")
    )

def _safe_list_models() -> Dict[str, type]:
    """
    Try to import the Phase-4 loader and list models (id -> class).
    Falls back to {} if p4 is not installed/available.
    """
    try:
        from streamline.p4_feature_importance.utils.fi_loader import list_importances  # type: ignore
        return list_importances() or {}
    except Exception:
        return {}

def _build_alias_map(models: Dict[str, type]) -> Dict[str, str]:
    """
    Build a dictionary mapping aliases (id, small_name, path_name, model_name)
    to the canonical path_name (i.e., folder name under feature_importance/).
    """
    alias: Dict[str, str] = {}
    for mid, cls in models.items():
        path_name = getattr(cls, "path_name", mid).strip().lower()
        small = getattr(cls, "small_name", "").strip()
        mname = getattr(cls, "model_name", "").strip()

        for key in filter(None, [
            mid,
            path_name,
            small,
            mname,
            getattr(cls, "__name__", "").strip(),
        ]):
            alias[key.lower()] = path_name
            alias[normalize_algorithm_key(key)] = path_name
    return alias

def _scan_fs_algorithms(fs_root: str) -> List[str]:
    """
    Return a list of subfolders under feature_importance/, which correspond to available algorithms.
    """
    if not os.path.isdir(fs_root):
        return []
    return sorted([
        name for name in os.listdir(fs_root)
        if os.path.isdir(os.path.join(fs_root, name))
        and not name.startswith(".")
    ])

def resolve_algorithms(dataset_dir: str, algorithms: Optional[str | List[str]]) -> List[str]:
    """
    Resolve user-provided algorithms to Phase-4 path_names, dynamically.
    - If algorithms is None or 'auto', discover from filesystem: feature_importance/*/ .
    - Else, map each token against P4 registry (id/small_name/path_name/model_name),
      falling back to filesystem names when needed.
    Returns a list of unique path_names (folder names).
    """
    fs_root = os.path.join(dataset_dir, "feature_importance")
    models = _safe_list_models()
    alias = _build_alias_map(models)  # alias -> path_name
    available_fs = set(_scan_fs_algorithms(fs_root))

    # Auto-discover everything present
    if algorithms is None or (isinstance(algorithms, str) and algorithms.strip().lower() == "auto"):
        return sorted(list(available_fs))

    # Normalize CSV/string/list to tokens
    if isinstance(algorithms, str):
        tokens = [t.strip() for t in algorithms.split(",") if t.strip()]
    else:
        tokens = [str(t).strip() for t in algorithms if str(t).strip()]

    resolved: List[str] = []
    seen = set()
    for tok in tokens:
        key = tok.lower()
        # try registry alias (id/small/safe names)
        pn = alias.get(key) or alias.get(normalize_algorithm_key(tok))
        if pn is None:
            # if user passed a folder name, and it exists, accept it
            if key in available_fs:
                pn = key
            else:
                # last resort: strip spaces/normalize and see if present
                key2 = key.replace(" ", "")
                pn = key2 if key2 in available_fs else None
        if pn and pn not in seen:
            resolved.append(pn)
            seen.add(pn)

    # If nothing resolved AND user asked explicitly, surface a helpful error
    if not resolved:
        msg = "No algorithms resolved. Known (P4 registry): "
        if alias:
            msg += ", ".join(sorted(set(alias.keys()))) + ". "
        if available_fs:
            msg += "Found on disk: " + ", ".join(sorted(available_fs))
        raise ValueError(msg)

    return resolved

# streamline/phases/p5_feature_selection/runner.py  (add near top)
import glob
import logging

# If someone passes small names, map → folder names used by Phase 4
ALG_MAP = {
    "MI": "mutualinformation",
    "MS": "multisurf",
    "MS*": "multisurfstar",
    "MSS": "multisurfstar",
    "MULTISURF": "multisurf",
    "MULTISURF*": "multisurfstar",
    "MULTISURFSTAR": "multisurfstar",
    "MSWRFDB": "multiswrfdb",
    "MSWRFDB*": "multiswrfdbstar",
    "MULTISWRFDB": "multiswrfdb",
    "MULTISWRFDB*": "multiswrfdbstar",
    "MULTISWRFDBSTAR": "multiswrfdbstar",
}

def _normalize_algorithms(algorithms):
    """Accept list or CSV; map small names to path names; dedupe/preserve order."""
    if algorithms is None:
        return None
    if isinstance(algorithms, str):
        algorithms = [a.strip() for a in algorithms.split(",") if a.strip()]
    out, seen = [], set()
    for a in algorithms:
        key = str(a).strip()
        a = ALG_MAP.get(
            key,
            ALG_MAP.get(key.upper(), ALG_MAP.get(normalize_algorithm_key(key).upper(), key)),
        )
        if a not in seen:
            out.append(a); seen.add(a)
    return out

def _discover_algorithms(dataset_dir: str, n_splits: int, strict: bool = False):
    """
    Scan <dataset_dir>/feature_importance/*/ for *_scores_cv_*.csv files.
    If strict=True, require exactly n_splits files per algorithm.
    Returns list of folder names (e.g., ["mutualinformation","multisurf"]).
    """
    root = os.path.join(dataset_dir, "feature_importance")
    if not os.path.isdir(root):
        return []
    algs = []
    for name in sorted(os.listdir(root)):
        alg_dir = os.path.join(root, name)
        if not os.path.isdir(alg_dir):
            continue
        # match the Phase 4 file convention
        pattern = os.path.join(alg_dir, f"{name}_scores_cv_*.csv")
        files = glob.glob(pattern)
        if not files:
            continue
        if strict:
            # keep only if we have all splits 0..n_splits-1
            expected = [os.path.join(alg_dir, f"{name}_scores_cv_{i}.csv") for i in range(n_splits)]
            if not all(os.path.exists(p) for p in expected):
                continue
        algs.append(name)
    return algs
