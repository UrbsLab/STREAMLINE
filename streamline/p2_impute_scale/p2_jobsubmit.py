# streamline/phases/p2_impute_scale/p2_jobsubmit.py
import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from .impute_scale import ImputeAndScale
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from streamline.p2_impute_scale.impute_scale import ImputeAndScale

def _coalesce_bool(v: Optional[str], default: bool) -> bool:
    if v is None or v == '':
        return default
    if isinstance(v, str):
        vl = v.strip().lower()
        if vl in ('1', 'true', 'yes', 'y'):
            return True
        if vl in ('0', 'false', 'no', 'n'):
            return False
    return bool(v)


def _load_metadata(exp_path: str) -> Dict[str, Any]:
    meta_path = os.path.join(exp_path, "metadata.pickle")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            try:
                return pickle.load(f) or {}
            except Exception:
                return {}
    return {}


def parse_sampling_strategy(value):
    if value in (None, "", "auto"):
        return "auto"
    try:
        return json.loads(value)
    except Exception:
        try:
            return float(value)
        except Exception:
            return value


def main():
    ap = argparse.ArgumentParser("P2 single-CV-pair jobsubmit")
    ap.add_argument("--cv_train_path", required=True)
    ap.add_argument("--cv_test_path", required=True)
    ap.add_argument("--experiment_path", required=True)

    # Optional overrides (can be empty strings; we fallback to metadata)
    ap.add_argument("--scale_data", default=None)      # "1"/"0" or "true"/"false"
    ap.add_argument("--impute_data", default=None)
    ap.add_argument("--multi_impute", default=None)
    ap.add_argument("--overwrite_cv", default=None)

    ap.add_argument("--outcome_label", default=None)
    ap.add_argument("--outcome_type", default=None)
    ap.add_argument("--instance_label", default=None)
    ap.add_argument("--random_state", default=None)

    ap.add_argument("--imputer_id", default=None)
    ap.add_argument("--imputer_params", default="{}")  # JSON string

    ap.add_argument("--scaler_id", default=None)
    ap.add_argument("--scaler_params", default="{}")  # JSON string

    ap.add_argument("--smote", default=None)
    ap.add_argument("--smote_method", default="auto")
    ap.add_argument("--smote_sampling_strategy", default="auto")
    ap.add_argument("--smote_k_neighbors", default=5)

    args = ap.parse_args()
    exp_path = args.experiment_path

    meta = _load_metadata(exp_path)

    # coalesce params
    scale_data   = _coalesce_bool(args.scale_data,   meta.get('Use Data Scaling', True))
    impute_data  = _coalesce_bool(args.impute_data,  meta.get('Use Data Imputation', True))
    multi_impute = _coalesce_bool(args.multi_impute, meta.get('Use Multivariate Imputation', False))
    overwrite_cv = _coalesce_bool(args.overwrite_cv, True)

    outcome_label = args.outcome_label or meta.get('Outcome Label', 'Class')
    outcome_type = args.outcome_type if args.outcome_type not in (None, '') else meta.get('Outcome Type', None)
    instance_label = args.instance_label if args.instance_label not in (None, '') else meta.get('Instance Label', None)
    random_state = None
    if args.random_state not in (None, ''):
        try:
            random_state = int(args.random_state)
        except Exception:
            random_state = meta.get('Random Seed', 0)
    else:
        random_state = meta.get('Random Seed', 0)

    # imputer choices (CLI overrides metadata)
    imputer_id = args.imputer_id if args.imputer_id not in (None, '') else meta.get('P2 Imputer Id', None)
    try:
        imputer_params_cli = json.loads(args.imputer_params or "{}")
    except Exception:
        imputer_params_cli = {}
    mp = meta.get('P2 Imputer Params', '{}')
    if isinstance(mp, str):
        try:
            mp = json.loads(mp or "{}")
        except Exception:
            mp = {}
    imputer_params = imputer_params_cli or mp or {}

    scaler_id = args.scaler_id if args.scaler_id not in (None, '') else meta.get('P2 Scaler Id', None)
    try:
        scaler_params_cli = json.loads(args.scaler_params or "{}")
    except Exception:
        scaler_params_cli = {}
    sp = meta.get('P2 Scaler Params', '{}')
    if isinstance(sp, str):
        try: sp = json.loads(sp or "{}")
        except Exception: sp = {}
    scaler_params = scaler_params_cli or sp or {}

    smote = _coalesce_bool(args.smote, meta.get('Use SMOTE', False))
    smote_method = args.smote_method or meta.get('P2 SMOTE Method', 'auto')
    smote_sampling_strategy = parse_sampling_strategy(args.smote_sampling_strategy)
    if smote_sampling_strategy == "auto":
        smote_sampling_strategy = meta.get('P2 SMOTE Sampling Strategy', 'auto')
    try:
        smote_k_neighbors = int(args.smote_k_neighbors)
    except Exception:
        smote_k_neighbors = int(meta.get('P2 SMOTE K Neighbors', 5))

    job = ImputeAndScale(
        cv_train_path=args.cv_train_path,
        cv_test_path=args.cv_test_path,
        experiment_path=exp_path,
        scale_data=scale_data,
        impute_data=impute_data,
        multi_impute=multi_impute,
        overwrite_cv=overwrite_cv,
        outcome_label=outcome_label,
        outcome_type=outcome_type,
        instance_label=instance_label,
        random_state=random_state,
        imputer_id=imputer_id,
        imputer_params=imputer_params,
        scaler_id=scaler_id,
        scaler_params=scaler_params,
        smote=smote,
        smote_method=smote_method,
        smote_sampling_strategy=smote_sampling_strategy,
        smote_k_neighbors=smote_k_neighbors,
    )
    job.run()


if __name__ == "__main__":
    main()
