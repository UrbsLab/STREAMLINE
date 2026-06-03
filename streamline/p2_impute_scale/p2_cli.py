# streamline/phases/p2_impute_scale/cli.py
import argparse
import json
from typing import Optional, Dict, Any

from streamline.p2_impute_scale.p2_runner import P2Runner
from streamline.p2_impute_scale.utils.impute_loader import list_imputers
from streamline.p2_impute_scale.utils.scale_loader import list_scalers
from streamline.utils.run_commands import (
    add_run_command_args,
    apply_saved_run_command,
    save_run_command_from_args,
    snapshot_args,
)


def _maybe_json(v: Optional[str]) -> Dict[str, Any]:
    if v in (None, "", "{}", "null"):
        return {}
    try:
        return json.loads(v)
    except Exception:
        return {}


def _bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    vl = str(v).strip().lower()
    if vl in ("1", "true", "t", "yes", "y"):
        return True
    if vl in ("0", "false", "f", "no", "n"):
        return False
    return default


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
    ap = argparse.ArgumentParser(
        "STREAMLINE Phase 2 Runner (CV-based, Dask-aware)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--experiment_name", required=True)

    # Optional overrides; if omitted, pulled from metadata.pickle
    ap.add_argument("--scale_data", default=None)
    ap.add_argument("--impute_data", default=None)
    ap.add_argument("--multi_impute", default=None)
    ap.add_argument("--overwrite_cv", default=None)
    ap.add_argument("--outcome_label", default=None)
    ap.add_argument("--outcome_type", default=None, choices=["Binary", "Multiclass", "Continuous"])
    ap.add_argument("--instance_label", default=None)
    ap.add_argument("--random_state", default=None, type=int)

    # Imputer registry
    ap.add_argument("--imputer_id", default=None)
    ap.add_argument("--imputer_params", default="{}")

    # Scaler registry
    ap.add_argument("--scaler_id", default=None)
    ap.add_argument("--scaler_params", default="{}")

    # SMOTE/SMOTENC oversampling; applied after imputation/scaling to train folds only.
    ap.add_argument("--smote", default=None, help="1/true enables SMOTE for classification training folds")
    ap.add_argument("--smote_method", default="auto", choices=["auto", "smote", "smotenc"])
    ap.add_argument("--smote_sampling_strategy", default="auto")
    ap.add_argument("--smote_k_neighbors", default=5, type=int)

    # Execution/modes
    ap.add_argument("--run_cluster", default="Serial", help='Serial | Local | BashSLURM | BashLSF | "<dask-cluster-name>"')
    ap.add_argument("--queue", default="defq")
    ap.add_argument("--reserved_memory", default=4, type=int)

    # Discovery-only flags
    ap.add_argument("--list-imputers", action="store_true", help="List dynamically discovered imputers and exit")
    ap.add_argument("--list-scalers", action="store_true", help="List dynamically discovered scalers and exit")
    add_run_command_args(ap)

    args = ap.parse_args()
    args = apply_saved_run_command(ap, args, "p2_impute_scale")
    run_command_args = snapshot_args(args)

    if args.list_imputers:
        imps = list_imputers()
        print("Available imputers:")
        for k, cls in sorted(imps.items()):
            print(f"  {k:15s} -> {cls.__module__}.{cls.__name__}")
        return

    if args.list_scalers:
        scs = list_scalers()
        print("Available scalers:")
        for k, cls in sorted(scs.items()):
            print(f"  {k:15s} -> {cls.__module__}.{cls.__name__}")
        return

    runner = P2Runner(
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        # overrides (None → take from metadata)
        scale_data=None if args.scale_data is None else _bool(args.scale_data, True),
        impute_data=None if args.impute_data is None else _bool(args.impute_data, True),
        multi_impute=None if args.multi_impute is None else _bool(args.multi_impute, False),
        overwrite_cv=None if args.overwrite_cv is None else _bool(args.overwrite_cv, True),
        outcome_label=args.outcome_label,
        outcome_type=args.outcome_type,
        instance_label=args.instance_label,
        random_state=args.random_state,
        # registries
        imputer_id=args.imputer_id,
        imputer_params=_maybe_json(args.imputer_params),
        scaler_id=args.scaler_id,
        scaler_params=_maybe_json(args.scaler_params),
        smote=None if args.smote is None else _bool(args.smote, False),
        smote_method=args.smote_method,
        smote_sampling_strategy=parse_sampling_strategy(args.smote_sampling_strategy),
        smote_k_neighbors=args.smote_k_neighbors,
        # execution
        run_cluster=args.run_cluster if args.run_cluster not in (None, "Serial", "False", "false") else False,
        queue=args.queue,
        reserved_memory=args.reserved_memory,
    )
    runner.run()
    save_run_command_from_args(args, "p2_impute_scale", run_command_args, runner=runner)


if __name__ == "__main__":
    # Inspect dynamic components
    # python -m streamline.p2_impute_scale.p2_cli --output_path ./out --experiment_name MyExp --list-imputers
    # python -m streamline.p2_impute_scale.p2_cli --output_path ./out --experiment_name MyExp --list-scalers

    # # Serial run (use defaults from metadata.pickle)
    # python -m streamline.p2_impute_scale.p2_cli \
    # --output_path ./test \
    # --experiment_name MyExp

    # # Force specific imputer & scaler (with params)
    # python -m streamline.p2_impute_scale.p2_cli \
    # --output_path ./out \
    # --experiment_name MyExp \
    # --imputer_id knn \
    # --imputer_params '{"n_neighbors": 7, "weights": "distance"}' \
    # --scaler_id minmax \
    # --scaler_params '{"feature_range":[0,1]}'

    # # Enable post-imputation/scaling SMOTE for train folds only
    # python -m streamline.p2_impute_scale.p2_cli \
    # --output_path ./out \
    # --experiment_name MyExp \
    # --smote 1 \
    # --smote_method auto

    main()
