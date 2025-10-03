# streamline/phases/p2_impute_scale/cli.py
import argparse
import json
from typing import Optional, Dict, Any

from .runner import P2Runner
from .loader import list_imputers
from .scale_loader import list_scalers


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
    ap.add_argument("--instance_label", default=None)
    ap.add_argument("--random_state", default=None, type=int)

    # Imputer registry
    ap.add_argument("--imputer_id", default=None)
    ap.add_argument("--imputer_params", default="{}")

    # Scaler registry
    ap.add_argument("--scaler_id", default=None)
    ap.add_argument("--scaler_params", default="{}")

    # Execution/modes
    ap.add_argument("--run_cluster", default="Serial", help='Serial | Local | BashSLURM | BashLSF | "<dask-cluster-name>"')
    ap.add_argument("--queue", default="defq")
    ap.add_argument("--reserved_memory", default=4, type=int)

    # Discovery-only flags
    ap.add_argument("--list-imputers", action="store_true", help="List dynamically discovered imputers and exit")
    ap.add_argument("--list-scalers", action="store_true", help="List dynamically discovered scalers and exit")

    args = ap.parse_args()

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
        instance_label=args.instance_label,
        random_state=args.random_state,
        # registries
        imputer_id=args.imputer_id,
        imputer_params=_maybe_json(args.imputer_params),
        scaler_id=args.scaler_id,
        scaler_params=_maybe_json(args.scaler_params),
        # execution
        run_cluster=args.run_cluster if args.run_cluster not in (None, "Serial", "False", "false") else False,
        queue=args.queue,
        reserved_memory=args.reserved_memory,
    )
    runner.run()


if __name__ == "__main__":
    # Inspect dynamic components
    # python -m streamline.phases.p2_impute_scale.cli --output_path ./out --experiment_name MyExp --list-imputers
    # python -m streamline.phases.p2_impute_scale.cli --output_path ./out --experiment_name MyExp --list-scalers

    # # Serial run (use defaults from metadata.pickle)
    # python -m streamline.phases.p2_impute_scale.cli \
    # --output_path ./out \
    # --experiment_name MyExp

    # # Force specific imputer & scaler (with params)
    # python -m streamline.phases.p2_impute_scale.cli \
    # --output_path ./out \
    # --experiment_name MyExp \
    # --imputer_id knn \
    # --imputer_params '{"n_neighbors": 7, "weights": "distance"}' \
    # --scaler_id minmax \
    # --scaler_params '{"feature_range":[0,1]}'

    main()
