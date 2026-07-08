# streamline/p4_feature_importance/cli.py
import argparse, json
from streamline.p4_feature_importance.p4_runner import P4Runner
from streamline.p4_feature_importance.utils.fi_loader import list_importances
from streamline.utils.run_commands import (
    add_run_command_args,
    apply_saved_run_command,
    save_run_command_from_args,
    snapshot_args,
)

def _parse_models_csv(s: str):
    if not s: return []
    return [m.strip() for m in s.split(",") if m.strip()]

def _maybe_json(s: str):
    import json
    try: 
        v = json.loads(s or "{}")
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}

def _maybe_bool(s, default=None):
    if s is None: return default
    v = str(s).lower()
    if v in ("1","true","t","yes","y"): return True
    if v in ("0","false","f","no","n"): return False
    return default

def main():
    ap = argparse.ArgumentParser("STREAMLINE Phase 4 (Feature Importance) CLI",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--experiment_name", required=True)

    # NOW: comma-separated only (e.g., "mutualinformation,multiswrfdb,multiswrfdbstar")
    ap.add_argument("--models", required=False, help='Comma-separated: e.g. "mutualinformation,multiswrfdb,multiswrfdbstar"')
    # keep JSON dict for params (per-model)
    ap.add_argument(
        "--models_params",
        default=None,
        help='JSON dict: {"mutualinformation": {...}, "multiswrfdb": {...}}; ReBATE categorical_features is injected by STREAMLINE.',
    )

    ap.add_argument("--top_k", default=None, type=int)
    ap.add_argument("--threshold", default=None, type=float)
    ap.add_argument("--keep_original_features", default=None)
    ap.add_argument("--overwrite_cv", default=None)
    ap.add_argument("--outcome_label", default=None)
    ap.add_argument("--outcome_type", default=None)
    ap.add_argument("--instance_label", default=None)
    ap.add_argument("--random_state", default=None, type=int)
    ap.add_argument("--instance_subset", default=None, type=int)

    ap.add_argument("--run_cluster", default="Serial",
                    help="Serial | Local | Parallel | BashSLURM | BashLSF | <dask-cluster-name>")
    ap.add_argument("--queue", default="defq")
    ap.add_argument("--reserved_memory", default=4, type=int)

    ap.add_argument("--list-models", action="store_true")
    add_run_command_args(ap)

    args = ap.parse_args()
    args = apply_saved_run_command(ap, args, "p4_feature_importance")
    run_command_args = snapshot_args(args)

    if args.list_models:
        models = list_importances()
        print("Available feature-importance models:")
        for k, cls in sorted(models.items()):
            model_name = getattr(cls, "model_name", k)
            small = getattr(cls, "small_name", k)
            print(f"  {k:16s} -> {cls.__module__}.{cls.__name__}  [{model_name} | {small}]")
        return

    runner = P4Runner(
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        models=_parse_models_csv(args.models) if args.models else None,
        models_params=_maybe_json(args.models_params) if args.models_params else None,
        top_k=args.top_k,
        threshold=args.threshold,
        keep_original_features=_maybe_bool(args.keep_original_features),
        overwrite_cv=_maybe_bool(args.overwrite_cv),
        outcome_label=args.outcome_label,
        outcome_type=args.outcome_type,
        instance_label=args.instance_label,
        random_state=args.random_state,
        instance_subset=args.instance_subset,
        run_cluster=args.run_cluster if args.run_cluster not in (None,"Serial","False","false") else False,
        queue=args.queue,
        reserved_memory=args.reserved_memory,
    )
    runner.run()
    save_run_command_from_args(args, "p4_feature_importance", run_command_args, runner=runner)

if __name__ == "__main__":
    # # run three models (comma-separated), pass params to one ReBATE method via JSON
    # python -m streamline.p4_feature_importance.p4_cli \
    # --output_path ./test --experiment_name MyExp \
    # --models "mutualinformation,multiswrfdb,multiswrfdbstar" \
    # --models_params '{"multiswrfdb":{"use_turf": true, "turf_pct": 0.5, "n_jobs": 1}}' \
    # --top_k 100 --instance_subset 2000  # optional sampling limit
    main()
