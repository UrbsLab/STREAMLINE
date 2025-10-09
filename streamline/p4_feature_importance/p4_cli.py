# streamline/phases/p4_feature_selection/cli.py
import argparse, json
from streamline.p4_feature_importance.p4_runner import P4Runner
from streamline.p4_feature_importance.utils.fi_loader import list_importances

def _maybe_json(s: str):
    try: return json.loads(s or "{}")
    except Exception: return {}

def _maybe_bool(s, default=None):
    if s is None: return default
    v = str(s).lower()
    if v in ("1","true","t","yes","y"): return True
    if v in ("0","false","f","no","n"): return False
    return default

def main():
    ap = argparse.ArgumentParser("STREAMLINE Phase 4 (Feature Selection) CLI",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--experiment_name", required=True)

    ap.add_argument("--importance_id", default=None, help="mutualinformation | multisurf | multisurfstar")
    ap.add_argument("--importance_params", default=None, help="JSON dict")
    ap.add_argument("--top_k", default=None, type=int)
    ap.add_argument("--threshold", default=None, type=float)
    ap.add_argument("--keep_original_features", default=None)
    ap.add_argument("--overwrite_cv", default=None)
    ap.add_argument("--outcome_label", default=None)
    ap.add_argument("--outcome_type", default=None)
    ap.add_argument("--instance_label", default=None)
    ap.add_argument("--random_state", default=None, type=int)

    ap.add_argument("--run_cluster", default="Serial")
    ap.add_argument("--queue", default="defq")
    ap.add_argument("--reserved_memory", default=4, type=int)

    ap.add_argument("--list-importances", action="store_true")

    args = ap.parse_args()

    if args.list_importances:
        sels = list_importances()
        print("Available importances:")
        for k, cls in sorted(sels.items()):
            print(f"  {k:15s} -> {cls.__module__}.{cls.__name__}")
        return

    runner = P4Runner(
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        importance_id=args.importance_id,
        importance_params=_maybe_json(args.importance_params) if args.importance_params else None,
        top_k=args.top_k,
        threshold=args.threshold,
        keep_original_features=_maybe_bool(args.keep_original_features),
        overwrite_cv=_maybe_bool(args.overwrite_cv),
        outcome_label=args.outcome_label,
        outcome_type=args.outcome_type,
        instance_label=args.instance_label,
        random_state=args.random_state,
        run_cluster=args.run_cluster if args.run_cluster not in (None,"Serial","False","false") else False,
        queue=args.queue,
        reserved_memory=args.reserved_memory,
    )
    runner.run()

if __name__ == "__main__":
    # # MultiSURF with TURF, sample 2000 rows for fitting, write
    # # ./<exp>/<dataset>/feature_selection/multisurf/multisurf_scores_cv_1.csv
    # python -m streamline.phases.p4_feature_selection.cli \
    # --output_path ./out --experiment_name MyExp \
    # --selector_id multisurf \
    # --selector_params '{"use_turf": true, "turf_pct": 0.5, "n_jobs": 4}' \
    # --top_k 100 \
    # --instance_subset 2000
    main()
