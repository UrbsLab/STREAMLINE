import argparse
from streamline.p6_modeling.p6_runner import P6Runner
from streamline.p6_modeling.utils.loader import list_models, list_all_models

def _print_models(entries, title: str):
    print(f"\n{title}")
    if not entries:
        print("  (none found)")
        return
    # neat, one-per-line: <type>  <id>  (<alt_id>)  -  <name>  [module]
    for e in entries:
        alt = f" ({e['alt_id']})" if e.get("alt_id") else ""
        mod = f"  [{e['module']}]" if e.get("module") else ""
        print(f"  {e['model_type']:<24} {e['small_name']:<12} {e['model_name']}")
    print("")

def main():
    ap = argparse.ArgumentParser("STREAMLINE Phase 6 (Modeling) CLI",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--experiment_name", required=True)

    ap.add_argument("--outcome_label", default="Class")
    ap.add_argument("--model_type", default="Binary",
                    help="Binary | Multiclass | Regression")
    ap.add_argument("--instance_label", default=None)
    ap.add_argument("--n_splits", type=int, required=True)
    ap.add_argument("--models", default=None,
                    help="CSV of model ids (small_name or underscored model_name). Omit to auto-discover.")

    # NEW: listing modes
    ap.add_argument("--list_models", action="store_true",
                    help="List models for --model_type and exit.")
    ap.add_argument("--list_models_all", action="store_true",
                    help="List models for all types and exit.")

    # calibration (handled in BaseModel.fit)
    ap.add_argument("--calibrate", type=int, default=0, help="1 to enable probability calibration")
    ap.add_argument("--calibrate_method", default="sigmoid", help="sigmoid | isotonic")
    ap.add_argument("--calibrate_cv", type=int, default=5)

    # ModelJob controls
    ap.add_argument("--scoring_metric", default="balanced_accuracy")
    ap.add_argument("--metric_direction", default="maximize")
    ap.add_argument("--n_trials", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--training_subsample", type=int, default=0)
    ap.add_argument("--uniform_fi", type=int, default=0)
    ap.add_argument("--save_plot", type=int, default=0)
    ap.add_argument("--random_state", default=None)

    # execution
    ap.add_argument("--run_cluster", default="Serial",
                    help="Serial | Local | BashSLURM | BashLSF | <dask-cluster-name>")
    ap.add_argument("--queue", default="defq")
    ap.add_argument("--reserved_memory", type=int, default=4)

    args = ap.parse_args()

    # ---- NEW: handle listing and exit ----
    if args.list_models_all:
        entries = list_all_models()
        _print_models(entries, title="Available models (ALL types):")
        return

    if args.list_models:
        entries = list_models(args.model_type)
        _print_models(entries, title=f"Available models ({args.model_type}):")
        return

    # ---- normal run ----
    P6Runner(
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        outcome_label=args.outcome_label,
        model_type=args.model_type,
        instance_label=args.instance_label,
        n_splits=args.n_splits,
        models=args.models,

        calibrate=bool(args.calibrate),
        calibrate_method=args.calibrate_method,
        calibrate_cv=args.calibrate_cv,

        scoring_metric=args.scoring_metric,
        metric_direction=args.metric_direction,
        n_trials=args.n_trials,
        timeout=args.timeout,
        training_subsample=args.training_subsample,
        uniform_fi=bool(args.uniform_fi),
        save_plot=bool(args.save_plot),
        random_state=(int(args.random_state) if (args.random_state not in (None,"","None")) else None),

        run_cluster=args.run_cluster,
        queue=args.queue,
        reserved_memory=args.reserved_memory,
    ).run()

if __name__ == "__main__":
    main()
