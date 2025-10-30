import argparse
from streamline.p6_modeling.p6_runner import P6Runner

def main():
    ap = argparse.ArgumentParser("STREAMLINE Phase 6 (Modeling) CLI",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--experiment_name", required=True)

    ap.add_argument("--outcome_label", default="Class")
    ap.add_argument("--model_type", default="BinaryClassification",
                    help="BinaryClassification | MulticlassClassification | Regression")
    ap.add_argument("--instance_label", default=None)
    ap.add_argument("--n_splits", type=int, required=True)
    ap.add_argument("--models", default=None,
                    help="CSV of model ids (small_name or underscored model_name). Omit to auto-discover.")

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
