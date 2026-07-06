import argparse
from streamline.p8_summary_statistics.p8_runner import P8Runner
from streamline.utils.run_commands import (
    add_run_command_args,
    apply_saved_run_command,
    require_args,
    save_run_command_from_args,
    snapshot_args,
)


def main():
    ap = argparse.ArgumentParser("STREAMLINE Phase 8 (Statistics) CLI",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--experiment_name", required=True)

    ap.add_argument("--outcome_label", default="Class")
    ap.add_argument("--outcome_type", default=None,
                    help="Binary | Multiclass | Continuous; "
                         "if omitted, loaded from experiment metadata.pickle")
    ap.add_argument("--instance_label", default=None)
    ap.add_argument("--n_splits", type=int, default=None)

    ap.add_argument("--scoring_metric", default="balanced_accuracy")
    ap.add_argument("--metric_weight", default="balanced_accuracy",
                    help="Metric used to weight composite FI (e.g. balanced_accuracy, explained_variance)")
    ap.add_argument("--top_features", type=int, default=40)
    ap.add_argument("--sig_cutoff", type=float, default=0.05)
    ap.add_argument("--scale_data", type=int, default=1)
    ap.add_argument("--exclude_plots", default="",
                    help="Comma-separated subset of: plot_ROC,plot_PRC,plot_FI_box,plot_metric_boxplots")
    ap.add_argument("--show_plots", type=int, default=0)
    ap.add_argument("--include_ensembles", type=int, default=1,
                    help="1 to summarize ensembles from Phase 7 if present")
    ap.add_argument("--multiclass_average", default="micro",
                    help="Averaging method for multiclass metrics: micro | macro")

    # execution
    ap.add_argument("--run_cluster", default="Serial",
                    help="Serial | Local | Parallel | BashSLURM | BashLSF | <dask-cluster-name>")
    ap.add_argument("--queue", default="defq")
    ap.add_argument("--reserved_memory", type=int, default=4)
    add_run_command_args(ap)

    args = ap.parse_args()
    args = apply_saved_run_command(ap, args, "p8_summary_statistics")
    require_args(ap, args, ["n_splits"])
    run_command_args = snapshot_args(args)

    runner = P8Runner(
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        outcome_label=args.outcome_label,
        outcome_type=args.outcome_type,
        instance_label=args.instance_label,
        n_splits=args.n_splits,
        scoring_metric=args.scoring_metric,
        metric_weight=args.metric_weight,
        top_features=args.top_features,
        sig_cutoff=args.sig_cutoff,
        scale_data=bool(args.scale_data),
        exclude_plots=args.exclude_plots,
        show_plots=bool(args.show_plots),
        include_ensembles=bool(args.include_ensembles),
        multiclass_average=args.multiclass_average,
        run_cluster=args.run_cluster,
        queue=args.queue,
        reserved_memory=args.reserved_memory,
    )
    runner.run()
    save_run_command_from_args(args, "p8_summary_statistics", run_command_args, runner=runner)


if __name__ == "__main__":
    main()
