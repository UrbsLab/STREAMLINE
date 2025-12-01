# streamline/p10_replication/p10_cli.py

from __future__ import annotations
import argparse

from streamline.p10_replication.p10_runner import P10ReplicationRunner


def main():
    ap = argparse.ArgumentParser(
        "STREAMLINE Phase 10 (Replication / External Validation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--rep_data_path", required=True,
                    help="Directory containing replication / holdout datasets (.csv/.tsv/.txt)")
    ap.add_argument("--dataset_for_rep", required=True,
                    help="Path to the *training* dataset whose models will be applied")
    ap.add_argument("--output_path", required=True,
                    help="STREAMLINE output root (same as earlier phases)")
    ap.add_argument("--experiment_name", required=True,
                    help="Experiment name (same as earlier phases)")

    ap.add_argument("--outcome_label", default=None,
                    help="Override outcome label (otherwise taken from metadata)")
    ap.add_argument("--instance_label", default=None,
                    help="Override instance label (otherwise taken from metadata)")
    ap.add_argument("--match_label", default=None,
                    help="Match label if used in training")

    ap.add_argument(
        "--exclude_plots",
        default=None,
        help=(
            "Comma-separated list of plots to exclude "
            "(plot_ROC,plot_PRC,plot_metric_boxplots,feature_correlations)"
        ),
    )

    ap.add_argument(
        "--run_cluster",
        default="Serial",
        help="Serial | Local | BashSLURM | BashLSF | <dask-cluster-name>",
    )
    ap.add_argument("--queue", default="defq")
    ap.add_argument("--reserved_memory", type=int, default=4)
    ap.add_argument("--show_plots", type=int, default=0)

    args = ap.parse_args()

    exclude_plots = (
        [s for s in args.exclude_plots.split(",") if s]
        if args.exclude_plots
        else None
    )

    runner = P10ReplicationRunner(
        rep_data_path=args.rep_data_path,
        dataset_for_rep=args.dataset_for_rep,
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        outcome_label=args.outcome_label,
        instance_label=args.instance_label,
        match_label=args.match_label,
        exclude_plots=exclude_plots,
        run_cluster=args.run_cluster,
        queue=args.queue,
        reserved_memory=args.reserved_memory,
        show_plots=bool(args.show_plots),
    )
    runner.run()


if __name__ == "__main__":
    main()
