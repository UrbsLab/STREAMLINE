from __future__ import annotations

import argparse

from streamline.p10_replication.p10_runner import P10Runner
from streamline.utils.run_commands import (
    add_run_command_args,
    apply_saved_run_command,
    require_args,
    save_run_command_from_args,
    snapshot_args,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        "STREAMLINE Phase 10 (Replication / External Validation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--rep_data_path",
        default=None,
        help="Directory containing replication datasets (.csv/.tsv/.txt)",
    )
    parser.add_argument(
        "--dataset_for_rep",
        default=None,
        help="Path to the original training dataset file whose trained pipeline is reused",
    )
    parser.add_argument("--output_path", required=True, help="STREAMLINE output root")
    parser.add_argument("--experiment_name", required=True, help="Experiment name")

    parser.add_argument("--outcome_label", default=None, help="Override outcome label")
    parser.add_argument("--instance_label", default=None, help="Override instance label")
    parser.add_argument("--match_label", default=None, help="Override match label")

    parser.add_argument(
        "--exclude_plots",
        default="",
        help=(
            "Comma-separated list of plots to exclude "
            "(plot_ROC,plot_PRC,plot_metric_boxplots,plot_FI_box,feature_correlations)"
        ),
    )

    parser.add_argument(
        "--run_cluster",
        default="Serial",
        help="Serial | Local | BashSLURM | BashLSF | <dask-cluster-name>",
    )
    parser.add_argument("--queue", default="defq")
    parser.add_argument("--reserved_memory", type=int, default=4)
    parser.add_argument("--show_plots", type=int, default=0)
    add_run_command_args(parser)

    args = parser.parse_args()
    args = apply_saved_run_command(parser, args, "p10_replication")
    require_args(parser, args, ["rep_data_path", "dataset_for_rep"])
    run_command_args = snapshot_args(args)

    exclude_plots = [x.strip() for x in args.exclude_plots.split(",") if x.strip()]

    runner = P10Runner(
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
    save_run_command_from_args(args, "p10_replication", run_command_args)


if __name__ == "__main__":
    main()
