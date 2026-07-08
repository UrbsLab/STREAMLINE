# streamline/p9_compare_datasets/p9_cli.py
from __future__ import annotations

import argparse

from streamline.p9_compare_datasets.p9_runner import P9Runner
from streamline.utils.run_commands import (
    add_run_command_args,
    apply_saved_run_command,
    save_run_command_from_args,
    snapshot_args,
)


def main():
    ap = argparse.ArgumentParser(
        "STREAMLINE Phase 9 (Dataset Comparisons)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--experiment_name", required=True)
    ap.add_argument(
        "--outcome_label",
        default="Class",
        help="Outcome column name.",
    )
    ap.add_argument(
        "--outcome_type",
        choices=["Binary", "Multiclass", "Continuous"],
        default="Binary",
        help="Outcome type (affects metric list / plots).",
    )
    ap.add_argument(
        "--instance_label",
        default=None,
        help="Optional instance ID column name.",
    )
    ap.add_argument(
        "--sig_cutoff",
        type=float,
        default=0.05,
        help="Significance cutoff for non-parametric tests.",
    )
    ap.add_argument(
        "--show_plots",
        type=int,
        default=0,
        help="1 to show plots interactively, 0 to only save to disk.",
    )
    ap.add_argument(
        "--run_cluster",
        default="Serial",
        help="Serial | Local | Parallel | BashSLURM | BashLSF | <dask-cluster-name>",
    )
    ap.add_argument("--queue", default="defq")
    ap.add_argument("--reserved_memory", type=int, default=4)
    add_run_command_args(ap)

    args = ap.parse_args()
    args = apply_saved_run_command(ap, args, "p9_compare_datasets")
    run_command_args = snapshot_args(args)

    runner = P9Runner(
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        outcome_label=args.outcome_label,
        outcome_type=args.outcome_type,
        instance_label=args.instance_label,
        sig_cutoff=args.sig_cutoff,
        show_plots=bool(args.show_plots),
        run_cluster=args.run_cluster,
        queue=args.queue,
        reserved_memory=args.reserved_memory,
    )
    runner.run()
    save_run_command_from_args(args, "p9_compare_datasets", run_command_args, runner=runner)


if __name__ == "__main__":
    main()
