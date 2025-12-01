# streamline/p9_compare_datasets/p9_jobsubmit.py
from __future__ import annotations

import argparse

from streamline.p9_compare_datasets.p9_runner import P9Runner


def main():
    ap = argparse.ArgumentParser(
        "STREAMLINE Phase 9 (Dataset Comparisons) jobsubmit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--experiment_name", required=True)
    ap.add_argument("--outcome_label", default="Class")
    ap.add_argument(
        "--outcome_type",
        choices=["Binary", "Multiclass", "Continuous"],
        default="Binary",
    )
    ap.add_argument("--instance_label", default=None)
    ap.add_argument("--sig_cutoff", type=float, default=0.05)
    ap.add_argument("--show_plots", type=int, default=0)

    args = ap.parse_args()

    # On the compute node we just run serially; scheduling is handled by SLURM/LSF.
    P9Runner(
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        outcome_label=args.outcome_label,
        outcome_type=args.outcome_type,
        instance_label=args.instance_label,
        sig_cutoff=args.sig_cutoff,
        show_plots=bool(args.show_plots),
        run_cluster="Serial",
    ).run()


if __name__ == "__main__":
    main()
