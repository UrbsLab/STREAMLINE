from __future__ import annotations

import argparse

from streamline.p11_reporting.p11_runner import P11Runner


def _none_if_empty(val: str | None) -> str | None:
    """
    Normalize 'empty-ish' CLI values back to None.
    """
    if val is None:
        return None
    val_str = str(val).strip()
    if val_str == "" or val_str.lower() in {"none", "null"}:
        return None
    return val_str


def main():
    """
    Entry point for Phase 10 reporting when launched on a compute node
    via a SLURM/LSF bash wrapper.

    This script intentionally forces run_cluster="Serial"; the parallelism is
    handled by the scheduler that launched this process.
    """
    ap = argparse.ArgumentParser(
        "STREAMLINE Phase 10 (Reporting)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--output_path", required=True, help="Top-level output directory")
    ap.add_argument("--experiment_name", required=True, help="Experiment folder name")

    ap.add_argument("--outcome_label", default="Class")
    ap.add_argument(
        "--outcome_type",
        default="Binary",
        choices=["Binary", "Multiclass", "Continuous"],
        help="Outcome type to drive metric/plot selection in the report",
    )
    ap.add_argument(
        "--instance_label",
        default=None,
        help="Optional instance ID column used in earlier phases",
    )

    ap.add_argument(
        "--report_name",
        default="STREAMLINE_Report",
        help="Base name for the generated report (HTML/PDF)",
    )

    ap.add_argument(
        "--sig_cutoff",
        type=float,
        default=0.05,
        help="Significance cutoff used when annotating stats in the report",
    )

    ap.add_argument(
        "--show_plots",
        type=int,
        default=0,
        help="1 = keep Streamlit UI open / debug locally; 0 = non-interactive export only",
    )
    
    ap.add_argument(
        "--run_cluster",
        default="Serial",
        help="Serial | Local | BashSLURM | BashLSF | <dask-cluster-name>",
    )
    ap.add_argument("--queue", default="defq")
    ap.add_argument("--reserved_memory", type=int, default=4)

    args = ap.parse_args()

    P11Runner(
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        outcome_label=args.outcome_label,
        outcome_type=args.outcome_type,
        instance_label=args.instance_label,
        make_pdf=bool(args.make_pdf),
        run_cluster=args.run_cluster,
        queue=args.queue,
        reserved_memory=args.reserved_memory,
    ).run()


if __name__ == "__main__":
    main()
