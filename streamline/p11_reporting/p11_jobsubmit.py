from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .p11_runner import P11Runner
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from streamline.p11_reporting.p11_runner import P11Runner

def _none_if_empty(val: str | None) -> str | None:
    if val is None:
        return None
    val_str = str(val).strip()
    if val_str == "" or val_str.lower() in {"none", "null"}:
        return None
    return val_str


def main():
    """
    Entry point for Phase 11 reporting when launched on a compute node via
    a SLURM/LSF bash wrapper.

    This script intentionally forces run_cluster="Serial"; scheduler-level
    parallelism is handled outside this process.
    """
    ap = argparse.ArgumentParser(
        "STREAMLINE Phase 11 (Reporting)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--experiment_path", default=None, help="Path to experiment output directory")
    ap.add_argument("--output_path", default=None, help="Parent output directory")
    ap.add_argument("--experiment_name", default=None, help="Experiment folder name")
    ap.add_argument(
        "--reporting_dir",
        default=None,
        help="Optional output directory for report artifacts (report_data.json, experiment-named PDF, figures)",
    )
    ap.add_argument(
        "--report_mode",
        default="standard",
        choices=["standard", "replication"],
        help="Reporting scope: standard (training datasets) or replication (datasets under replication folders)",
    )

    ap.add_argument("--outcome_label", default=None)
    ap.add_argument("--outcome_type", default=None)
    ap.add_argument("--instance_label", default=None)

    ap.add_argument("--make_pdf", type=int, default=1, help="1 = export PDF; 0 = skip PDF")
    ap.add_argument("--enable_plots", type=int, default=1, help="1 = generate missing plots; 0 = disable plot generation")
    ap.add_argument(
        "--reuse_existing_figures",
        type=int,
        default=1,
        help="1 = reuse existing report PNGs when present; 0 = regenerate",
    )

    ap.add_argument("--queue", default="defq")
    ap.add_argument("--reserved_memory", type=int, default=4)

    args = ap.parse_args()

    if not args.experiment_path and not (args.output_path and args.experiment_name):
        ap.error("Provide --experiment_path OR both --output_path and --experiment_name")

    P11Runner(
        output_path=_none_if_empty(args.output_path),
        experiment_name=_none_if_empty(args.experiment_name),
        experiment_path=_none_if_empty(args.experiment_path),
        reporting_dir=_none_if_empty(args.reporting_dir),
        report_mode=args.report_mode,
        outcome_label=_none_if_empty(args.outcome_label),
        outcome_type=_none_if_empty(args.outcome_type),
        instance_label=_none_if_empty(args.instance_label),
        make_pdf=bool(args.make_pdf),
        enable_plots=bool(args.enable_plots),
        reuse_existing_figures=bool(args.reuse_existing_figures),
        run_cluster="Serial",
        queue=args.queue,
        reserved_memory=args.reserved_memory,
    ).run()


if __name__ == "__main__":
    main()
