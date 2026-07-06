from __future__ import annotations

import argparse
import logging

from streamline.p11_reporting.p11_runner import P11Runner
from streamline.utils.run_commands import (
    add_run_command_args,
    apply_saved_run_command,
    save_run_command_from_args,
    snapshot_args,
)

logging.basicConfig(level=logging.INFO)


def _none_if_empty(val: str | None) -> str | None:
    if val is None:
        return None
    text = str(val).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return text


def main():
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

    ap.add_argument(
        "--run_cluster",
        default="Serial",
        help="Serial | Local | BashSLURM | BashLSF | <dask-cluster-name>",
    )
    ap.add_argument("--queue", default="defq")
    ap.add_argument("--reserved_memory", type=int, default=4)
    add_run_command_args(ap)

    args = ap.parse_args()
    args = apply_saved_run_command(ap, args, "p11_reporting")
    run_command_args = snapshot_args(args)

    if not args.experiment_path and not (args.output_path and args.experiment_name):
        ap.error("Provide --experiment_path OR both --output_path and --experiment_name")

    job = P11Runner(
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
        run_cluster=args.run_cluster,
        queue=args.queue,
        reserved_memory=args.reserved_memory,
    )
    job.run()
    save_run_command_from_args(args, "p11_reporting", run_command_args, runner=job)


if __name__ == "__main__":
    main()
