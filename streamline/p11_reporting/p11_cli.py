from __future__ import annotations

import argparse
import logging

from streamline.p11_reporting.reporting import ReportPhaseJob

logging.basicConfig(level=logging.INFO)


def main():
    ap = argparse.ArgumentParser(
        "STREAMLINE Phase 10 (Reporting)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--experiment_name", required=True)
    ap.add_argument("--outcome_label", default="Class")
    ap.add_argument("--outcome_type", default="Binary")
    ap.add_argument("--instance_label", default=None)
    ap.add_argument("--make_pdf", type=int, default=1,
                    help="1 to generate PDF via WeasyPrint if available; 0 to skip.")
    ap.add_argument(
        "--run_cluster",
        default="Serial",
        help="Serial | Local | BashSLURM | BashLSF | <dask-cluster-name>",
    )
    ap.add_argument("--queue", default="defq")
    ap.add_argument("--reserved_memory", type=int, default=4)
    args = ap.parse_args()

    job = ReportPhaseJob(
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        outcome_label=args.outcome_label,
        outcome_type=args.outcome_type,
        instance_label=args.instance_label,
        make_pdf=bool(args.make_pdf),
        run_cluster=args.run_cluster,
        queue=args.queue,
        reserved_memory=args.reserved_memory,
    )
    job.run()


if __name__ == "__main__":
    main()
