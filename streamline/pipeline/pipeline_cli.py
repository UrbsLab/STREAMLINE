from __future__ import annotations

import argparse
import logging

from streamline.pipeline.pipeline_runner import PipelineRunner, normalize_phase_list


def main() -> None:
    parser = argparse.ArgumentParser(
        "STREAMLINE config-driven pipeline runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", required=True, help="Path to a .cfg STREAMLINE config file")
    parser.add_argument("--dry_run", action="store_true", help="Print resolved phase calls without running them")
    parser.add_argument("--start_at", default=None, help="Start at a phase alias, e.g. p4 or p4_feature_importance")
    parser.add_argument("--stop_after", default=None, help="Stop after a phase alias, e.g. p8")
    parser.add_argument("--only", default=None, help="Comma-separated phase aliases to run")
    parser.add_argument("--skip", default=None, help="Comma-separated phase aliases to skip")
    parser.add_argument("--log_level", default="INFO", help="Python logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    PipelineRunner(
        config_path=args.config,
        dry_run=args.dry_run,
        start_at=args.start_at,
        stop_after=args.stop_after,
        only=normalize_phase_list(args.only) if args.only else None,
        skip=normalize_phase_list(args.skip) if args.skip else None,
    ).run()


if __name__ == "__main__":
    main()
