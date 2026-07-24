#!/usr/bin/env python
"""
STREAMLINE — Phase 5 (Feature Selection) job-submit
Runs a single dataset directory with the Phase 5 FeatureSelectionJob.
Intended to be launched by bash submit scripts (SLURM/LSF) or directly.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from .feature_selection import FeatureSelection
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from streamline.p5_feature_selection.feature_selection import FeatureSelection

def _to_bool(v, default=False):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    return default


def _to_dict(s):
    if s is None:
        return {}
    if isinstance(s, dict):
        return s
    try:
        d = json.loads(s)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def build_parser():
    ap = argparse.ArgumentParser("P5 Feature Selection jobsubmit (single dataset)")
    # Required
    ap.add_argument("--dataset_dir", required=True,
                    help="Path to a single dataset folder: <output>/<experiment>/<dataset>")
    ap.add_argument("--n_splits", required=True, type=int,
                    help="Number of CV splits")
    ap.add_argument("--algorithms", default="auto",
                    help='Comma-separated ids/names or "auto" to discover from feature_importance/*')

    # Labels
    ap.add_argument("--outcome_label", default="Class")
    ap.add_argument("--instance_label", default=None)

    # Selection controls
    ap.add_argument("--max_features_to_keep", default=2000, type=int)
    ap.add_argument("--filter_poor_features", default="1",
                    help="Keep only features with score > 0 before capping (1/0, true/false)")
    ap.add_argument("--overwrite_cv", default="0",
                    help="Overwrite CV CSVs instead of renaming to *_CVPre_* (1/0, true/false)")

    # Strategy (registry)
    ap.add_argument("--selector_id", default="default",
                    help='Strategy id from registry (default: "default")')
    ap.add_argument("--selector_params", default=None,
                    help="JSON dict of extra params for the strategy (e.g., {'export_scores': true})")

    # Plotting/summary (forwarded to default strategy)
    ap.add_argument("--export_scores", default="1",
                    help="Write TopAverageScores.png per algorithm (1/0, true/false)")
    ap.add_argument("--top_features", default=20, type=int,
                    help="Top-N features to visualize in the median score plot")
    ap.add_argument("--show_plots", default="0",
                    help="Render plots to screen (usually off on HPC) (1/0, true/false)")
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    FeatureSelection(
        dataset_dir=args.dataset_dir,
        n_splits=int(args.n_splits),
        algorithms=args.algorithms,  # CSV string OK
        outcome_label=args.outcome_label,
        instance_label=(args.instance_label if args.instance_label else None),
        max_features_to_keep=int(args.max_features_to_keep),
        filter_poor_features=_to_bool(args.filter_poor_features, True),
        overwrite_cv=_to_bool(args.overwrite_cv, False),
        selector_id=(args.selector_id or "default"),
        selector_params=_to_dict(args.selector_params),
        export_scores=_to_bool(args.export_scores, True),
        top_features=int(args.top_features),
        show_plots=_to_bool(args.show_plots, False),
    ).run()


if __name__ == "__main__":
    main()
