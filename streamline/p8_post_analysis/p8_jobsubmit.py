import argparse
from streamline.p8_post_analysis.statistics import StatisticsPhaseJob


def _b(x):
    if x is None:
        return False
    return str(x).strip().lower() in ("1", "true", "t", "yes", "y")


def main():
    ap = argparse.ArgumentParser("P8 Statistics jobsubmit (single dataset)")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--outcome_label", default="Class")
    ap.add_argument("--outcome_type", default="Binary")  # Binary | Multiclass | Continuous
    ap.add_argument("--instance_label", default=None)
    ap.add_argument("--n_splits", type=int, required=True)

    ap.add_argument("--scoring_metric", default="balanced_accuracy")
    ap.add_argument("--metric_weight", default="balanced_accuracy")
    ap.add_argument("--top_features", type=int, default=40)
    ap.add_argument("--sig_cutoff", type=float, default=0.05)
    ap.add_argument("--scale_data", default="1")
    ap.add_argument("--exclude_plots", default="")
    ap.add_argument("--show_plots", default="0")
    ap.add_argument("--include_ensembles", default="1")

    args = ap.parse_args()

    exclude_plots = [
        x.strip() for x in args.exclude_plots.split(",") if x.strip()
    ] if args.exclude_plots else []

    StatisticsPhaseJob(
        dataset_dir=args.dataset_dir,
        outcome_label=args.outcome_label,
        outcome_type=args.outcome_type,
        instance_label=(args.instance_label if args.instance_label else None),
        scoring_metric=args.scoring_metric,
        cv_partitions=int(args.n_splits),
        top_features=int(args.top_features),
        sig_cutoff=float(args.sig_cutoff),
        metric_weight=args.metric_weight,
        scale_data=_b(args.scale_data),
        exclude_plots=exclude_plots,
        show_plots=_b(args.show_plots),
        include_ensembles=_b(args.include_ensembles),
    ).run()


if __name__ == "__main__":
    main()
