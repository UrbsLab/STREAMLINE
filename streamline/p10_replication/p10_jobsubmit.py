from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

try:
    from .replication import ReplicationJob
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from streamline.p10_replication.replication import ReplicationJob

def main() -> None:
    parser = argparse.ArgumentParser("STREAMLINE Phase 10 Replication JobSubmit")
    parser.add_argument("--dataset_filename", required=True)
    parser.add_argument("--dataset_for_rep", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--outcome_label", default=None)
    parser.add_argument("--instance_label", default=None)
    parser.add_argument("--match_label", default=None)
    parser.add_argument("--exclude_plots", default="")
    parser.add_argument("--show_plots", type=int, default=0)
    args = parser.parse_args()

    exp_root = Path(args.output_path) / args.experiment_name
    with (exp_root / "metadata.pickle").open("rb") as f:
        metadata = pickle.load(f)

    outcome_label = args.outcome_label or metadata["Outcome Label"]
    instance_label = args.instance_label or metadata.get("Instance Label")
    outcome_type = metadata["Outcome Type"]

    train_name = Path(args.dataset_for_rep).stem
    train_dataset_root = exp_root / train_name

    job = ReplicationJob(
        dataset_filename=args.dataset_filename,
        dataset_for_rep=args.dataset_for_rep,
        full_path=str(train_dataset_root),
        outcome_label=outcome_label,
        outcome_type=outcome_type,
        instance_label=instance_label,
        match_label=args.match_label,
        ignore_features=metadata.get("Ignored Features", []),
        cv_partitions=metadata.get("CV Partitions", 5),
        exclude_plots=[x.strip() for x in args.exclude_plots.split(",") if x.strip()],
        categorical_cutoff=metadata.get("Categorical Cutoff", 10),
        sig_cutoff=metadata.get("Statistical Significance Cutoff", 0.05),
        featureeng_missingness=metadata.get("Engineering Missingness Cutoff", 0.5),
        cleaning_missingness=metadata.get("Cleaning Missingness Cutoff", 0.5),
        scale_data=metadata.get("Use Data Scaling", True),
        impute_data=metadata.get("Use Data Imputation", True),
        multi_impute=metadata.get("Use Multivariate Imputation", False),
        show_plots=bool(args.show_plots),
        scoring_metric=metadata.get("Primary Metric", "balanced_accuracy"),
        random_state=metadata.get("Random Seed"),
    )
    job.run()


if __name__ == "__main__":
    main()
