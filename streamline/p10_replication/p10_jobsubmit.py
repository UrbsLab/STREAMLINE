# streamline/p10_replication/p10_jobsubmit.py

from __future__ import annotations
import argparse
import os
import pickle
from pathlib import Path

from streamline.p10_replication.replication import ReplicationJob


def main():
    ap = argparse.ArgumentParser("STREAMLINE Phase 10 Replication JobSubmit")
    ap.add_argument("--dataset_filename", required=True)
    ap.add_argument("--dataset_for_rep", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--experiment_name", required=True)
    ap.add_argument("--outcome_label", default=None)
    ap.add_argument("--instance_label", default=None)
    ap.add_argument("--match_label", default=None)
    ap.add_argument("--exclude_plots", default="")
    ap.add_argument("--show_plots", type=int, default=0)
    args = ap.parse_args()

    exp_root = Path(args.output_path) / args.experiment_name
    with open(exp_root / "metadata.pickle", "rb") as f:
        metadata = pickle.load(f)

    outcome_label = args.outcome_label or metadata["Outcome Label"]
    outcome_type = metadata["Outcome Type"]
    instance_label = args.instance_label or metadata["Instance Label"]

    ignore_features = metadata.get("Ignored Features", [])
    categorical_cutoff = metadata["Categorical Cutoff"]
    sig_cutoff = metadata["Statistical Significance Cutoff"]
    cv_partitions = metadata["CV Partitions"]
    scale_data = metadata["Use Data Scaling"]
    impute_data = metadata["Use Data Imputation"]
    multi_impute = metadata["Use Multivariate Imputation"]
    scoring_metric = metadata["Primary Metric"]
    random_state = metadata["Random Seed"]

    exclude_plots = [s for s in args.exclude_plots.split(",") if s]

    data_name = Path(args.dataset_for_rep).stem
    full_path = str(exp_root / data_name)

    job = ReplicationJob(
        dataset_filename=args.dataset_filename,
        dataset_for_rep=args.dataset_for_rep,
        full_path=full_path,
        outcome_label=outcome_label,
        outcome_type=outcome_type,
        instance_label=instance_label,
        match_label=args.match_label,
        ignore_features=ignore_features,
        cv_partitions=cv_partitions,
        exclude_plots=exclude_plots,
        categorical_cutoff=categorical_cutoff,
        sig_cutoff=sig_cutoff,
        scale_data=scale_data,
        impute_data=impute_data,
        multi_impute=multi_impute,
        show_plots=bool(args.show_plots),
        scoring_metric=scoring_metric,
        random_state=random_state,
    )
    job.run()


if __name__ == "__main__":
    main()
