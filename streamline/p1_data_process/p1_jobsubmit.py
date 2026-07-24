import argparse
import os
import sys
from pathlib import Path

import pandas as pd

try:
    from .data_process import DataProcess
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from streamline.p1_data_process.data_process import DataProcess

def _maybe_list(s):
    if s is None or s == '':
        return None
    return [x.strip() for x in s.split(',') if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', default='')
    ap.add_argument('--dataset_name', default='')
    ap.add_argument('--output_path', required=True)
    ap.add_argument('--experiment_name', required=True)
    ap.add_argument('--exclude', default='')

    ap.add_argument('--outcome_label', required=True)
    ap.add_argument('--outcome_type', default='')
    ap.add_argument('--instance_label', default='')
    ap.add_argument('--match_label', default='')

    ap.add_argument('--n_splits', type=int, default=10)
    ap.add_argument('--partition_method', default='Stratified')

    ap.add_argument('--ignore_features', default='')
    ap.add_argument('--categorical_features', default='')
    ap.add_argument('--quantitative_features', default='')
    ap.add_argument('--top_features', type=int, default=20)

    ap.add_argument('--categorical_cutoff', type=int, default=10)
    ap.add_argument('--sig_cutoff', type=float, default=0.05)
    ap.add_argument('--featureeng_missingness', type=float, default=0.5)
    ap.add_argument('--cleaning_missingness', type=float, default=0.5)
    ap.add_argument('--correlation_removal_threshold', type=float, default=1.0)
    ap.add_argument('--random_state', default='')
    ap.add_argument('--one_hot_encoding', type=int, default=1)
    ap.add_argument('--cv_provided', type=int, default=0)
    ap.add_argument('--cv_input_root', default='')

    # plotting flags
    ap.add_argument('--enable_plots', type=int, default=0)
    ap.add_argument('--plot_missingness', type=int, default=0)
    ap.add_argument('--plot_class_counts', type=int, default=0)
    ap.add_argument('--plot_correlation', type=int, default=0)
    ap.add_argument('--correlation_plot_max_features', type=int, default=200)
    ap.add_argument('--plot_univariate', type=int, default=0)
    ap.add_argument('--univariate_top_k', type=int, default=20)
    ap.add_argument('--plot_anomalies', type=int, default=0)
    args = ap.parse_args()

    if args.dataset_path:
        ext = args.dataset_path.split('.')[-1].lower()
        if ext == 'csv':
            df = pd.read_csv(args.dataset_path, na_values='NA', sep=',')
        elif ext == 'tsv':
            df = pd.read_csv(args.dataset_path, na_values='NA', sep='\t')
        else:
            df = pd.read_csv(args.dataset_path, na_values='NA', delim_whitespace=True)
        dataset_name = args.dataset_name or os.path.basename(args.dataset_path).split('.')[0]
    else:
        if not args.cv_provided or not args.cv_input_root:
            raise ValueError("dataset_path is required unless cv_provided=1 and cv_input_root is set")
        cv_dir = os.path.join(args.cv_input_root, "CVDatasets")
        if not os.path.isdir(cv_dir):
            raise ValueError(f"Expected CVDatasets/ under cv_input_root: {args.cv_input_root}")
        train_files = sorted(
            f for f in os.listdir(cv_dir)
            if f.endswith("_Train.csv") and "_CV_" in f
        )
        if not train_files:
            raise ValueError(f"No Train split CSVs found under {cv_dir}")
        df = pd.read_csv(os.path.join(cv_dir, train_files[0]), na_values='NA', sep=',')
        dataset_name = args.dataset_name or os.path.basename(args.cv_input_root.rstrip(os.sep))
    df.columns = df.columns.str.strip()

    experiment_path = os.path.join(args.output_path, args.experiment_name)

    dp = DataProcess(
        data=df,
        experiment_path=experiment_path,
        outcome_label=args.outcome_label,
        outcome_type=(args.outcome_type or None),
        match_label=(args.match_label if args.match_label in df.columns else None) or None,
        instance_label=(args.instance_label if args.instance_label in df.columns else None) or None,
        ignore_features=_maybe_list(args.ignore_features),
        categorical_features=_maybe_list(args.categorical_features),
        quantitative_features=_maybe_list(args.quantitative_features),
        exclude_eda_output=_maybe_list(args.exclude),
        categorical_cutoff=args.categorical_cutoff,
        sig_cutoff=args.sig_cutoff,
        featureeng_missingness=args.featureeng_missingness,
        cleaning_missingness=args.cleaning_missingness,
        correlation_removal_threshold=args.correlation_removal_threshold,
        partition_method=args.partition_method,
        n_splits=args.n_splits,
        one_hot_encoding=bool(args.one_hot_encoding),
        random_state=(int(args.random_state) if args.random_state != '' else None),
        show_plots=False,  # batch jobs shouldn't pop plots
        cv_provided=bool(args.cv_provided),
        cv_input_path=(args.cv_input_root or None),
        dataset_name=dataset_name,

        # plotting flags
        enable_plots=bool(args.enable_plots),
        plot_missingness=bool(args.plot_missingness),
        plot_class_counts=bool(args.plot_class_counts),
        plot_correlation=bool(args.plot_correlation),
        correlation_plot_max_features=int(args.correlation_plot_max_features),
        plot_univariate=bool(args.plot_univariate),
        univariate_top_k=int(args.univariate_top_k),
        plot_anomalies=bool(args.plot_anomalies),
    )

    dp.run(top_features=int(args.top_features))


if __name__ == "__main__":
    main()
