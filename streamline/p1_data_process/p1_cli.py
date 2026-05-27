# streamline/phases/p1_data_process/cli.py
import argparse
import json
from typing import List, Optional

from streamline.p1_data_process.p1_runner import P1Runner
from streamline.utils.run_commands import (
    add_run_command_args,
    apply_saved_run_command,
    save_run_command_from_args,
    snapshot_args,
)


def _csv_or_list(v: Optional[str]) -> Optional[List[str]]:
    if v is None or v == "":
        return None
    # allow JSON array OR comma-separated
    v = v.strip()
    if v.startswith("["):
        try:
            arr = json.loads(v)
            return [str(x) for x in arr]
        except Exception:
            pass
    return [x.strip() for x in v.split(",") if x.strip() != ""]


def _csv_or_str(v: Optional[str]):
    # for features that allow either single string or list
    if v is None:
        return None
    v = v.strip()
    return v if ("," not in v and not v.startswith("[")) else _csv_or_list(v)


def _bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    vl = str(v).strip().lower()
    if vl in ("1", "true", "t", "yes", "y"):
        return True
    if vl in ("0", "false", "f", "no", "n"):
        return False
    return default


def main():
    ap = argparse.ArgumentParser(
        "STREAMLINE Phase 1 Runner (dataset-free, Dask-aware)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Core paths
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--experiment_name", required=True)

    # Labels & schema
    ap.add_argument("--outcome_label", default="Class")
    ap.add_argument("--outcome_type", default=None)
    ap.add_argument("--instance_label", default=None)
    ap.add_argument("--match_label", default=None)
    ap.add_argument("--ignore_features", default=None)
    ap.add_argument("--categorical_features", default=None)
    ap.add_argument("--quantitative_features", default=None)
    ap.add_argument("--categorical_cutoff", default=10, type=int)
    ap.add_argument(
        "--one_hot_encoding",
        default="true",
        help="Set false/0 to keep non-binary categorical features unexpanded for model-stage handling.",
    )

    # CV
    ap.add_argument("--n_splits", default=10, type=int)
    ap.add_argument("--partition_method", default="Stratified")

    # EDA / thresholds
    ap.add_argument("--top_features", default=20, type=int)
    ap.add_argument("--sig_cutoff", default=0.05, type=float)
    ap.add_argument("--featureeng_missingness", default=0.5, type=float)
    ap.add_argument("--cleaning_missingness", default=0.5, type=float)
    ap.add_argument("--correlation_removal_threshold", default=1.0, type=float)

    # Execution
    ap.add_argument("--run_cluster", default="Serial", help='Serial | Local | BashSLURM | BashLSF | "<dask-cluster-name>"')
    ap.add_argument("--queue", default="defq")
    ap.add_argument("--reserved_memory", default=4, type=int)
    ap.add_argument("--random_state", default=None, type=int)
    ap.add_argument("--show_plots", default="false")
    ap.add_argument("--force", default="false")

    # Import-only CV
    ap.add_argument("--cv_provided", default="false")
    ap.add_argument("--cv_input_root", default=None)

    # EDA outputs & plotting flags
    ap.add_argument("--exclude_eda_output", default=None, help='JSON array or comma list (e.g. ["describe_csv","correlation"])')
    ap.add_argument("--enable_plots", default="false")
    ap.add_argument("--plot_missingness", default="false")
    ap.add_argument("--plot_class_counts", default="false")
    ap.add_argument("--plot_correlation", default="false")
    ap.add_argument("--correlation_plot_max_features", default=200, type=int)
    ap.add_argument("--plot_univariate", default="false")
    ap.add_argument("--univariate_top_k", default=20, type=int)
    ap.add_argument("--plot_anomalies", default="false")
    add_run_command_args(ap)

    args = ap.parse_args()
    args = apply_saved_run_command(ap, args, "p1_data_process")
    run_command_args = snapshot_args(args)

    runner = P1Runner(
        data_path=args.data_path,
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        exclude_eda_output=_csv_or_list(args.exclude_eda_output),
        outcome_label=args.outcome_label,
        outcome_type=args.outcome_type,
        instance_label=args.instance_label,
        match_label=args.match_label,
        n_splits=args.n_splits,
        partition_method=args.partition_method,
        ignore_features=_csv_or_str(args.ignore_features),
        categorical_features=_csv_or_str(args.categorical_features),
        quantitative_features=_csv_or_str(args.quantitative_features),
        top_features=args.top_features,
        categorical_cutoff=args.categorical_cutoff,
        sig_cutoff=args.sig_cutoff,
        featureeng_missingness=args.featureeng_missingness,
        cleaning_missingness=args.cleaning_missingness,
        correlation_removal_threshold=args.correlation_removal_threshold,
        random_state=args.random_state,
        run_cluster=args.run_cluster if args.run_cluster not in (None, "Serial", "False", "false") else False,
        queue=args.queue,
        reserved_memory=args.reserved_memory,
        show_plots=_bool(args.show_plots, False),
        one_hot_encoding=_bool(args.one_hot_encoding, True),
        cv_provided=_bool(args.cv_provided, False),
        cv_input_root=args.cv_input_root,
        enable_plots=_bool(args.enable_plots, False),
        plot_missingness=_bool(args.plot_missingness, False),
        plot_class_counts=_bool(args.plot_class_counts, False),
        plot_correlation=_bool(args.plot_correlation, False),
        correlation_plot_max_features=args.correlation_plot_max_features,
        plot_univariate=_bool(args.plot_univariate, False),
        univariate_top_k=args.univariate_top_k,
        plot_anomalies=_bool(args.plot_anomalies, False),
        force=_bool(args.force, False),
    )

    runner.run()
    save_run_command_from_args(args, "p1_data_process", run_command_args)


if __name__ == "__main__":
    # # Serial
    # python -m streamline.p1_data_process.p1_cli \
    # --data_path ./data/UCIBinaryClassification \
    # --output_path ./test \
    # --experiment_name MyExp \
    # --outcome_label Class \
    # --outcome_type Binary \
    # --instance_label InstanceID

    # # Local Dask with 1 worker per core
    # python -m streamline.p1_data_process.p1_cli \
    # --data_path ./data/UCIBinaryClassification \
    # --output_path ./test \
    # --experiment_name MyExp \
    # --outcome_label Class \
    # --outcome_type Binary \
    # --instance_label InstanceID \
    # --run_cluster Local

    main()
