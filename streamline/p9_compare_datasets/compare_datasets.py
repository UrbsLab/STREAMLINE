# streamline/p9_compare_datasets/compare_datasets.py
from __future__ import annotations

import os
import time
import logging
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, wilcoxon, mannwhitneyu

import seaborn as sns

from streamline.p6_modeling.utils.loader import list_models, get_model_by_id

sns.set_theme()
logger = logging.getLogger(__name__)


class DatasetCompareJob:
    """
    Phase 9: Dataset-level performance comparison across all datasets
    in an experiment.

    Modernized CompareJob that:
      * discovers base models from Phase 6 / 8 outputs + registry
      * **optionally includes Phase 7 ensembles** if ensemble_evaluation
        artifacts are present in the datasets.
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        experiment_name: Optional[str] = None,
        experiment_path: Optional[str] = None,
        outcome_label: str = "Class",
        outcome_type: str = "Binary",
        instance_label: Optional[str] = None,
        sig_cutoff: float = 0.05,
        show_plots: bool = False,
    ):
        super().__init__()
        assert (output_path is not None and experiment_name is not None) or (
            experiment_path is not None
        ), "Either (output_path, experiment_name) or experiment_path must be provided."

        if experiment_path is None:
            self.output_path = output_path
            self.experiment_name = experiment_name
            self.experiment_path = os.path.join(self.output_path, self.experiment_name)
        else:
            self.experiment_path = experiment_path
            self.experiment_name = Path(self.experiment_path).name
            self.output_path = str(Path(self.experiment_path).parent)

        self.outcome_label = outcome_label
        self.outcome_type = outcome_type
        self.instance_label = instance_label
        self.sig_cutoff = sig_cutoff
        self.show_plots = show_plots

        self.exp_root = os.path.join(self.output_path, self.experiment_name)
        if not os.path.isdir(self.exp_root):
            raise Exception("Experiment must exist before Phase 9 can begin")

        # Collect dataset dirs that have CVDatasets
        datasets = [
            os.path.join(self.exp_root, name)
            for name in sorted(os.listdir(self.exp_root))
            if os.path.isdir(os.path.join(self.exp_root, name))
            and name
            not in {
                "jobsCompleted",
                "jobs",
                "logs",
                "dask_logs",
                "runtime",
                "DatasetComparisons",
            }
            and os.path.isdir(os.path.join(self.exp_root, name, "CVDatasets"))
        ]
        if not datasets:
            logging.warning("No datasets found for Phase 9 under %s", self.exp_root)
            return

        self.datasets: List[str] = [Path(d).name for d in datasets]
        self.dataset_directory_paths: List[str] = datasets

        if not self.dataset_directory_paths:
            raise RuntimeError(
                f"No dataset folders found under experiment: {self.experiment_path}"
            )

        # Discover base models from Phase 6 metrics
        (
            base_algorithms,
            base_abbrev,
            base_colors,
        ) = self._discover_algorithms_from_metrics(
            Path(self.dataset_directory_paths[0]), self.outcome_type
        )

        # Discover ensembles (if any) from Phase 7 artifacts
        (
            ensemble_algorithms,
            ensemble_abbrev,
            ensemble_colors,
        ) = self._discover_ensembles_from_metrics(Path(self.dataset_directory_paths[0]))

        self.base_algorithms: List[str] = sorted(base_algorithms)
        self.ensemble_algorithms: List[str] = sorted(ensemble_algorithms)

        # merge base + ensembles into unified view
        self.algorithms: List[str] = sorted(self.base_algorithms + self.ensemble_algorithms)
        self.abbrev: Dict[str, str] = {}
        self.abbrev.update(base_abbrev)
        self.abbrev.update(ensemble_abbrev)

        self.colors: Dict[str, Any] = {}
        self.colors.update(base_colors)
        self.colors.update(ensemble_colors)

        self.metrics: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Algorithm discovery (base models, from model_evaluation/metrics_by_cv)
    # ------------------------------------------------------------------
    def _discover_algorithms_from_metrics(
        self,
        dataset_dir: Path,
        outcome_type: str,
    ) -> Tuple[List[str], Dict[str, str], Dict[str, Any]]:
        """
        Discover base modeling algorithms for dataset comparison from:
            <dataset_dir>/model_evaluation/metrics_by_cv/<ALG>_CV_<k>.json

        Returns:
            algorithms: list of model_name
            abbrev: mapping model_name -> small_name (file prefix)
            colors: mapping model_name -> color (named or RGB tuples)
        """
        metrics_dir = dataset_dir / "model_evaluation" / "metrics_by_cv"
        present_algs: List[str] = []

        if metrics_dir.is_dir():
            for fn in os.listdir(metrics_dir):
                if not fn.endswith(".json"):
                    continue
                # Expect pattern "<ALG>_CV_<fold>.json"
                parts = fn.split("_CV_")
                if len(parts) != 2:
                    continue
                alg = parts[0]
                if alg:
                    present_algs.append(alg)

        present_set = set(present_algs)
        if not present_set:
            logger.warning(
                "DatasetComparePhaseJob: no base-model metrics found under %s",
                metrics_dir,
            )

        algorithms: List[str] = []
        abbrev: Dict[str, str] = {}
        colors: Dict[str, Any] = {}

        registry_entries: List[Dict[str, Any]] = []
        if list_models is not None:
            try:
                registry_entries = list_models(outcome_type)
            except Exception as e:
                logger.warning(
                    "DatasetComparePhaseJob: list_models(%s) failed: %r",
                    outcome_type,
                    e,
                )

        # small_name -> (model_type, entry)
        by_small: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for entry in registry_entries:
            small = (entry.get("small_name") or "").strip()
            mt = (entry.get("model_type") or "").strip()
            if small:
                by_small[small] = (mt, entry)

        for small in sorted(present_set):
            mt, entry = by_small.get(small, ("", {}))
            cls = None
            if get_model_by_id is not None and mt:
                # try by small_name first
                try:
                    cls = get_model_by_id(mt, small)
                except Exception:
                    model_name = entry.get("model_name") or entry.get("alt_id") or small
                    try:
                        cls = get_model_by_id(mt, model_name)
                    except Exception:
                        cls = None

            model_name = cls.model_name if cls is not None else small
            algorithms.append(model_name)
            abbrev[model_name] = cls.small_name if cls is not None else small

            if cls is not None and hasattr(cls, "color"):
                colors[model_name] = cls.color

        # Fallback colors via seaborn palette
        if algorithms:
            palette = sns.color_palette("tab10", n_colors=len(algorithms))
            for i, alg in enumerate(algorithms):
                colors.setdefault(alg, palette[i % len(algorithms)])
        else:
            algorithms = sorted(present_set)
            abbrev = {a: a for a in algorithms}
            palette = sns.color_palette("tab10", n_colors=max(len(algorithms), 1))
            colors = {a: palette[i % len(algorithms)] for i, a in enumerate(algorithms)}

        return algorithms, abbrev, colors

    # ------------------------------------------------------------------
    # Ensemble discovery (from ensemble_evaluation/metrics_by_cv + summaries)
    # ------------------------------------------------------------------
    def _discover_ensembles_from_metrics(
        self,
        dataset_dir: Path,
    ) -> Tuple[List[str], Dict[str, str], Dict[str, Any]]:
        """
        Discover ensembles for dataset comparison.

        Uses:
          <dataset_dir>/ensemble_evaluation/metrics_by_cv/<ENS>_CV_<k>.json

        and additionally the summary tables:
          <dataset_dir>/ensemble_evaluation/Ensembles_performance_mean.csv
          <dataset_dir>/ensemble_evaluation/Ensembles_performance_median.csv
          <dataset_dir>/ensemble_evaluation/Ensembles_performance_std.csv

        to detect presence and ensure metric names line up, but we rely on
        metrics_by_cv for per-CV distributions.
        """
        ens_root = dataset_dir / "ensemble_evaluation"
        metrics_dir = ens_root / "metrics_by_cv"

        present_ens: List[str] = []

        if metrics_dir.is_dir():
            for fn in os.listdir(metrics_dir):
                if not fn.endswith(".json"):
                    continue
                # Expect pattern "<ENS>_CV_<fold>.json"
                parts = fn.split("_CV_")
                if len(parts) != 2:
                    continue
                ens_id = parts[0]
                if ens_id:
                    present_ens.append(ens_id)

        present_set = sorted(set(present_ens))
        if not present_set:
            # No ensembles; this is fine.
            return [], {}, {}

        algorithms: List[str] = []
        abbrev: Dict[str, str] = {}
        colors: Dict[str, Any] = {}

        # For now, just use ensemble ID as both name and abbrev.
        for ens_id in present_set:
            algorithms.append(ens_id)
            abbrev[ens_id] = ens_id

        # Colors: append to existing palette after base models
        palette = sns.color_palette("tab10", n_colors=max(len(algorithms), 1))
        for i, ens in enumerate(algorithms):
            colors[ens] = palette[i % len(algorithms)]

        return algorithms, abbrev, colors

    # ------------------------------------------------------------------
    # PUBLIC ENTRY
    # ------------------------------------------------------------------
    def run(self):
        self.job_start_time = time.time()
        logger.info(
            "Running dataset comparison (Phase 9) for experiment %s",
            self.experiment_name,
        )

        # metrics from first dataset (Summary_performance_mean)
        first_summary = (
            Path(self.dataset_directory_paths[0])
            / "model_evaluation"
            / "Summary_performance_mean.csv"
        )
        if not first_summary.is_file():
            raise RuntimeError(
                f"Expected Summary_performance_mean.csv under {first_summary.parent}"
            )
        data = pd.read_csv(first_summary, sep=",")
        self.metrics = data.columns.values.tolist()[1:]

        # Create output directory
        dc_dir = Path(self.experiment_path) / "DatasetComparisons"
        dc_dir.mkdir(exist_ok=True)

        logger.info("Running Kruskal-Wallis across datasets...")
        self.kruscall_wallis()

        logger.info("Running Mann-Whitney U across datasets...")
        self.mann_whitney_u()

        logger.info("Running Wilcoxon rank-sum across datasets...")
        self.wilcoxon_rank()

        logger.info("Running 'best algorithm per dataset' Kruskal-Wallis...")
        global_data = self.best_kruscall_wallis()

        logger.info("Running 'best algorithm per dataset' Mann-Whitney...")
        self.best_mann_whitney_u(global_data)

        logger.info("Running 'best algorithm per dataset' Wilcoxon rank...")
        self.best_wilcoxon_rank(global_data)

        logger.info("Generating dataset comparison boxplots (all models)...")
        self.data_compare_bp_all()

        logger.info("Generating dataset comparison boxplots (per algorithm)...")
        self.data_compare_bp()

        self.save_runtime()
        logger.info("Phase 9 dataset comparison complete.")
        jobs_dir = Path(self.experiment_path) / "jobsCompleted"
        jobs_dir.mkdir(exist_ok=True)
        with open(jobs_dir / "job_compare_datasets.txt", "w") as f:
            f.write("complete")

    # ------------------------------------------------------------------
    # Core comparison methods
    # ------------------------------------------------------------------
    def kruscall_wallis(self):
        label = ["Statistic", "P-Value", "Sig(*)"]
        for i in range(1, len(self.datasets) + 1):
            label.append(f"Median_D{i}")
            label.append(f"Mean_D{i}")
            label.append(f"Std_D{i}")

        dc_dir = Path(self.experiment_path) / "DatasetComparisons"

        for algorithm in self.algorithms:
            kruskal_summary = pd.DataFrame(index=self.metrics, columns=label)
            for metric in self.metrics:
                temp_array = []
                med_list = []
                mean_list = []
                std_list = []

                for dataset_path in self.dataset_directory_paths:
                    td = self._load_performance_df(dataset_path, algorithm)
                    if metric not in td.columns:
                        # If ensemble/base is missing this metric, skip this dataset for it.
                        continue
                    vals = td[metric].astype(float)
                    temp_array.append(vals)
                    med_list.append(vals.median())
                    mean_list.append(vals.mean())
                    std_list.append(vals.std())

                if not temp_array:
                    # nothing to compare for this metric / algorithm
                    kruskal_summary.at[metric, "Statistic"] = "NA"
                    kruskal_summary.at[metric, "P-Value"] = "1.0"
                    kruskal_summary.at[metric, "Sig(*)"] = ""
                    continue

                try:
                    result = kruskal(*temp_array)
                except Exception:
                    result = ["NA", 1.0]

                try:
                    kruskal_summary.at[metric, "Statistic"] = str(round(result[0], 6))
                except TypeError:
                    kruskal_summary.at[metric, "Statistic"] = "NA"
                kruskal_summary.at[metric, "P-Value"] = str(round(result[1], 6))
                kruskal_summary.at[metric, "Sig(*)"] = (
                    "*" if result[1] < self.sig_cutoff else ""
                )

                for j in range(len(med_list)):
                    kruskal_summary.at[metric, f"Median_D{j+1}"] = str(
                        round(med_list[j], 6)
                    )
                for j in range(len(mean_list)):
                    kruskal_summary.at[metric, f"Mean_D{j+1}"] = str(
                        round(mean_list[j], 6)
                    )
                for j in range(len(std_list)):
                    kruskal_summary.at[metric, f"Std_D{j+1}"] = str(
                        round(std_list[j], 6)
                    )

            out = dc_dir / f"KruskalWallis_{self.abbrev[algorithm]}.csv"
            kruskal_summary.to_csv(out)

    def wilcoxon_rank(self):
        label = ["Metric", "Data1", "Data2", "Statistic", "P-Value", "Sig(*)"]
        for i in range(1, 3):
            label.append(f"Median_Data{i}")
            label.append(f"Mean_Data{i}")
            label.append(f"Std_Data{i}")

        master: List[List[Any]] = []
        for algorithm in self.algorithms:
            master.extend(self.inter_set_fn(wilcoxon, algorithm))

        df = pd.DataFrame(master, columns=label)
        out = Path(self.experiment_path) / "DatasetComparisons" / "WilcoxonRank_all.csv"
        df.to_csv(out, index=False)

    def mann_whitney_u(self):
        label = ["Metric", "Data1", "Data2", "Statistic", "P-Value", "Sig(*)"]
        for i in range(1, 3):
            label.append(f"Median_Data{i}")
            label.append(f"Mean_Data{i}")
            label.append(f"Std_Data{i}")

        master: List[List[Any]] = []
        for algorithm in self.algorithms:
            master.extend(self.inter_set_fn(mannwhitneyu, algorithm))

        df = pd.DataFrame(master, columns=label)
        out = Path(self.experiment_path) / "DatasetComparisons" / "MannWhitney_all.csv"
        df.to_csv(out, index=False)

    def best_kruscall_wallis(self):
        label = ["Statistic", "P-Value", "Sig(*)"]
        for i in range(1, len(self.datasets) + 1):
            label.append(f"Best_Alg_D{i}")
            label.append(f"Median_D{i}")
            label.append(f"Mean_D{i}")
            label.append(f"Std_D{i}")

        dc_dir = Path(self.experiment_path) / "DatasetComparisons"
        kruskal_summary = pd.DataFrame(index=self.metrics, columns=label)
        global_data: List[Any] = []

        for metric in self.metrics:
            best_list = []
            best_data = []

            for dataset_path in self.dataset_directory_paths:
                alg_med = []
                alg_mean = []
                alg_std = []
                alg_data = []
                alg_names = []

                for algorithm in self.algorithms:
                    td = self._load_performance_df(dataset_path, algorithm)
                    if metric not in td.columns:
                        continue
                    vals = td[metric].astype(float)
                    alg_med.append(vals.median())
                    alg_mean.append(vals.mean())
                    alg_std.append(vals.std())
                    alg_data.append(vals)
                    alg_names.append(algorithm)

                if not alg_mean:
                    # no algorithm has this metric for this dataset
                    continue

                best_mean = max(alg_mean)
                best_index = alg_mean.index(best_mean)
                best_alg = alg_names[best_index]
                best_data.append(alg_data[best_index])
                best_list.append(
                    [
                        best_alg,
                        alg_med[best_index],
                        alg_mean[best_index],
                        alg_std[best_index],
                    ]
                )

            global_data.append([best_data, best_list])

            if not best_data:
                kruskal_summary.at[metric, "Statistic"] = str(round(np.nan, 6))
                kruskal_summary.at[metric, "P-Value"] = str(round(np.nan, 6))
                kruskal_summary.at[metric, "Sig(*)"] = ""
                continue

            try:
                result = kruskal(*best_data)
                kruskal_summary.at[metric, "Statistic"] = str(round(result[0], 6))
                kruskal_summary.at[metric, "P-Value"] = str(round(result[1], 6))
                kruskal_summary.at[metric, "Sig(*)"] = (
                    "*" if result[1] < self.sig_cutoff else ""
                )
            except ValueError:
                kruskal_summary.at[metric, "Statistic"] = str(round(np.nan, 6))
                kruskal_summary.at[metric, "P-Value"] = str(round(np.nan, 6))
                kruskal_summary.at[metric, "Sig(*)"] = ""

            for j, (alg, medv, meanv, stdv) in enumerate(best_list):
                kruskal_summary.at[metric, f"Best_Alg_D{j+1}"] = str(alg)
                kruskal_summary.at[metric, f"Median_D{j+1}"] = str(round(medv, 6))
                kruskal_summary.at[metric, f"Mean_D{j+1}"] = str(round(meanv, 6))
                kruskal_summary.at[metric, f"Std_D{j+1}"] = str(round(stdv, 6))

        out = dc_dir / "BestCompare_KruskalWallis.csv"
        kruskal_summary.to_csv(out)
        return global_data

    def best_mann_whitney_u(self, global_data):
        df = self.inter_set_best_fn(mannwhitneyu, global_data)
        out = Path(self.experiment_path) / "DatasetComparisons" / "BestCompare_MannWhitney.csv"
        df.to_csv(out, index=False)

    def best_wilcoxon_rank(self, global_data):
        df = self.inter_set_best_fn(wilcoxon, global_data)
        out = Path(self.experiment_path) / "DatasetComparisons" / "BestCompare_WilcoxonRank.csv"
        df.to_csv(out, index=False)

    def data_compare_bp_all(self):
        """
        For each metric, generate boxplots comparing algorithm performance across datasets.
        Includes both base models and ensembles (if present).

        Uses:
          model_evaluation/Summary_performance_mean.csv
          + ensemble_evaluation/Ensembles_performance_mean.csv (if present)
        """
        dc_bp_dir = (
            Path(self.experiment_path) / "DatasetComparisons" / "dataCompBoxplots"
        )
        dc_bp_dir.mkdir(parents=True, exist_ok=True)

        for metric in self.metrics:
            df = pd.DataFrame()
            data_name_list: List[str] = []
            alg_values_dict: Dict[str, List[float]] = {alg: [] for alg in self.algorithms}

            for each in self.dataset_directory_paths:
                data_name_list.append(Path(each).name)

                # Base model summary
                base_path = (
                    Path(each)
                    / "model_evaluation"
                    / "Summary_performance_mean.csv"
                )
                if not base_path.is_file():
                    continue
                data = pd.read_csv(base_path, sep=",", index_col=0)

                # Append ensemble summary if present
                ens_path = (
                    Path(each)
                    / "ensemble_evaluation"
                    / "Ensembles_performance_mean.csv"
                )
                if ens_path.is_file():
                    ens_df = pd.read_csv(ens_path, sep=",")
                    if "Ensemble" in ens_df.columns:
                        ens_df = ens_df.set_index("Ensemble")
                    # align on metric columns if needed
                    if metric in ens_df.columns:
                        data = pd.concat([data, ens_df], axis=0)

                if metric not in data.columns:
                    # no values for this metric in this dataset
                    col = pd.Series([], dtype=float)
                else:
                    col = data[metric]

                col_list = col.tolist()
                rownames = list(data.index.values)

                # record per-algorithm trajectories
                for j, alg in enumerate(rownames):
                    if j < len(col_list):
                        val = col_list[j]
                        alg_values_dict.setdefault(alg, []).append(val)

                df = pd.concat([df, col], axis=1)

            if df.empty:
                continue

            df.columns = data_name_list
            df.boxplot(column=data_name_list, rot=90)

            # overlay algorithm trajectories
            for alg in self.algorithms:
                vals = alg_values_dict.get(alg)
                if not vals or len(vals) != len(self.dataset_directory_paths):
                    continue
                plt.plot(
                    np.arange(len(self.dataset_directory_paths)) + 1,
                    vals,
                    color=self.colors.get(alg, "C0"),
                    label=alg,
                )

            plt.ylabel(str(metric))
            plt.xlabel("Dataset")
            plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
            out = dc_bp_dir / f"DataCompareAllModels_{metric}.png"
            plt.savefig(out, bbox_inches="tight")
            if self.show_plots:
                plt.show()
            else:
                plt.close("all")

    def data_compare_bp(self):
        """
        Per-algorithm boxplots comparing distributions of a target metric
        across datasets. Works for both base models and ensembles.
        """
        dc_bp_dir = (
            Path(self.experiment_path) / "DatasetComparisons" / "dataCompBoxplots"
        )
        dc_bp_dir.mkdir(parents=True, exist_ok=True)

        if self.outcome_type == "Binary":
            metric_list = ["ROC AUC", "PRC AUC"]
        else:
            metric_list = [
                "Max Error",
                "Mean Absolute Error",
                "Mean Squared Error",
                "Median Absolute Error",
                "Explained Variance",
                "Pearson Correlation",
            ]

        for algorithm in self.algorithms:
            for metric in metric_list:
                df = pd.DataFrame()
                data_name_list: List[str] = []
                for each in self.dataset_directory_paths:
                    td = self._load_performance_df(each, algorithm)
                    if metric not in td.columns:
                        continue
                    col = td[metric].astype(float)
                    df = pd.concat([df, col], axis=1)
                    data_name_list.append(Path(each).name)

                if df.empty:
                    continue

                df.columns = data_name_list
                df.boxplot(column=data_name_list, rot=90)
                plt.ylabel(str(metric))
                plt.xlabel("Dataset")
                plt.title(algorithm)
                out = dc_bp_dir / f"DataCompare_{self.abbrev[algorithm]}_{metric}.png"
                plt.savefig(out, bbox_inches="tight")
                if self.show_plots:
                    plt.show()
                else:
                    plt.close("all")

    def save_runtime(self):
        runtime_dir = Path(self.experiment_path) / "runtime"
        runtime_dir.mkdir(exist_ok=True)
        runtime_file = runtime_dir / "runtime_compare_datasets.txt"
        with runtime_file.open("w") as f:
            f.write(str(time.time() - self.job_start_time))

    # ------------------------------------------------------------------
    # Shared helper methods
    # ------------------------------------------------------------------
    def _load_performance_df(self, dataset_path: str, algorithm: str) -> pd.DataFrame:
        """
        Unified loader for per-CV performance metrics for both base models
        and ensembles.

        Base models:
          <dataset>/model_evaluation/<abbr>_performance.csv

        Ensembles:
          <dataset>/ensemble_evaluation/metrics_by_cv/<ens_id>_CV_<k>.json
          -> converted on the fly to a DataFrame with one row per CV.
        """
        ds = Path(dataset_path)
        abbr = self.abbrev[algorithm]

        if algorithm in self.base_algorithms:
            path = ds / "model_evaluation" / f"{abbr}_performance.csv"
            if not path.is_file():
                # fallback: empty DF
                return pd.DataFrame()
            return pd.read_csv(path)

        # ensembles
        metrics_dir = ds / "ensemble_evaluation" / "metrics_by_cv"
        if not metrics_dir.is_dir():
            return pd.DataFrame()

        rows = []
        for fn in sorted(metrics_dir.glob(f"{abbr}_CV_*.json")):
            with fn.open("r") as f:
                data = json.load(f)
            # Accept both flat dicts and {"Balanced Accuracy": ..., ...}
            # They should already be flat from Phase 7.
            rows.append(data)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # ensure numeric where possible
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def inter_set_fn(self, fn, algorithm: str) -> List[List[Any]]:
        master_list: List[List[Any]] = []
        for metric in self.metrics:
            for x in range(0, len(self.dataset_directory_paths) - 1):
                for y in range(x + 1, len(self.dataset_directory_paths)):
                    td1 = self._load_performance_df(
                        self.dataset_directory_paths[x], algorithm
                    )
                    td2 = self._load_performance_df(
                        self.dataset_directory_paths[y], algorithm
                    )
                    if metric not in td1.columns or metric not in td2.columns:
                        continue

                    set1 = td1[metric].astype(float)
                    med1 = set1.median()
                    mean1 = set1.mean()
                    std1 = set1.std()

                    set2 = td2[metric].astype(float)
                    med2 = set2.median()
                    mean2 = set2.mean()
                    std2 = set2.std()

                    temp_list = self.temp_summary(set1, set2, x, y, metric, fn)
                    temp_list.extend(
                        [
                            str(round(med1, 6)),
                            str(round(med2, 6)),
                            str(round(mean1, 6)),
                            str(round(mean2, 6)),
                            str(round(std1, 6)),
                            str(round(std2, 6)),
                        ]
                    )
                    master_list.append(temp_list)
        return master_list

    def inter_set_best_fn(self, fn, global_data) -> pd.DataFrame:
        label = ["Metric", "Data1", "Data2", "Statistic", "P-Value", "Sig(*)"]
        for i in range(1, 3):
            label.append(f"Best_Alg_Data{i}")
            label.append(f"Median_Data{i}")
            label.append(f"Mean_Data{i}")
            label.append(f"Std_Data{i}")

        master_list: List[List[Any]] = []
        for j, metric in enumerate(self.metrics):
            for x in range(0, len(self.datasets) - 1):
                for y in range(x + 1, len(self.datasets)):
                    if not global_data[j][0]:
                        continue
                    set1 = global_data[j][0][x]
                    med1 = global_data[j][1][x][1]
                    mean1 = global_data[j][1][x][2]
                    std1 = global_data[j][1][x][3]

                    set2 = global_data[j][0][y]
                    med2 = global_data[j][1][y][1]
                    mean2 = global_data[j][1][y][2]
                    std2 = global_data[j][1][y][3]

                    temp_list = self.temp_summary(set1, set2, x, y, metric, fn)

                    temp_list.append(global_data[j][1][x][0])
                    temp_list.append(str(round(med1, 6)))
                    temp_list.append(str(round(mean1, 6)))
                    temp_list.append(str(round(std1, 6)))
                    temp_list.append(global_data[j][1][y][0])
                    temp_list.append(str(round(med2, 6)))
                    temp_list.append(str(round(mean2, 6)))
                    temp_list.append(str(round(std2, 6)))
                    master_list.append(temp_list)

        df = pd.DataFrame(master_list, columns=label)
        return df

    def temp_summary(self, set1, set2, x, y, metric, fn) -> List[Any]:
        temp_list: List[Any] = []
        if set1.equals(set2):
            result = ["NA", 1.0]
        else:
            try:
                result = fn(set1, set2)
            except Exception:
                result = ["NA_error", 1.0]

        temp_list.append(str(metric))
        temp_list.append(f"D{x+1}")
        temp_list.append(f"D{y+1}")

        if set1.equals(set2):
            temp_list.append(result[0])
        else:
            try:
                temp_list.append(str(round(result[0], 6)))
            except Exception:
                temp_list.append(result[0])

        temp_list.append(str(round(result[1], 6)))
        temp_list.append("*" if result[1] < self.sig_cutoff else "")

        return temp_list
