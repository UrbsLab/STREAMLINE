# streamline/p10_replication/replication.py

from __future__ import annotations
import csv
import glob
import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

from streamline.p1_data_process.data_process import DataProcess


from streamline.p6_modeling.utils.submodels import (
    BinaryClassificationModel,
    MulticlassClassificationModel,
    RegressionModel
)

from streamline.p8_summary_statistics.statistics import StatisticsPhaseJob as StatsJob


class ReplicationJob:
    """
    Phase 10 replication job.

    Applies all trained models from a single training dataset to a *replication* dataset:
      - Replays the full preprocessing / feature engineering pipeline
      - Replays scaling / imputation per CV partition
      - Applies models for each CV partition to the replication data
      - Runs the Phase 8-style statistics summary on replication results

    This is essentially the modernized version of the legacy ReplicateJob.
    """

    def __init__(
        self,
        dataset_filename: str,
        dataset_for_rep: str,
        full_path: str,
        outcome_label: str,
        outcome_type: str,
        instance_label: Optional[str],
        match_label: Optional[str],
        ignore_features: Optional[List[str]] = None,
        cv_partitions: int = 3,
        exclude_plots: Optional[List[str]] = None,
        categorical_cutoff: int = 10,
        sig_cutoff: float = 0.05,
        scale_data: bool = True,
        impute_data: bool = True,
        multi_impute: bool = True,
        show_plots: bool = False,
        scoring_metric: str = "balanced_accuracy",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.dataset_filename = dataset_filename
        self.dataset_for_rep = dataset_for_rep
        self.full_path = full_path  # <output>/<experiment>/<train_dataset>
        self.outcome_label = outcome_label
        self.outcome_type = outcome_type
        self.instance_label = instance_label
        self.match_label = match_label

        partial_path = str(Path(full_path).parent)
        with open(os.path.join(partial_path, "algInfo.pickle"), "rb") as f:
            alg_info = pickle.load(f)

        algorithms = []
        abbrev: Dict[str, str] = {}
        colors: Dict[str, str] = {}
        for algorithm, (use_flag, short_name, color) in alg_info.items():
            if use_flag:
                algorithms.append(algorithm)
                abbrev[algorithm] = short_name
                colors[algorithm] = color

        self.algorithms = algorithms
        self.abbrev = abbrev
        self.colors = colors

        known_exclude_options = [
            "plot_ROC",
            "plot_PRC",
            "plot_metric_boxplots",
            "feature_correlations",
        ]
        if exclude_plots is not None:
            for x in exclude_plots:
                if x not in known_exclude_options:
                    logging.warning("Unknown exclusion option %s", x)
        else:
            exclude_plots = []

        self.plot_roc = "plot_ROC" not in exclude_plots
        self.plot_prc = "plot_PRC" not in exclude_plots
        self.plot_metric_boxplots = "plot_metric_boxplots" not in exclude_plots
        self.exclude_plots = exclude_plots

        self.export_feature_correlations = "feature_correlations" not in exclude_plots
        self.show_plots = show_plots
        self.cv_partitions = cv_partitions

        self.categorical_cutoff = categorical_cutoff
        self.sig_cutoff = sig_cutoff
        self.scale_data = scale_data
        self.impute_data = impute_data
        self.scoring_metric = scoring_metric
        self.multi_impute = multi_impute
        self.ignore_features = ignore_features or []
        self.random_state = random_state

        self.train_name = Path(self.full_path).name
        self.experiment_path = str(Path(self.full_path).parent)
        self.apply_name = Path(self.dataset_filename).stem  # replication dataset name

    # ------------------------------------------------------------------ #
    # Main run
    # ------------------------------------------------------------------ #

    def run(self):
        # Load replication dataset
        rep_data = Dataset(
            self.dataset_filename,
            self.outcome_label,
            self.match_label,
            self.instance_label,
        )
        rep_feature_list = list(rep_data.data.columns)
        rep_feature_list.remove(self.outcome_label)
        if self.match_label is not None and self.match_label in rep_feature_list:
            rep_feature_list.remove(self.match_label)
        if self.instance_label is not None and self.instance_label in rep_feature_list:
            rep_feature_list.remove(self.instance_label)

        # Load original training dataset (used to enforce feature alignment)
        train_data = Dataset(
            self.dataset_for_rep,
            self.outcome_label,
            self.match_label,
            self.instance_label,
        )
        all_train_feature_list = list(train_data.data.columns)
        all_train_feature_list.remove(self.outcome_label)
        if self.match_label is not None and self.match_label in all_train_feature_list:
            all_train_feature_list.remove(self.match_label)
        if self.instance_label is not None and self.instance_label in all_train_feature_list:
            all_train_feature_list.remove(self.instance_label)

        # Check feature coverage
        if not set(all_train_feature_list).issubset(rep_feature_list):
            raise Exception(
                "Error: One or more features in training dataset did not appear in replication dataset!"
            )

        # Order replication columns to match training columns exactly
        rep_data.data = rep_data.data[train_data.data.columns]

        # Create basic folder hierarchy for replication outputs
        rep_root = Path(self.full_path) / "replication" / self.apply_name
        exploratory_dir = rep_root / "exploratory" / "initial"
        model_eval_dir = rep_root / "model_evaluation" / "pickled_metrics"
        exploratory_dir.mkdir(parents=True, exist_ok=True)
        model_eval_dir.mkdir(parents=True, exist_ok=True)

        # Load categorical / quantitative lists from training
        with open(
            os.path.join(self.full_path, "exploratory", "initial", "initial_categorical_features.pickle"),
            "rb",
        ) as f:
            categorical_variables = pickle.load(f)
        with open(
            os.path.join(self.full_path, "exploratory", "initial", "initial_quantitative_features.pickle"),
            "rb",
        ) as f:
            quantitative_variables = pickle.load(f)

        rep_data.categorical_variables = categorical_variables
        rep_data.quantitative_variables = quantitative_variables

        # EDA / data process, reusing training decisions
        eda = DataProcess(
            rep_data,
            self.full_path,
            ignore_features=self.ignore_features,
            categorical_features=categorical_variables,
            quantitative_features=quantitative_variables,
            exclude_eda_output=None,
            categorical_cutoff=self.categorical_cutoff,
            sig_cutoff=self.sig_cutoff,
            random_state=self.random_state,
            show_plots=self.show_plots,
        )
        eda.dataset.name = f"replication/{self.apply_name}"

        eda.identify_feature_types()
        n_class = len(eda.counts_summary(save=False)) - 6

        if self.outcome_type == "Binary":
            transition_cols = [
                "Instances",
                "Total Features",
                "Categorical Features",
                "Quantitative Features",
                "Missing Values",
                "Missing Percent",
                "Class 0",
                "Class 1",
            ]
        elif self.outcome_type == "Multiclass":
            transition_cols = [
                "Instances",
                "Total Features",
                "Categorical Features",
                "Quantitative Features",
                "Missing Values",
                "Missing Percent",
            ] + [f"Class {i}" for i in range(n_class)]
        else:  # Continuous
            transition_cols = [
                "Instances",
                "Total Features",
                "Categorical Features",
                "Quantitative Features",
                "Missing Values",
                "Missing Percent",
            ]

        transition_df = pd.DataFrame(columns=transition_cols)
        transition_df.loc["Original"] = eda.counts_summary(save=False)

        # Binary categorical consistency check
        with open(
            os.path.join(self.full_path, "exploratory", "binary_categorical_dict.pickle"), "rb"
        ) as f:
            binary_categorical_dict = dict(pickle.load(f))

        for key, train_vals in binary_categorical_dict.items():
            if key not in eda.dataset.data.columns:
                continue
            unique_vals = [x for x in eda.dataset.data[key].unique() if not pd.isnull(x)]
            if sorted(unique_vals) != sorted(train_vals):
                new_values = list(set(unique_vals) - set(train_vals))
                logging.warning(
                    "New Value found in Binary Categorical Variable %s, replacing with NaN", key
                )
                for feat in new_values:
                    logging.warning("\t%s", feat)
                eda.dataset.data[key].replace(new_values, np.nan, inplace=True)

        # Ordinal encoding alignment with training
        self._apply_ordinal_encoding(eda)

        # Baseline clean-up
        eda.drop_ignored_rowcols()
        transition_df.loc["C1"] = eda.counts_summary(save=False)

        eda.dataset.initial_eda(os.path.join(self.experiment_path, self.train_name))

        # ------------------------------------------------------------------
        # Feature engineering & post-processing replay
        # ------------------------------------------------------------------

        # engineered_features: usually training-phase engineered features (e.g., missingness, etc.)
        try:
            with open(
                os.path.join(self.full_path, "exploratory", "engineered_features.pickle"), "rb"
            ) as f:
                eda.engineered_features = pickle.load(f)
        except FileNotFoundError:
            eda.engineered_features = []

        # Recreate missingness features
        for feat in eda.engineered_features:
            if feat in eda.dataset.data.columns:
                eda.dataset.data["Miss_" + feat] = eda.dataset.data[feat].isnull().astype(int)
                eda.categorical_features.append("Miss_" + feat)
        eda.engineered_features = ["Miss_" + feat for feat in eda.engineered_features]

        # Remove features that were dropped in training
        try:
            with open(
                os.path.join(self.full_path, "exploratory", "removed_features.pickle"),
                "rb",
            ) as f:
                removed_features = list(pickle.load(f))
            for feat in removed_features:
                if feat in eda.categorical_features:
                    eda.categorical_features.remove(feat)
                if feat in eda.quantitative_features:
                    eda.quantitative_features.remove(feat)
                if feat in eda.dataset.data.columns:
                    eda.dataset.data.drop(feat, axis=1, inplace=True)
        except FileNotFoundError:
            removed_features = []

        # Load post-processed variable list (final train variables)
        with open(
            os.path.join(self.full_path, "exploratory", "post_processed_features.pickle"),
            "rb",
        ) as f:
            post_processed_vars = pickle.load(f)

        # One-hot encode multi-level categoricals
        non_binary_categorical = [
            feat
            for feat in eda.categorical_features
            if feat in eda.dataset.data.columns and eda.dataset.data[feat].nunique() > 2
        ]
        if non_binary_categorical:
            one_hot_df = pd.get_dummies(
                eda.dataset.data[non_binary_categorical],
                columns=non_binary_categorical,
            )
            eda.one_hot_features = list(one_hot_df.columns)
            eda.dataset.data.drop(non_binary_categorical, axis=1, inplace=True)
            eda.dataset.data = pd.concat([eda.dataset.data, one_hot_df], axis=1)
        else:
            eda.one_hot_features = []

        # Ensure all one-hot features from training are present
        for feat in post_processed_vars:
            if feat not in eda.dataset.data.columns:
                eda.dataset.data[feat] = 0
                eda.one_hot_features.append(feat)

        eda.categorical_features += eda.one_hot_features

        # Load correlated features that were removed
        try:
            with open(
                os.path.join(self.full_path, "exploratory", "correlated_features.pickle"),
                "rb",
            ) as f:
                correlated_features = list(pickle.load(f))
        except FileNotFoundError:
            correlated_features = []

        # Drop any extra features not in final train variable set or correlated features
        for feat in list(eda.dataset.data.columns):
            if feat not in post_processed_vars and feat not in correlated_features:
                eda.drop_ignored_rowcols([feat])

        # Remove correlated features
        for feat in correlated_features:
            if feat in eda.categorical_features:
                eda.categorical_features.remove(feat)
            if feat in eda.quantitative_features:
                eda.quantitative_features.remove(feat)
            if feat in eda.dataset.data.columns:
                eda.dataset.data.drop(feat, axis=1, inplace=True)

        eda.categorical_features = list(set(post_processed_vars).intersection(eda.categorical_features))
        eda.quantitative_features = list(set(post_processed_vars).intersection(eda.quantitative_features))

        # Final alignment with train post-processed vars
        eda.dataset.data = eda.dataset.data[post_processed_vars]
        transition_df.loc["R1"] = eda.counts_summary(save=False)

        rep_exploratory = rep_root / "exploratory"
        rep_exploratory.mkdir(exist_ok=True)
        transition_df.to_csv(
            rep_exploratory / "DataProcessSummary.csv",
            index=True,
        )

        # Persist categorical + post-processed features for replication dataset
        with open(rep_exploratory / "categorical_features.pickle", "wb") as f:
            pickle.dump(eda.categorical_features, f)

        with open(rep_exploratory / "post_processed_features.pickle", "wb") as f:
            pickle.dump(list(eda.dataset.data.columns), f)

        with open(rep_exploratory / "ProcessedFeatureNames.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(eda.dataset.data.columns))

        # Save processed replication dataset (used by notebooks / predictions)
        eda.dataset.data.to_csv(
            rep_root / f"{self.apply_name}_Processed.csv", index=False
        )

        # EDA outputs (describe, missingness, correlation)
        eda.dataset.describe_data(os.path.join(self.experiment_path, self.train_name))
        total_missing = eda.dataset.missingness_counts(os.path.join(self.experiment_path, self.train_name))
        eda.counts_summary(total_missing, plot=True, replicate=True)

        x_rep_data = eda.dataset.feature_only_data()
        if self.export_feature_correlations:
            eda.dataset.feature_correlation(
                os.path.join(self.experiment_path, self.train_name),
                x_rep_data,
                show_plots=False,
            )
        del x_rep_data

        # ------------------------------------------------------------------
        # Apply CV-specific pipeline: Imputation, Scaling, Feature selection
        # ------------------------------------------------------------------
        master_list: List[Dict[str, Any]] = []
        cv_dataset_paths = glob.glob(os.path.join(self.full_path, "CVDatasets", "*_CV_*Train.csv"))
        cv_dataset_paths = [str(Path(p)) for p in cv_dataset_paths]
        cv_partitions = len(cv_dataset_paths)

        for cv_count in range(cv_partitions):
            cv_train_path = os.path.join(
                self.full_path,
                "CVDatasets",
                f"{self.train_name}_CV_{cv_count}_Train.csv",
            )
            cv_train_data = pd.read_csv(cv_train_path, sep=",", na_values="NA")

            train_feature_list = list(cv_train_data.columns)
            train_feature_list.remove(self.outcome_label)
            if self.instance_label is not None and self.instance_label in train_feature_list:
                train_feature_list.remove(self.instance_label)
            if self.match_label is not None and self.match_label in train_feature_list:
                train_feature_list.remove(self.match_label)

            cv_rep_data = rep_data.data.copy()

            feature_name_list = list(post_processed_vars)
            if self.outcome_label in feature_name_list:
                feature_name_list.remove(self.outcome_label)
            if self.instance_label and self.instance_label in feature_name_list:
                feature_name_list.remove(self.instance_label)
            if self.match_label and self.match_label in feature_name_list:
                feature_name_list.remove(self.match_label)

            # Impute
            if self.impute_data:
                try:
                    cv_rep_data = self.impute_rep_data(
                        cv_count,
                        cv_rep_data,
                        feature_name_list,
                        eda.categorical_features,
                        eda.quantitative_features,
                    )
                except Exception as e:
                    logging.warning("Unknown Exception in Imputation for %s", self.apply_name)
                    logging.warning(e)

            # Scale
            if self.scale_data:
                try:
                    cv_rep_data = self.scale_rep_data(cv_count, cv_rep_data, feature_name_list)
                except Exception as e:
                    logging.warning(
                        "Notice: Scaling was not conducted for training dataset for %s, "
                        "so scaling was not applied to replication data.",
                        self.apply_name,
                    )
                    logging.debug("Scaling error: %s", e)

            # Feature selection: keep only training CV columns
            cv_rep_data = cv_rep_data[cv_train_data.columns]
            del cv_train_data

            if self.instance_label is not None and self.instance_label in cv_rep_data.columns:
                cv_rep_data = cv_rep_data.drop(self.instance_label, axis=1)

            x_test = cv_rep_data.drop(self.outcome_label, axis=1).values
            y_test = cv_rep_data[self.outcome_label].values

            eval_dict: Dict[str, Any] = {}
            for algorithm in self.algorithms:
                if self.outcome_type in ("Binary", "Multiclass"):
                    ret = self.eval_model(algorithm, cv_count, x_test, y_test)
                else:
                    ret, residuals = self.eval_model(algorithm, cv_count, x_test, y_test)
                eval_dict[algorithm] = ret

                out_pkl = (
                    model_eval_dir
                    / f"{self.abbrev[algorithm]}_CV_{cv_count}_metrics.pickle"
                )
                with open(out_pkl, "wb") as f:
                    pickle.dump(ret, f)

            master_list.append(eval_dict)

        # ------------------------------------------------------------------
        # Phase 8-style statistics on replication data
        # ------------------------------------------------------------------
        stats = StatsJob(
            str(rep_root),
            self.outcome_label,
            self.instance_label,
            self.scoring_metric,
            cv_partitions=self.cv_partitions,
            top_features=40,
            sig_cutoff=self.sig_cutoff,
            metric_weight="balanced_accuracy",
            scale_data=self.scale_data,
            exclude_plots=self.exclude_plots,
            show_plots=self.show_plots,
        )

        if self.outcome_type == "Binary":
            result_table, metric_dict = stats.primary_stats_classification(
                master_list, rep_data.data
            )
            if self.plot_roc:
                stats.do_plot_roc(result_table)
            if self.plot_prc:
                stats.do_plot_prc(result_table, rep_data.data, True)
        elif self.outcome_type == "Multiclass":
            result_table, metric_dict = stats.primary_stats_multiclass(
                master_list, rep_data.data
            )
            if self.plot_roc:
                stats.do_plot_roc(result_table)
            if self.plot_prc:
                stats.do_plot_prc(result_table, rep_data.data, True)
        else:  # Continuous
            result_table, metric_dict = stats.primary_stats_regression(master_list)
            # TODO: if you have new residuals / regression plots in p8, call them here

        metrics = list(metric_dict[self.algorithms[0]].keys())
        stats.save_metric_stats(metrics, metric_dict)

        if self.plot_metric_boxplots:
            stats.metric_boxplots(metrics, metric_dict)

        # Inter-algorithm non-parametric tests if >1 algorithm
        if len(self.algorithms) > 1:
            kruskal_summary = stats.kruskal_wallis(metrics, metric_dict)
            stats.mann_whitney_u(metrics, metric_dict, kruskal_summary)
            stats.wilcoxon_rank(metrics, metric_dict, kruskal_summary)

        # TODO: if your new P8 summary supports ensembles on replication sets,
        #       this is where you’d extend master_list / metric_dict, or run
        #       a separate ensemble StatsJob.

        logging.info("%s replication phase complete", self.apply_name)
        jobs_completed_dir = Path(self.experiment_path) / "jobsCompleted"
        jobs_completed_dir.mkdir(exist_ok=True)
        with open(jobs_completed_dir / f"job_apply_{self.apply_name}.txt", "w") as f:
            f.write("complete")

    # ------------------------------------------------------------------ #
    # Helpers (imputation / scaling / eval) - mostly old logic
    # ------------------------------------------------------------------ #

    def impute_rep_data(
        self,
        cv_count: int,
        cv_rep_data: pd.DataFrame,
        all_train_feature_list: List[str],
        cat_features: List[str],
        quant_features: List[str],
    ) -> pd.DataFrame:
        # Categorical imputation
        try:
            impute_cat_info = os.path.join(
                self.full_path, "scale_impute", f"categorical_imputer_cv{cv_count}.pickle"
            )
            with open(impute_cat_info, "rb") as f:
                mode_dict = pickle.load(f)
            for c in cv_rep_data.columns:
                if c in mode_dict:
                    cv_rep_data[c].fillna(mode_dict[c], inplace=True)
        except Exception:
            if cv_rep_data.isna().sum().sum() > 0:
                logging.warning(
                    "Categorical imputation pickle missing; imputing medians for %s",
                    self.apply_name,
                )
                for feat in cat_features:
                    if feat in cv_rep_data.columns and cv_rep_data[feat].isnull().sum() > 0:
                        cv_rep_data[feat].fillna(cv_rep_data[feat].median(), inplace=True)

        impute_rep_df: Optional[pd.DataFrame] = None

        try:
            impute_ordinal_info = os.path.join(
                self.full_path, "scale_impute", f"ordinal_imputer_cv{cv_count}.pickle"
            )
            if self.multi_impute:
                with open(impute_ordinal_info, "rb") as f:
                    imputer = pickle.load(f)

                if self.instance_label is None or self.instance_label == "None":
                    x_rep = cv_rep_data.drop([self.outcome_label], axis=1).values
                    inst_rep = None
                else:
                    x_rep = cv_rep_data.drop(
                        [self.outcome_label, self.instance_label], axis=1
                    ).values
                    inst_rep = cv_rep_data[self.instance_label].values

                y_rep = cv_rep_data[self.outcome_label].values
                x_rep_impute = imputer.transform(x_rep)

                if self.instance_label is None or self.instance_label == "None":
                    impute_rep_df = pd.concat(
                        [
                            pd.DataFrame(y_rep, columns=[self.outcome_label]),
                            pd.DataFrame(x_rep_impute, columns=all_train_feature_list),
                        ],
                        axis=1,
                        sort=False,
                    )
                else:
                    impute_rep_df = pd.concat(
                        [
                            pd.DataFrame(y_rep, columns=[self.outcome_label]),
                            pd.DataFrame(inst_rep, columns=[self.instance_label]),
                            pd.DataFrame(x_rep_impute, columns=all_train_feature_list),
                        ],
                        axis=1,
                        sort=False,
                    )
            else:
                with open(impute_ordinal_info, "rb") as f:
                    median_dict = pickle.load(f)
                for c in cv_rep_data.columns:
                    if c in median_dict:
                        cv_rep_data[c].fillna(median_dict[c], inplace=True)
        except FileNotFoundError:
            if cv_rep_data.isna().sum().sum() > 0:
                logging.warning(
                    "Quantitative imputation pickle missing; imputing means for %s",
                    self.apply_name,
                )
                for feat in quant_features:
                    if feat in cv_rep_data.columns and cv_rep_data[feat].isnull().sum() > 0:
                        cv_rep_data[feat].fillna(cv_rep_data[feat].mean(), inplace=True)
            impute_rep_df = cv_rep_data

        return impute_rep_df

    def scale_rep_data(
        self,
        cv_count: int,
        cv_rep_data: pd.DataFrame,
        all_train_feature_list: List[str],
    ) -> pd.DataFrame:
        scale_info = os.path.join(
            self.full_path, "scale_impute", f"scaler_cv{cv_count}.pickle"
        )
        with open(scale_info, "rb") as f:
            scaler = pickle.load(f)
        decimal_places = 7

        if self.instance_label is None or self.instance_label == "None":
            x_rep = cv_rep_data.drop([self.outcome_label], axis=1)
            inst_rep = None
        else:
            x_rep = cv_rep_data.drop([self.outcome_label, self.instance_label], axis=1)
            inst_rep = cv_rep_data[self.instance_label]

        y_rep = cv_rep_data[self.outcome_label]
        x_rep_scaled = pd.DataFrame(
            scaler.transform(x_rep).round(decimal_places),
            columns=x_rep.columns,
        )

        if self.instance_label is None or self.instance_label == "None":
            scale_rep_df = pd.concat(
                [
                    pd.DataFrame(y_rep, columns=[self.outcome_label]),
                    pd.DataFrame(x_rep_scaled, columns=all_train_feature_list),
                ],
                axis=1,
                sort=False,
            )
        else:
            scale_rep_df = pd.concat(
                [
                    pd.DataFrame(y_rep, columns=[self.outcome_label]),
                    pd.DataFrame(inst_rep, columns=[self.instance_label]),
                    pd.DataFrame(x_rep_scaled, columns=all_train_feature_list),
                ],
                axis=1,
                sort=False,
            )
        return scale_rep_df

    def eval_model(
        self,
        algorithm: str,
        cv_count: int,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ):
        model_info = os.path.join(
            self.full_path,
            "models",
            "pickledModels",
            f"{self.abbrev[algorithm]}_{cv_count}.pickle",
        )
        with open(model_info, "rb") as f:
            model = pickle.load(f)

        if self.outcome_type == "Binary":
            m = BinaryClassificationModel(None, algorithm, scoring_metric=self.scoring_metric)
        elif self.outcome_type == "Multiclass":
            m = MulticlassClassificationModel(None, algorithm, scoring_metric=self.scoring_metric)
        else:
            m = RegressionModel(None, algorithm, scoring_metric=self.scoring_metric)

        m.model = model
        m.model_name = algorithm
        m.small_name = self.abbrev[algorithm]

        if self.outcome_type == "Continuous":
            metric_list = m.model_evaluation(x_test, y_test)
            y_pred = m.predict(x_test)
            residual_test = y_test - y_pred
            return ([metric_list, None], [residual_test, y_pred, y_test])
        else:
            metric_list, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_ = m.model_evaluation(
                x_test, y_test
            )
            return [
                metric_list,
                fpr,
                tpr,
                roc_auc,
                prec,
                recall,
                prec_rec_auc,
                ave_prec,
                None,
                probas_,
            ]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _apply_ordinal_encoding(self, eda: DataProcess) -> None:
        """
        Replays the ordinal encoding learned during training onto the replication dataset,
        including handling of new levels.
        """
        try:
            ord_path = os.path.join(
                self.full_path,
                "exploratory",
                "ordinal_encoding.pickle",
            )
            with open(ord_path, "rb") as f:
                ord_labels = pickle.load(f)

            for feat in ord_labels.index:
                if feat not in eda.dataset.data.columns:
                    continue
                temp_y, labels = pd.factorize(eda.dataset.data[feat])

                if set(ord_labels.loc[feat]["Category"]) == set(labels):
                    eda.dataset.data[feat] = temp_y
                elif len(ord_labels.loc[feat]["Category"]) == 2:
                    new_labels = list(set(labels) - set(ord_labels.loc[feat]["Category"]))
                    labels = ord_labels.loc[feat]["Category"]
                    rename_dict = dict(enumerate(labels))
                    for lab in new_labels:
                        rename_dict[None] = lab
                    rename_dict = {v: k for k, v in rename_dict.items()}
                    eda.dataset.data.replace({feat: rename_dict}, inplace=True)
                    ord_labels.loc[feat]["Category"] = list(labels) + new_labels
                    ord_labels.loc[feat]["Encoding"] = list(range(len(labels))) + [None] * len(
                        new_labels
                    )
                    logging.warning(
                        "New Value found in Textual Binary Categorical Variable %s; replaced with None encoding",
                        feat,
                    )
                    for x in new_labels:
                        logging.warning("\t%s", x)
                else:
                    new_labels = list(set(labels) - set(ord_labels.loc[feat]["Category"]))
                    labels = ord_labels.loc[feat]["Category"]
                    rename_dict = dict(enumerate(list(labels) + new_labels))
                    rename_dict = {v: k for k, v in rename_dict.items()}
                    eda.dataset.data.replace({feat: rename_dict}, inplace=True)
                    ord_labels.loc[feat]["Category"] = list(labels) + new_labels
                    ord_labels.loc[feat]["Encoding"] = list(
                        range(len(list(labels) + new_labels))
                    )

            rep_exploratory = (
                Path(self.full_path) / "replication" / self.apply_name / "exploratory"
            )
            rep_exploratory.mkdir(parents=True, exist_ok=True)
            with open(rep_exploratory / "apply_ordinal_encoding.pickle", "wb") as f:
                pickle.dump(ord_labels, f)
            ord_labels.to_csv(
                rep_exploratory / "Numerical_Encoding_Map.csv",
            )
        except FileNotFoundError:
            # No ordinal encoding used in training
            return
