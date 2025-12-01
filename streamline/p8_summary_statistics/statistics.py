from __future__ import annotations

import csv
import glob
import os
import re
import pickle
import time
import json
import logging
from pathlib import Path
from statistics import mean, median, stdev
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics import auc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from streamline.p8_summary_statistics.utils.plot_curves import (
    plot_model_roc,
    plot_model_prc,
    plot_summary_roc,
    plot_summary_prc,
    plot_metric_boxplots,
    plot_ensemble_roc_summary,
    plot_ensemble_prc_summary,
)
from streamline.p8_summary_statistics.utils.plot_fi import (
    plot_fi_boxplots,
    plot_fi_histogram,
    plot_composite_fi,
)
from streamline.p8_summary_statistics.utils.fi_core import (
    prep_fi,
    select_for_composite_viz,
    get_fi_to_viz_sorted,
    frac_fi,
    weight_fi,
    weight_frac_fi,
)
from streamline.p8_summary_statistics.utils.plot_regression import residuals_regression


from scipy import stats
from scipy.stats import kruskal, wilcoxon, mannwhitneyu
from streamline.p6_modeling.utils.loader import list_models, get_model_by_id

import seaborn as sns

sns.set_theme()
logger = logging.getLogger(__name__)


class StatisticsPhaseJob:
    """
    Phase 8: Statistics & post-analysis summary for STREAMLINE3.

    This is the modernized version of the legacy StatsJob, adapted so that:
      * algorithms are discovered from model_evaluation/pickled_metrics
      * ensemble summaries can be added on top (if present from Phase 7)
    """

    def __init__(
        self,
        full_path: str,
        outcome_label: str,
        outcome_type: str,
        instance_label: Optional[str],
        scoring_metric: str = "balanced_accuracy",
        cv_partitions: int = 5,
        top_features: int = 40,
        sig_cutoff: float = 0.05,
        metric_weight: str = "balanced_accuracy",
        scale_data: bool = True,
        exclude_plots: Optional[List[str]] = None,
        show_plots: bool = False,
        include_ensembles: bool = True,
        multiclass_average: str = "micro",
    ):
        """
        Args:
            full_path: path to dataset dir: <output>/<experiment>/<dataset>
            outcome_label: column name of outcome (e.g. 'Class')
            outcome_type: 'Binary' | 'Multiclass' | 'Continuous'
            instance_label: e.g. 'InstanceID' or None
            scoring_metric: sklearn metric name used in modeling
            cv_partitions: number of CV splits
            top_features: number of top features for FI visualisations
            sig_cutoff: alpha for Kruskal / Wilcoxon / Mann-Whitney
            metric_weight: metric used for composite FI weighting
            scale_data: kept for API parity (not used directly here)
            exclude_plots: list of strings from
                           ['plot_ROC', 'plot_PRC', 'plot_FI_box', 'plot_metric_boxplots']
            show_plots: whether to show figures interactively
            include_ensembles: if True, also summarize Phase 7 ensemble metrics if present
        """
        self.full_path = full_path
        self.outcome_label = outcome_label
        self.outcome_type = outcome_type
        self.instance_label = instance_label
        self.data_name = self.full_path.split("/")[-1]
        self.experiment_path = "/".join(self.full_path.split("/")[:-1])
        self.cv_partitions = cv_partitions
        self.scale_data = scale_data
        self.scoring_metric = scoring_metric
        self.top_features = top_features
        self.sig_cutoff = sig_cutoff
        self.metric_weight = metric_weight
        self.show_plots = show_plots
        self.include_ensembles = include_ensembles
        
        # multiclass averaging choice
        self.multiclass_average = multiclass_average
        if self.outcome_type == "Multiclass" and self.multiclass_average not in ("micro", "macro"):
            raise ValueError(
                f"multiclass_average must be 'micro' or 'macro', got {self.multiclass_average!r}"
            )
            
        # Plot exclusions
        known_exclude_options = [
            "plot_ROC",
            "plot_PRC",
            "plot_FI_box",
            "plot_metric_boxplots",
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
        self.plot_fi_box = "plot_FI_box" not in exclude_plots

        # Map metric_weight from sklearn name to human-friendly text used in plots
        if self.outcome_type == "Continuous" and (
            self.scoring_metric == "balanced_accuracy"
            or self.metric_weight == "balanced_accuracy"
        ):
            self.metric_weight = "explained_variance"
            self.scoring_metric = "explained_variance"

        if self.outcome_type == "Binary":
            metric_term_dict = {
                "balanced_accuracy": "Balanced Accuracy",
                "accuracy": "Accuracy",
                "f1": "F1 Score",
                "recall": "Sensitivity (Recall)",
                "precision": "Precision (PPV)",
                "roc_auc": "ROC AUC",
            }
        elif self.outcome_type == "Continuous":
            metric_term_dict = {
                "max_error": "Max Error",
                "mean_absolute_error": "Mean Absolute Error",
                "mean_squared_error": "Mean Squared Error",
                "median_absolute_error": "Median Absolute Error",
                "explained_variance": "Explained Variance",
                "pearson_correlation": "Pearson Correlation",
                "f1": "F1 Score",
            }
        elif self.outcome_type == "Multiclass":
            metric_term_dict = {
                "balanced_accuracy": "Balanced Accuracy",
                "accuracy": "Accuracy",
                "f1": "F1 Score",
                "recall": "Sensitivity (Recall)",
                "precision": "Precision (PPV)",
                "roc_auc": "ROC AUC",
            }
        else:
            raise ValueError(f"Unknown outcome_type: {self.outcome_type}")

        self.metric_weight = metric_term_dict[self.metric_weight]

        # Prepare feature headers
        if self.plot_fi_box:
            self.feature_headers = pd.read_csv(
                self.full_path + "/exploratory/ProcessedFeatureNames.csv", sep=","
            ).columns.values.tolist()
            self.original_headers = self.feature_headers
        else:
            try:
                self.feature_headers = pd.read_csv(
                    self.full_path + "/exploratory/ProcessedFeatureNames.csv", sep=","
                ).columns.values.tolist()
                self.original_headers = self.feature_headers
            except Exception:
                self.original_headers = None
                self.feature_headers = None

        # NEW: discover algorithms purely from model_evaluation outputs
        (
            self.algorithms,
            self.abbrev,
            self.colors,
        ) = self._discover_algorithms_from_metrics()

        if not self.algorithms:
            logging.warning(
                "No algorithms discovered in %s; stats will be limited.",
                self.full_path,
            )

    # ------------------------------------------------------------------
    # NEW: Algorithm discovery
    # ------------------------------------------------------------------
    def _discover_algorithms_from_metrics(
        self,
    ) -> Tuple[List[str], Dict[str, str], Dict[str, Tuple[float, float, float]]]:
        """
        Discover modeling algorithms for statistics from:
            <dataset_dir>/model_evaluation/pickled_metrics/<ALG>_CV_<k>_metrics.pickle

        Returns:
            algorithms: list of small_names (e.g. "LR", "SVM")
            abbrev: mapping algorithm -> abbrev used in file names (here same as algorithm)
            colors: mapping algorithm -> RGB triple for plotting
        """
        metrics_dir = Path(self.full_path) / "model_evaluation" / "metrics_by_cv"
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
            logging.warning(
                "StatsPhaseJob: no modeling metrics found under %s", metrics_dir
            )

        # Registry-driven discovery
        algorithms: List[str] = []
        abbrev: Dict[str, str] = {}
        colors: Dict[str, Tuple[float, float, float]] = {}

        registry_entries = []
        if list_models is not None:
            try:
                registry_entries = list_models(self.outcome_type)
            except Exception as e:
                logging.warning("StatsPhaseJob: list_all_models() failed: %r", e)

        # Build a quick lookup: small_name -> (model_type, entry)
        entries_by_small = {}
        for entry in registry_entries:
            small = (entry.get("small_name") or "").strip()
            mt = (entry.get("model_type") or "").strip()
            if small:
                entries_by_small[small] = (mt, entry)

        for alg in sorted(present_set):
            mt, entry = entries_by_small.get(alg, ("", {}))
            cls = None
            if get_model_by_id is not None and mt:
                # Try resolving the model class to read color attribute
                try:
                    cls = get_model_by_id(mt, alg)  # small_name
                except Exception:
                    # Fallback: try model_name / alt_id if small_name lookup fails
                    model_name = entry.get("model_name") or entry.get("alt_id") or alg
                    try:
                        cls = get_model_by_id(mt, model_name)
                    except Exception:
                        cls = None
                        
            alg_name = cls.model_name if cls is not None else alg
            algorithms.append(alg_name)
            abbrev[alg_name] = cls.small_name if cls is not None else alg

            if cls is not None and hasattr(cls, "color"):
                colors[alg_name] = cls.color  # expect either named color or RGB tuple

        # -----------------------------
        # Color fallback using seaborn
        # -----------------------------
        if algorithms:
            palette = sns.color_palette("tab10", n_colors=len(algorithms))
            for i, alg in enumerate(algorithms):
                if alg not in colors:
                    colors[alg] = palette[i % len(palette)]
        else:
            # Keep old behavior (empty but valid return)
            algorithms = sorted(present_set)
            abbrev = {a: a for a in algorithms}
            palette = sns.color_palette("tab10", n_colors=max(len(algorithms), 1))
            colors = {a: palette[i % len(palette)] for i, a in enumerate(algorithms)}

        return algorithms, abbrev, colors

    # ------------------------------------------------------------------
    # PUBLIC ENTRY
    # ------------------------------------------------------------------
    def run(self):
        self.job_start_time = time.time()
        logging.info("Running Statistics Summary for %s", self.data_name)

        # Ensure dirs exist
        self.preparation()

        # Core stats for base models (phase 6)
        if self.outcome_type == "Binary":
            result_table, metric_dict = self.primary_stats_classification()
        elif self.outcome_type == "Multiclass":
            result_table, metric_dict = self.primary_stats_multiclass()
        elif self.outcome_type == "Continuous":
            result_table, metric_dict = self.primary_stats_regression()
        else:
            raise ValueError(f"Unknown outcome_type: {self.outcome_type}")
        
        # Summary ROC / PRC across algorithms
        if self.outcome_type in ("Binary", "Multiclass"):
            if self.plot_roc:
                plot_summary_roc(
                    full_path=self.full_path,
                    colors=self.colors,
                    result_table=result_table,
                    show_plots=self.show_plots,
                )
            if self.plot_prc:
                plot_summary_prc(
                    full_path=self.full_path,
                    colors=self.colors,
                    result_table=result_table,
                    outcome_label=self.outcome_label,
                    data_name=self.data_name,
                    instance_label=self.instance_label,
                    rep_data=None,
                    replicate=False,
                    show_plots=self.show_plots,
                )
        else:
            # Regression residual plots
            residuals_regression(
                full_path=self.full_path,
                algorithms=self.algorithms,
                abbrev=self.abbrev,
                cv_partitions=self.cv_partitions,
                colors=self.colors,
                show_plots=self.show_plots,
            )

        # Summaries of metrics across CV folds
        metrics = list(metric_dict[self.algorithms[0]].keys())
        logging.info("Saving Metric Summaries...")
        self.save_metric_stats(metrics, metric_dict)

        # Metric boxplots
        if self.plot_metric_boxplots:
            logging.info("Generating Metric Boxplots...")
            plot_metric_boxplots(
                full_path=self.full_path,
                algorithms=self.algorithms,
                metrics=metrics,
                metric_dict=metric_dict,
                show_plots=self.show_plots,
            )

        # Non-parametric tests (Kruskal, Wilcoxon, Mann-Whitney)
        if len(self.algorithms) > 1:
            logging.info(
                "Running Non-Parametric Statistical Significance Analysis..."
            )
            kruskal_summary = self.kruskal_wallis(metrics, metric_dict)
            self.wilcoxon_rank(metrics, metric_dict, kruskal_summary)
            self.mann_whitney_u(metrics, metric_dict, kruskal_summary)

        # Feature-importance stats & plots
        ave_or_median = (
            "median" if self.outcome_type in ("Binary", "Multiclass") else "mean"
        )
        self.fi_stats(metric_dict, ave_or_median)

        # Optional: summarize ensembles (Phase 7) if present
        if self.include_ensembles:
            try:
                self.ensemble_stats_summary()
            except Exception as e:
                logging.warning(
                    "Ensemble summary failed (non-fatal): %s", str(e)
                )

        # Save runtime for this phase
        self.save_runtime()
        self.parse_runtime()

        logging.info("%s statistics phase complete", self.data_name)
        job_file = open(
            self.experiment_path
            + "/jobsCompleted/job_stats_"
            + self.data_name
            + ".txt",
            "w",
        )
        job_file.write("complete")
        job_file.close()

    def preparation(self):
        """
        Creates directory for all results files, decodes included ML modeling
        algorithms that were run
        """
        if not os.path.exists(self.full_path + "/model_evaluation"):
            os.mkdir(self.full_path + "/model_evaluation")
        if not os.path.exists(self.full_path + "/model_evaluation/feature_importance/"):
            os.mkdir(self.full_path + "/model_evaluation/feature_importance/")


    def residuals_regression(self, result_file=None):
        s_res_trains = []  # training residual
        s_res_tests = []  # testing residual
        s_y_train_preds = []  # training prediction
        s_y_test_preds = []  # testing prediction
        s_y_trains = []  # training label
        s_y_tests = []  # testing label

        m_trains = []  # slope of training plot
        b_trains = []  # intercept of training plot
        m_tests = []  # slope of testing plot
        b_tests = []  # intercept of testing plot
        for algorithm in self.algorithms:
            s_res_train = []
            s_res_test = []
            s_y_train_pred = []
            s_y_test_pred = []
            s_y_train = []
            s_y_test = []
            for cv_count in range(0, self.cv_partitions):
                if result_file is None:
                    result_file = self.full_path + '/model_evaluation/pickled_metrics/' + self.abbrev[algorithm] \
                                  + "_CV_" + str(cv_count) + "_residuals.pickle"
                file = open(result_file, 'rb')
                results = pickle.load(file)
                file.close()
                # logging.warning(len(results))
                res_train = results[0]
                res_test = results[1]
                y_train_pred = results[2]
                y_test_pred = results[3]
                y_train = results[4]
                y_test = results[5]

                s_res_train = np.stack([res_train], axis=0)
                s_res_test = np.stack([res_test], axis=0)
                s_y_train_pred = np.stack([y_train_pred], axis=0)
                s_y_test_pred = np.stack([y_test_pred], axis=0)
                s_y_train = np.stack([y_train], axis=0)
                s_y_test = np.stack([y_test], axis=0)

            s_res_train = s_res_train[0]
            s_res_test = s_res_test[0]
            s_y_train_pred = s_y_train_pred[0]
            s_y_test_pred = s_y_test_pred[0]
            s_y_train = s_y_train[0]
            s_y_test = s_y_test[0]

            s_res_trains.append(s_res_train)
            s_res_tests.append(s_res_test)
            s_y_train_preds.append(y_train_pred)
            s_y_test_preds.append(y_test_pred)
            s_y_trains.append(y_train)
            s_y_tests.append(s_y_test)

            plt.figure()
            plt.rcdefaults()
            if not os.path.exists(self.full_path + '/model_evaluation/evalPlots'):
                os.mkdir(self.full_path + '/model_evaluation/evalPlots')

            m_1, b_1 = np.polyfit(s_y_train_pred, s_y_train, 1)
            m_2, b_2 = np.polyfit(s_y_test_pred, s_y_test, 1)
            m_trains.append(m_1)
            m_tests.append(m_2)
            b_trains.append(b_1)
            b_tests.append(b_2)

        train_df = []
        test_df = []
        for i in range(len(self.algorithms)):
            df = pd.DataFrame([s_res_trains[i], [self.algorithms[i]] * len(s_res_trains[i]),
                               ["Training"] * len(s_res_trains[i])]).transpose()
            df.columns = ["Residual", "Algorithm", "Type"]
            train_df.append(df)
        train_df = pd.concat(train_df).reset_index(drop=True)
        for i in range(len(self.algorithms)):
            df = pd.DataFrame(
                [s_res_tests[i], [self.algorithms[i]] * len(s_res_tests[i]),
                 ["Testing"] * len(s_res_tests[i])]).transpose()
            df.columns = ["Residual", "Algorithm", "Type"]
            test_df.append(df)
        test_df = pd.concat(test_df).reset_index(drop=True)

        train_df.to_csv(self.full_path + '/model_evaluation/residual_train.csv')
        train_df = pd.read_csv(self.full_path + '/model_evaluation/residual_train.csv')
        test_df.to_csv(self.full_path + '/model_evaluation/residual_test.csv')
        test_df = pd.read_csv(self.full_path + '/model_evaluation/residual_test.csv')

        fig_2, axes_2 = plt.subplots(2, 2, sharey='all', figsize=[20, 15])
        for i in range(len(self.algorithms)):
            axes_2[0, 0].scatter(s_y_train_preds[i], s_res_trains[i], alpha=0.4, c=self.colors[self.algorithms[i]],
                                 label=self.algorithms[i])
            axes_2[1, 0].scatter(s_y_test_preds[i], s_res_tests[i], alpha=0.4, c=self.colors[self.algorithms[i]])

        axes_2[0, 0].axhline(y=0, color='black', linestyle='-')
        axes_2[0, 1].axhline(y=0, color='black', linestyle='-')
        axes_2[1, 0].axhline(y=0, color='black', linestyle='-')
        sns.violinplot(x='Algorithm', y='Residual', data=train_df, color='b', ax=axes_2[0, 1])
        sns.violinplot(x='Algorithm', y='Residual', data=test_df, color='r', ax=axes_2[1, 1])
        axes_2[1, 1].axhline(y=0, color='black', linestyle='-')
        axes_2[0, 0].title.set_text("Residual vs Predicted Outcome (Training)")
        axes_2[1, 0].title.set_text("Residual vs Predicted Outcome (Testing)")
        axes_2[0, 1].title.set_text("Residual Distribution (Training)")
        axes_2[1, 1].title.set_text("Residual Distribution (Testing)")
        axes_2[0, 0].set_ylabel('Residual')
        axes_2[1, 0].set_ylabel('Residual')
        axes_2[1, 0].set_xlabel('Predicted Outcome')
        fig_2.legend(loc='upper right')
        fig_2.savefig(self.full_path + '/model_evaluation/evalPlots/residual_distrib_all_algorithms.png')
        if self.show_plots:
            plt.show()
        else:
            plt.close('all')

        fig_3, axes_3 = plt.subplots(1, 2, sharey='all', figsize=[20, 10])
        for i in range(len(self.algorithms)):
            axes_3[0].scatter(s_y_train_preds[i], s_y_trains[i], alpha=0.3, c=self.colors[self.algorithms[i]])
            axes_3[1].scatter(s_y_test_preds[i], s_y_tests[i], alpha=0.3, c=self.colors[self.algorithms[i]])
            axes_3[0].plot(s_y_train_preds[i], m_trains[i] * s_y_train_preds[i] + b_trains[i],
                           color=self.colors[self.algorithms[i]], label=self.algorithms[i])
            axes_3[1].plot(s_y_test_preds[i], m_tests[i] * s_y_test_preds[i] + b_tests[i],
                           color=self.colors[self.algorithms[i]])
        axes_3[0].title.set_text('Actual Outcome vs. Predicted Outcome (Train)')
        axes_3[1].title.set_text('Actual Outcome vs. Predicted Outcome (Test)')
        axes_3[0].set_ylabel('Actual Outcome')
        axes_3[0].set_xlabel('Predicted Outcome')
        axes_3[1].set_xlabel('Predicted Outcome')
        fig_3.legend(loc='upper right')
        fig_3.savefig(self.full_path + '/model_evaluation/evalPlots/actual_vs_predict_all_algorithms.png')
        if self.show_plots:
            plt.show()
        else:
            plt.close('all')

        fig_4, axes_4 = plt.subplots(1, 1, figsize=(10, 10))
        for i in range(len(self.algorithms)):
            stats.probplot(s_res_trains[i], dist=stats.norm, sparams=(2, 3), plot=plt, fit=False)
        for i in range(len(self.algorithms)):
            axes_4.get_lines()[i].set_markerfacecolor(self.colors[self.algorithms[i]])
            axes_4.get_lines()[i].set_alpha(0.5)
            axes_4.get_lines()[i].set_color(self.colors[self.algorithms[i]])
            axes_4.get_lines()[i].set_label(self.algorithms[i])
        axes_4.title.set_text("Probability Plot of Training Residual")
        axes_4.set_xlabel("Theoretical Quantiles")
        axes_4.set_ylabel("Ordered Residual")
        axes_4.legend(loc='upper right')
        fig_4.savefig(self.full_path + '/model_evaluation/evalPlots/probability_train_residual_all_algorithms.png')
        if self.show_plots:
            plt.show()
        else:
            plt.close('all')

        fig_5, axes_5 = plt.subplots(1, 1, figsize=(10, 10))
        for i in range(len(self.algorithms)):
            stats.probplot(s_res_tests[i], dist=stats.norm, sparams=(2, 3), plot=plt, fit=False)
        for i in range(len(self.algorithms)):
            axes_5.get_lines()[i].set_markerfacecolor(self.colors[self.algorithms[i]])
            axes_5.get_lines()[i].set_alpha(0.5)
            axes_5.get_lines()[i].set_color(self.colors[self.algorithms[i]])
            axes_5.get_lines()[i].set_label(self.algorithms[i])
        axes_5.title.set_text("Probability Plot of Testing Residual")
        axes_5.set_xlabel("Theoretical Quantiles")
        axes_5.set_ylabel("Ordered Residual")
        axes_5.legend(loc='upper right')
        fig_5.savefig(self.full_path + '/model_evaluation/evalPlots/probability_test_residual_all_algorithms.png')
        if self.show_plots:
            plt.show()
        else:
            plt.close('all')

    def fi_stats(self, metric_dict, ave_or_median='mean'):
        metric_ranking = ave_or_median
        metric_weighting = ave_or_median

        logging.info('Preparing for Model Feature Importance Plotting...')

        (
            fi_df_list,
            fi_med_list,
            fi_med_norm_list,
            med_metric_list,
            all_feature_list,
            non_zero_union_features,
            non_zero_union_indexes,
        ) = prep_fi(
            full_path=self.full_path,
            algorithms=self.algorithms,
            abbrev=self.abbrev,
            metric_dict=metric_dict,
            metric_ranking=metric_ranking,
            metric_weighting=metric_weighting,
            metric_weight_name=self.metric_weight,
        )

        # Select 'top' features for composite visualisation
        features_to_viz = select_for_composite_viz(
            non_zero_union_features=non_zero_union_features,
            non_zero_union_indexes=non_zero_union_indexes,
            ave_metric_list=med_metric_list,
            fi_ave_norm_list=fi_med_norm_list,
            algorithms=self.algorithms,
            top_features=self.top_features,
        )

        # per-algorithm FI plots
        if self.plot_fi_box:
            logging.info('Generating Feature Importance Boxplot and Histograms...')
            plot_fi_boxplots(
                full_path=self.full_path,
                algorithms=self.algorithms,
                feature_headers=self.feature_headers,
                fi_df_list=fi_df_list,
                fi_med_list=fi_med_list,
                metric_ranking=metric_ranking,
                show_plots=self.show_plots,
            )
            plot_fi_histogram(
                full_path=self.full_path,
                algorithms=self.algorithms,
                fi_med_list=fi_med_list,
                metric_ranking=metric_ranking,
                show_plots=self.show_plots,
            )

        # composite FI
        logging.info('Generating Composite Feature Importance Plots...')

        # Take top feature names to visualize and get associated feature importance values
        top_fi_med_norm_list, all_feature_list_to_viz = get_fi_to_viz_sorted(
            features_to_viz=features_to_viz,
            all_feature_list=all_feature_list,
            fi_med_norm_list=fi_med_norm_list,
            algorithms=self.algorithms,
        )

        if metric_ranking == 'mean':
            y_label = 'Normalized Mean Feature Importance'
        elif metric_ranking == 'median':
            y_label = 'Normalized Median Feature Importance'
        else:
            raise Exception("Error: metric_ranking selection not found (must be mean or median)")

        # normalized composite FI
        plot_composite_fi(
            full_path=self.full_path,
            algorithms=self.algorithms,
            colors=self.colors,
            fi_list=top_fi_med_norm_list,
            all_feature_list_to_viz=all_feature_list_to_viz,
            fig_name='Norm',
            y_label_text=y_label,
            metric_ranking=metric_ranking,
            metric_weighting=metric_weighting,
            metric_weight_label=self.metric_weight,
            show_plots=self.show_plots,
        )

        # Weighted FI (performance-weighted)
        weighted_lists, weights = weight_fi(
            med_metric_list=med_metric_list,
            top_fi_med_norm_list=top_fi_med_norm_list,
        )

        # Generate Normalized and Weighted Composite FI plot
        if metric_ranking == 'mean':
            y_label_w = 'Normalized and Weighted Mean Feature Importance'
        else:
            y_label_w = 'Normalized and Weighted Median Feature Importance'

        plot_composite_fi(
            full_path=self.full_path,
            algorithms=self.algorithms,
            colors=self.colors,
            fi_list=weighted_lists,
            all_feature_list_to_viz=all_feature_list_to_viz,
            fig_name='Norm_Weight',
            y_label_text=y_label_w,
            metric_ranking=metric_ranking,
            metric_weighting=metric_weighting,
            metric_weight_label=self.metric_weight,
            show_plots=self.show_plots,
        )
        
        # Code comments for fractionated composite FI - commented out for now
        # Fractionated composite FI
        # Weight the Fractionated FI scores for normalized,fractionated, and weighted compound FI plot
        # weighted_frac_lists = weight_frac_fi(frac_lists,weights)

        # Generate Normalized, Fractionated, and Weighted Compound FI plot
        # plot_composite_fi(
        #     full_path=self.full_path,
        #     algorithms=self.algorithms,
        #     colors=self.colors,
        #     fi_list=weighted_frac_lists,
        #     all_feature_list_to_viz=all_feature_list_to_viz,
        #     fig_name='Norm_Frac_Weight',
        #     y_label_text='Normalized, Fractionated, and Weighted Feature Importance',
        #     metric_ranking=metric_ranking,
        #     metric_weighting=metric_weighting,
        #     metric_weight_label=self.metric_weight,
        #     show_plots=self.show_plots,
        # )
        #                        all_feature_list_to_viz, 'Norm_Frac_Weight',
        #                        'Normalized, Fractionated, and Weighted Feature Importance')

    def preparation(self):
        """
        Creates directory for all results files, decodes included ML modeling
        algorithms that were run
        """
        # Create Directory
        if not os.path.exists(self.full_path + '/model_evaluation'):
            os.mkdir(self.full_path + '/model_evaluation')
        if not os.path.exists(self.full_path + '/model_evaluation/feature_importance/'):
            os.mkdir(self.full_path + '/model_evaluation/feature_importance/')

    def primary_stats_regression(self, master_list=None):
        """
        Combine regression metrics and model feature importance scores across all CV datasets.
        Now reads JSON from metrics_by_cv.
        """
        result_table = []
        metric_dict: Dict[str, Dict[str, List[float]]] = {}

        metrics_dir = Path(self.full_path) / "model_evaluation" / "metrics_by_cv"

        for algorithm in self.algorithms:
            fi_all = []
            mes, maes, mses, mdaes, evss, corrs = [[] for _ in range(6)]

            for cv_count in range(0, self.cv_partitions):
                if master_list is None:
                    mpath = metrics_dir / f"{self.abbrev[algorithm]}_CV_{cv_count}.json"
                    if not mpath.exists():
                        continue
                    with mpath.open("r") as f:
                        payload = json.load(f)
                    metric_payload = payload.get("metrics", payload)
                    fi = payload.get("feature_importance", [])
                else:
                    results = master_list[cv_count][algorithm]
                    metric_payload = results[0]
                    fi = results[1]

                me = metric_payload.get("max_error")
                mae = metric_payload.get("mean_absolute_error")
                mse = metric_payload.get("mean_squared_error")
                mdae = metric_payload.get("median_absolute_error")
                evs = metric_payload.get("explained_variance")
                corr = metric_payload.get("pearson_correlation")

                mes.append(me)
                maes.append(mae)
                mses.append(mse)
                mdaes.append(mdae)
                evss.append(evs)
                corrs.append(corr)

                if master_list is None:
                    temp_list = []
                    headers = pd.read_csv(
                        self.full_path + '/CVDatasets/' + self.data_name
                        + '_CV_' + str(cv_count) + '_Test.csv').columns.values.tolist()
                    if self.instance_label is not None and self.instance_label in headers:
                        headers.remove(self.instance_label)
                    headers.remove(self.outcome_label)

                    if self.original_headers is None:
                        self.original_headers = headers.copy()

                    for each in self.original_headers:
                        if each in headers:
                            f_index = headers.index(each)
                            temp_list.append(fi[f_index] if f_index < len(fi) else 0.0)
                        else:
                            temp_list.append(0.0)
                    fi_all.append(temp_list)

            logging.info("Running stats for " + algorithm)

            mean_me = np.mean(mes, axis=0) if mes else float("nan")
            mean_mae = np.mean(maes, axis=0) if maes else float("nan")
            mean_mse = np.mean(mses, axis=0) if mses else float("nan")
            mean_mdae = np.mean(mdaes, axis=0) if mdaes else float("nan")
            mean_evs = np.mean(evss, axis=0) if evss else float("nan")
            mean_corr = np.mean(corrs, axis=0) if corrs else float("nan")

            results = {
                'Max Error': mes,
                'Mean Absolute Error': maes,
                'Mean Squared Error': mses,
                'Median Absolute Error': mdaes,
                'Explained Variance': evss,
                'Pearson Correlation': corrs,
            }
            dr = pd.DataFrame(results)
            filepath = self.full_path + '/model_evaluation/' + self.abbrev[algorithm] + "_performance.csv"
            dr.to_csv(filepath, header=True, index=False)
            metric_dict[algorithm] = results

            if master_list is None:
                self.save_fi(fi_all, self.abbrev[algorithm], self.original_headers)

            result_dict = {
                'algorithm': algorithm,
                'max_error': mean_me,
                'mean_absolute_error': mean_mae,
                'mean_squared_error': mean_mse,
                'median_absolute_error': mean_mdae,
                'explained_variance': mean_evs,
                'pearson_correlation': mean_corr,
            }
            result_table.append(result_dict)

        result_table = pd.DataFrame.from_dict(result_table)
        if not result_table.empty:
            result_table.set_index('algorithm', inplace=True)
        return result_table, metric_dict


    def _get_multiclass_avg_metric(self, metrics_payload: Dict[str, Any], base: str):
        """
        Helper: pick the right averaged metric for multiclass.

        Preference order:
          requested (micro/macro) -> macro -> micro -> base
        """
        if self.outcome_type != "Multiclass":
            return metrics_payload.get(base)

        suffix = "_" + self.multiclass_average
        candidates = [base + suffix, base + "_macro", base + "_micro", base]
        for key in candidates:
            if key in metrics_payload:
                return metrics_payload.get(key)
        return None

    def primary_stats_multiclass(self, master_list=None, rep_data=None):
        """
        Multiclass classification stats using JSON metrics/curves.

        Metrics:
          - Lets you choose micro vs macro averaging for F1 / Recall / Precision
            via self.multiclass_average ("micro" or "macro").
        Curves:
          - Uses the same averaging key when available (e.g. "micro" or "macro"
            in the saved curve JSONs), otherwise falls back gracefully.
        """
        result_table = []
        metric_dict: Dict[str, Dict[str, List[float]]] = {}

        metrics_dir = Path(self.full_path) / "model_evaluation" / "metrics_by_cv"
        curves_dir = Path(self.full_path) / "model_evaluation" / "curves_by_cv"

        avg_key = self.multiclass_average if self.outcome_type == "Multiclass" else "micro"

        for algorithm in self.algorithms:
            alg_result_table = []

            s_bac, s_ac, s_f1, s_re, s_pr, s_bs = [[] for _ in range(5)]
            fi_all = []

            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)
            mean_recall = np.linspace(0, 1, 100)
            precs = []
            praucs = []
            aveprecs = []

            for cv_count in range(0, self.cv_partitions):
                if master_list is None:
                    mpath = metrics_dir / f"{self.abbrev[algorithm]}_CV_{cv_count}.json"
                    if not mpath.exists():
                        continue
                    with mpath.open("r") as f:
                        payload = json.load(f)
                    metrics_payload = payload.get("metrics", payload)
                    fi = payload.get("feature_importance", [])

                    roc_path = curves_dir / f"{self.abbrev[algorithm]}_CV_{cv_count}_roc.json"
                    prc_path = curves_dir / f"{self.abbrev[algorithm]}_CV_{cv_count}_prc.json"
                    if roc_path.exists():
                        with roc_path.open("r") as f:
                            roc_all = json.load(f)
                    else:
                        roc_all = {}
                    if prc_path.exists():
                        with prc_path.open("r") as f:
                            prc_all = json.load(f)
                    else:
                        prc_all = {}

                    # curves may hold multiple averages (micro/macro) or none
                    roc_m = roc_all.get(avg_key) or roc_all.get("micro") or roc_all.get("macro") or (roc_all or {})
                    prc_m = prc_all.get(avg_key) or prc_all.get("micro") or prc_all.get("macro") or (prc_all or {})

                    fpr = np.asarray(roc_m.get("fpr", []), dtype=float)
                    tpr = np.asarray(roc_m.get("tpr", []), dtype=float)
                    roc_auc = float(roc_m.get("auc", np.nan))

                    prec = np.asarray(prc_m.get("precision", []), dtype=float)
                    recall = np.asarray(prc_m.get("recall", []), dtype=float)
                    prec_rec_auc = float(prc_m.get("pr_auc", np.nan))
                    ave_prec = float(prc_m.get("aps", np.nan))
                else:
                    raise NotImplementedError("master_list not implemented for multiclass yet")

                # metrics
                s_bac.append(metrics_payload.get("balanced_accuracy"))
                s_ac.append(metrics_payload.get("accuracy"))
                s_f1.append(self._get_multiclass_avg_metric(metrics_payload, "f1"))
                s_re.append(self._get_multiclass_avg_metric(metrics_payload, "recall"))
                s_pr.append(self._get_multiclass_avg_metric(metrics_payload, "precision"))
                s_bs.append(metrics_payload.get("brier_score"))

                alg_result_table.append([fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec])

                if fpr.size > 0 and tpr.size > 0:
                    tprs.append(np.interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                    aucs.append(roc_auc)

                if recall.size > 0 and prec.size > 0:
                    precs.append(np.interp(mean_recall, recall, prec))
                    praucs.append(prec_rec_auc)
                    aveprecs.append(ave_prec)

                if master_list is None:
                    temp_list = []
                    headers = pd.read_csv(
                        self.full_path + '/CVDatasets/' + self.data_name
                        + '_CV_' + str(cv_count) + '_Test.csv').columns.values.tolist()
                    if self.instance_label is not None and self.instance_label in headers:
                        headers.remove(self.instance_label)
                    headers.remove(self.outcome_label)
                    if self.original_headers is None:
                        self.original_headers = headers.copy()
                    for each in self.original_headers:
                        if each in headers:
                            f_index = headers.index(each)
                            temp_list.append(fi[f_index] if f_index < len(fi) else 0.0)
                        else:
                            temp_list.append(0.0)
                    fi_all.append(temp_list)

            logging.info("Running stats on " + algorithm)

            # mean ROC curve + per-model ROC plot
            if tprs:
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = np.mean(aucs)
                if self.plot_roc:
                    plot_model_roc(
                        full_path=self.full_path,
                        algorithm=algorithm,
                        abbrev=self.abbrev[algorithm],
                        color=self.colors[algorithm],
                        cv_partitions=self.cv_partitions,
                        mean_fpr=mean_fpr,
                        tprs=tprs,
                        aucs=aucs,
                        alg_result_table=alg_result_table,
                        show_plots=self.show_plots,
                    )
            else:
                mean_tpr = np.zeros_like(mean_fpr)
                mean_auc = float("nan")

            # mean PRC curve + per-model PRC plot
            if precs:
                mean_prec = np.mean(precs, axis=0)
                mean_pr_auc = np.mean(praucs)
                if self.plot_prc:
                    plot_model_prc(
                        full_path=self.full_path,
                        algorithm=algorithm,
                        abbrev=self.abbrev[algorithm],
                        color=self.colors[algorithm],
                        cv_partitions=self.cv_partitions,
                        mean_recall=mean_recall,
                        precs=precs,
                        praucs=praucs,
                        alg_result_table=alg_result_table,
                        outcome_label=self.outcome_label,
                        data_name=self.data_name,
                        instance_label=self.instance_label,
                        rep_data=rep_data,
                        replicate=bool(master_list is not None),
                        show_plots=self.show_plots,
                    )
            else:
                mean_prec = np.zeros_like(mean_recall)
                mean_pr_auc = float("nan")

            results = {
                'Balanced Accuracy': s_bac,
                'Accuracy': s_ac,
                'F1 Score': s_f1,
                'Sensitivity (Recall)': s_re,
                'Precision (PPV)': s_pr,
                'Brier Score': s_bs,
                'ROC AUC': aucs,
                'PRC AUC': praucs,
                'PRC APS': aveprecs,
            }
            dr = pd.DataFrame(results)
            filepath = self.full_path + '/model_evaluation/' + self.abbrev[algorithm] + "_performance.csv"
            dr.to_csv(filepath, header=True, index=False)
            metric_dict[algorithm] = results

            if master_list is None:
                self.save_fi(fi_all, self.abbrev[algorithm], self.original_headers)

            mean_ave_prec = np.mean(aveprecs) if aveprecs else float("nan")
            result_dict = {
                'algorithm': algorithm,
                'fpr': mean_fpr,
                'tpr': mean_tpr,
                'auc': mean_auc,
                'prec': mean_prec,
                'recall': mean_recall,
                'pr_auc': mean_pr_auc,
                'ave_prec': mean_ave_prec,
            }
            result_table.append(result_dict)

        result_table = pd.DataFrame.from_dict(result_table)
        if not result_table.empty:
            result_table.set_index('algorithm', inplace=True)
        return result_table, metric_dict



    def primary_stats_classification(self, master_list=None, rep_data=None):
        """
        Combine binary classification metrics and FI + ROC/PRC data across CVs.

        Reads:
          metrics_by_cv/<ALG>_CV_<k>.json
          curves_by_cv/<ALG>_CV_<k>_roc.json
          curves_by_cv/<ALG>_CV_<k>_prc.json
        """
        result_table = []
        metric_dict: Dict[str, Dict[str, List[float]]] = {}

        metrics_dir = Path(self.full_path) / "model_evaluation" / "metrics_by_cv"
        curves_dir = Path(self.full_path) / "model_evaluation" / "curves_by_cv"

        for algorithm in self.algorithms:
            alg_result_table = []

            # lists of per-CV metrics (we keep the legacy names used in CSVs)
            s_bac, s_ac, s_f1, s_re, s_sp, s_pr = [[] for _ in range(6)]
            s_tp, s_tn, s_fp, s_fn, s_npv, s_lrp, s_lrm = [[] for _ in range(7)]

            fi_all = []

            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)
            mean_recall = np.linspace(0, 1, 100)
            precs = []
            praucs = []
            aveprecs = []

            for cv_count in range(0, self.cv_partitions):
                if master_list is None:
                    mpath = metrics_dir / f"{self.abbrev[algorithm]}_CV_{cv_count}.json"
                    if not mpath.exists():
                        continue
                    with mpath.open("r") as f:
                        payload = json.load(f)

                    metrics_payload = payload.get("metrics", payload)
                    fi = payload.get("feature_importance", [])

                    # curves
                    roc_path = curves_dir / f"{self.abbrev[algorithm]}_CV_{cv_count}_roc.json"
                    prc_path = curves_dir / f"{self.abbrev[algorithm]}_CV_{cv_count}_prc.json"
                    if roc_path.exists():
                        with roc_path.open("r") as f:
                            roc_data = json.load(f)
                    else:
                        roc_data = {}

                    if prc_path.exists():
                        with prc_path.open("r") as f:
                            prc_data = json.load(f)
                    else:
                        prc_data = {}

                    # For binary we store everything under "micro"
                    roc_m = roc_data.get("micro", roc_data or {})
                    prc_m = prc_data.get("micro", prc_data or {})

                    fpr = np.asarray(roc_m.get("fpr", []), dtype=float)
                    tpr = np.asarray(roc_m.get("tpr", []), dtype=float)
                    roc_auc = float(roc_m.get("auc", np.nan))

                    prec = np.asarray(prc_m.get("precision", []), dtype=float)
                    recall = np.asarray(prc_m.get("recall", []), dtype=float)
                    prec_rec_auc = float(prc_m.get("pr_auc", np.nan))
                    ave_prec = float(prc_m.get("aps", np.nan))
                else:
                    # legacy master_list path (if you still use it programmatically)
                    raise Exception("master_list parameter not supported with JSON metrics files")

                # map from JSON metric names to legacy Stats series
                s_bac.append(metrics_payload.get("balanced_accuracy"))
                s_ac.append(metrics_payload.get("accuracy"))
                s_f1.append(metrics_payload.get("f1"))
                s_re.append(metrics_payload.get("recall"))
                s_sp.append(metrics_payload.get("specificity"))
                s_pr.append(metrics_payload.get("precision"))
                s_tp.append(metrics_payload.get("tp"))
                s_tn.append(metrics_payload.get("tn"))
                s_fp.append(metrics_payload.get("fp"))
                s_fn.append(metrics_payload.get("fn"))
                s_npv.append(metrics_payload.get("npv"))
                s_lrp.append(metrics_payload.get("lr_plus"))
                s_lrm.append(metrics_payload.get("lr_minus"))

                alg_result_table.append([fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec])

                if fpr.size > 0 and tpr.size > 0:
                    tprs.append(np.interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                    aucs.append(roc_auc)

                if recall.size > 0 and prec.size > 0:
                    precs.append(np.interp(mean_recall, recall, prec))
                    praucs.append(prec_rec_auc)
                    aveprecs.append(ave_prec)

                if master_list is None:
                    # FI alignment as before
                    temp_list = []
                    headers = pd.read_csv(
                        self.full_path + '/CVDatasets/' + self.data_name
                        + '_CV_' + str(cv_count) + '_Test.csv').columns.values.tolist()
                    if self.instance_label is not None and self.instance_label in headers:
                        headers.remove(self.instance_label)
                    headers.remove(self.outcome_label)
                    if self.original_headers is None:
                        self.original_headers = headers.copy()
                    for each in self.original_headers:
                        if each in headers:
                            f_index = headers.index(each)
                            temp_list.append(fi[f_index] if f_index < len(fi) else 0.0)
                        else:
                            temp_list.append(0.0)
                    fi_all.append(temp_list)

            logging.info("Running stats on " + algorithm)

            # mean ROC curve + plot via helper
            if tprs:
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = np.mean(aucs)
                if self.plot_roc:
                    plot_model_roc(
                        full_path=self.full_path,
                        algorithm=algorithm,
                        abbrev=self.abbrev[algorithm],
                        color=self.colors[algorithm],
                        cv_partitions=self.cv_partitions,
                        mean_fpr=mean_fpr,
                        tprs=tprs,
                        aucs=aucs,
                        alg_result_table=alg_result_table,
                        show_plots=self.show_plots,
                    )
            else:
                mean_tpr = np.zeros_like(mean_fpr)
                mean_auc = float("nan")

            # mean PRC curve + plot via helper
            if precs:
                mean_prec = np.mean(precs, axis=0)
                mean_pr_auc = np.mean(praucs)
                if self.plot_prc:
                    plot_model_prc(
                        full_path=self.full_path,
                        algorithm=algorithm,
                        abbrev=self.abbrev[algorithm],
                        color=self.colors[algorithm],
                        cv_partitions=self.cv_partitions,
                        mean_recall=mean_recall,
                        precs=precs,
                        praucs=praucs,
                        alg_result_table=alg_result_table,
                        outcome_label=self.outcome_label,
                        data_name=self.data_name,
                        instance_label=self.instance_label,
                        rep_data=rep_data,
                        replicate=bool(master_list is not None),
                        show_plots=self.show_plots,
                    )
            else:
                mean_prec = np.zeros_like(mean_recall)
                mean_pr_auc = float("nan")

            results = {
                'Balanced Accuracy': s_bac,
                'Accuracy': s_ac,
                'F1 Score': s_f1,
                'Sensitivity (Recall)': s_re,
                'Specificity': s_sp,
                'Precision (PPV)': s_pr,
                'TP': s_tp,
                'TN': s_tn,
                'FP': s_fp,
                'FN': s_fn,
                'NPV': s_npv,
                'LR+': s_lrp,
                'LR-': s_lrm,
                'ROC AUC': aucs,
                'PRC AUC': praucs,
                'PRC APS': aveprecs,
            }
            dr = pd.DataFrame(results)
            filepath = self.full_path + '/model_evaluation/' + self.abbrev[algorithm] + "_performance.csv"
            dr.to_csv(filepath, header=True, index=False)
            metric_dict[algorithm] = results

            if master_list is None:
                if self.feature_headers is None:
                    self.feature_headers = self.original_headers
                self.save_fi(fi_all, self.abbrev[algorithm], self.feature_headers)

            mean_ave_prec = np.mean(aveprecs) if aveprecs else float("nan")
            result_dict = {
                'algorithm': algorithm,
                'fpr': mean_fpr,
                'tpr': mean_tpr,
                'auc': mean_auc,
                'prec': mean_prec,
                'recall': mean_recall,
                'pr_auc': mean_pr_auc,
                'ave_prec': mean_ave_prec,
            }
            result_table.append(result_dict)

        result_table = pd.DataFrame.from_dict(result_table)
        if not result_table.empty:
            result_table.set_index('algorithm', inplace=True)
        return result_table, metric_dict


    def save_fi(self, fi_all, algorithm, global_feature_list):
        """
        Creates directory to store model feature importance results and,
        for each algorithm, exports a file of feature importance scores from each CV.
        """
        dr = pd.DataFrame(fi_all)
        if not os.path.exists(self.full_path + '/model_evaluation/feature_importance/'):
            os.mkdir(self.full_path + '/model_evaluation/feature_importance/')
        filepath = self.full_path + '/model_evaluation/feature_importance/' + algorithm + "_FI.csv"
        dr.to_csv(filepath, header=global_feature_list, index=False)

    def save_metric_stats(self, metrics, metric_dict):
        """
        Exports csv file with mean, median and std dev metric values
        (over all CVs) for each ML modeling algorithm
        """
        # TODO: Clean this function up, save everything together
        with open(self.full_path + '/model_evaluation/Summary_performance_median.csv', mode='w', newline="") as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            e = ['']
            e.extend(metrics)
            writer.writerow(e)  # Write headers (balanced accuracy, etc.)
            for algorithm in metric_dict:
                astats = []
                for li in list(metric_dict[algorithm].values()):
                    li = [float(i) for i in li]
                    mediani = median(li)
                    astats.append(str(mediani))
                to_add = [algorithm]
                to_add.extend(astats)
                writer.writerow(to_add)
        file.close()
        with open(self.full_path + '/model_evaluation/Summary_performance_mean.csv', mode='w', newline="") as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            e = ['']
            e.extend(metrics)
            writer.writerow(e)  # Write headers (balanced accuracy, etc.)
            for algorithm in metric_dict:
                astats = []
                for li in list(metric_dict[algorithm].values()):
                    li = [float(i) for i in li]
                    meani = mean(li)
                    astats.append(str(meani))
                to_add = [algorithm]
                to_add.extend(astats)
                writer.writerow(to_add)
        file.close()
        with open(self.full_path + '/model_evaluation/Summary_performance_std.csv', mode='w', newline="") as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            e = ['']
            e.extend(metrics)
            writer.writerow(e)  # Write headers (balanced accuracy, etc.)
            for algorithm in metric_dict:
                astats = []
                for li in list(metric_dict[algorithm].values()):
                    li = [float(i) for i in li]
                    std = stdev(li)
                    astats.append(str(std))
                to_add = [algorithm]
                to_add.extend(astats)
                writer.writerow(to_add)
        file.close()

    def kruskal_wallis(self, metrics, metric_dict):
        """
        Apply non-parametric Kruskal Wallis one-way ANOVA on ranks.
        Determines if there is a statistically significant difference in algorithm performance across CV runs.
        Completed for each standard metric separately.
        """
        # Create directory to store significance testing results (used for both Kruskal Wallis and MannWhitney U-test)
        if not os.path.exists(self.full_path + '/model_evaluation/statistical_comparisons'):
            os.mkdir(self.full_path + '/model_evaluation/statistical_comparisons')
        # Create dataframe to store analysis results for each metric
        label = ['Statistic', 'P-Value', 'Sig(*)']
        kruskal_summary = pd.DataFrame(index=metrics, columns=label)
        # Apply Kruskal Wallis test for each metric
        for metric in metrics:
            temp_array = []
            for algorithm in self.algorithms:
                temp_array.append(metric_dict[algorithm][metric])
            try:
                result = kruskal(*temp_array)
            except Exception:
                result = [temp_array[0], 1]
            kruskal_summary.at[metric, 'Statistic'] = str(round(result[0], 6))
            kruskal_summary.at[metric, 'P-Value'] = str(round(result[1], 6))
            if result[1] < self.sig_cutoff:
                kruskal_summary.at[metric, 'Sig(*)'] = str('*')
            else:
                kruskal_summary.at[metric, 'Sig(*)'] = str('')
        # Export analysis summary to .csv file
        kruskal_summary.to_csv(self.full_path + '/model_evaluation/statistical_comparisons/KruskalWallis.csv')
        return kruskal_summary

    def wilcoxon_rank(self, metrics, metric_dict, kruskal_summary):
        """
        Apply non-parametric Wilcoxon signed-rank test (pairwise comparisons).
        If a significant Kruskal Wallis algorithm difference was found for a
        given metric, Wilcoxon tests individual algorithm pairs
        to determine if there is a statistically significant difference in
        algorithm performance across CV runs. Test statistic will be zero if
        all scores from one set are
        larger than the other.
        """
        for metric in metrics:
            if kruskal_summary['Sig(*)'][metric] == '*':
                wilcoxon_stats = []
                done = []
                for algorithm1 in self.algorithms:
                    for algorithm2 in self.algorithms:
                        if (not [algorithm1, algorithm2] in done) and \
                                (not [algorithm2, algorithm1] in done) and (algorithm1 != algorithm2):
                            set1 = metric_dict[algorithm1][metric]
                            set2 = metric_dict[algorithm2][metric]
                            # handle error when metric values are equal for both algorithms
                            if set1 == set2:  # Check if all nums are equal in sets
                                report = ['NA', 1]
                            else:  # Apply Wilcoxon Rank Sum test
                                try:
                                    report = wilcoxon(set1, set2)
                                except Exception:
                                    report = ['NA_error', 1]
                            # Summarize test information in list
                            tempstats = [algorithm1, algorithm2, report[0], report[1], '']
                            if report[1] < self.sig_cutoff:
                                tempstats[4] = '*'
                            wilcoxon_stats.append(tempstats)
                            done.append([algorithm1, algorithm2])
                # Export test results
                wilcoxon_stats_df = pd.DataFrame(wilcoxon_stats)
                wilcoxon_stats_df.columns = ['Algorithm 1', 'Algorithm 2', 'Statistic', 'P-Value', 'Sig(*)']
                wilcoxon_stats_df.to_csv(self.full_path
                                         + '/model_evaluation/statistical_comparisons/'
                                           'WilcoxonRank_' + metric + '.csv', index=False)

    def mann_whitney_u(self, metrics, metric_dict, kruskal_summary):
        """
        Apply non-parametric Mann Whitney U-test (pairwise comparisons).
        If a significant Kruskal Wallis algorithm difference was found for
        a given metric, Mann Whitney tests individual algorithm pairs
        to determine if there is a statistically significant difference
        in algorithm performance across CV runs. Test statistic will be
        zero if all scores from one set are larger than the other.
        """
        for metric in metrics:
            if kruskal_summary['Sig(*)'][metric] == '*':
                mann_stats = []
                done = []
                for algorithm1 in self.algorithms:
                    for algorithm2 in self.algorithms:
                        if (not [algorithm1, algorithm2] in done) and \
                                (not [algorithm2, algorithm1] in done) and (algorithm1 != algorithm2):
                            set1 = metric_dict[algorithm1][metric]
                            set2 = metric_dict[algorithm2][metric]
                            if set1 == set2:  # Check if all nums are equal in sets
                                report = ['NA', 1]
                            else:  # Apply Mann Whitney U test
                                try:
                                    report = mannwhitneyu(set1, set2)
                                except Exception:
                                    report = ['NA_error', 1]
                            # Summarize test information in list
                            tempstats = [algorithm1, algorithm2, report[0], report[1], '']
                            if report[1] < self.sig_cutoff:
                                tempstats[4] = '*'
                            mann_stats.append(tempstats)
                            done.append([algorithm1, algorithm2])
                # Export test results
                mann_stats_df = pd.DataFrame(mann_stats)
                mann_stats_df.columns = ['Algorithm 1', 'Algorithm 2', 'Statistic', 'P-Value', 'Sig(*)']
                mann_stats_df.to_csv(self.full_path +
                                     '/model_evaluation/'
                                     'statistical_comparisons/MannWhitneyU_' + metric + '.csv', index=False)
                
    def ensemble_stats_summary(self):
        """
        Summarize ensembles created in Phase 7 (if any exist) and
        generate ensemble-only ROC / PRC summary plots + metrics tables.

        Uses IO-only helpers:
          - _collect_ensemble_metrics_core
          - _plot_ensemble_roc_summary
          - _plot_ensemble_prc_summary
        """
        ens_root = Path(self.full_path) / "ensemble_evaluation"
        metrics_dir = ens_root / "metrics_by_cv"
        curves_dir = ens_root / "curves_by_cv"

        if not metrics_dir.exists():
            logging.info("No ensemble_evaluation/metrics_by_cv found. Skipping ensemble summary.")
            return

        logging.info("Collecting ensemble statistics from %s", str(ens_root))

        metrics_by_ens, metric_names = self._collect_ensemble_metrics_core(metrics_dir)
        if not metrics_by_ens:
            logging.info("No ensemble metrics found. Skipping ensemble summary.")
            return

        # write ensemble-only summary tables
        self._write_ensemble_metric_summaries(ens_root, metrics_by_ens, metric_names)

        # curves summary & plots
        roc_summary, prc_summary = self._collect_ensemble_curves_core(curves_dir, metrics_by_ens.keys())
        if self.plot_roc and roc_summary:
            plot_ensemble_roc_summary(
                ens_root=ens_root,
                roc_summary=roc_summary,
                show_plots=self.show_plots,
            )
        if self.plot_prc and prc_summary:
            plot_ensemble_prc_summary(
                ens_root=ens_root,
                prc_summary=prc_summary,
                full_path=self.full_path,
                data_name=self.data_name,
                outcome_label=self.outcome_label,
                instance_label=self.instance_label,
                show_plots=self.show_plots,
            )

    # ------------------- ensemble core helpers -------------------------
    def _collect_ensemble_metrics_core(self, metrics_dir: Path) -> Tuple[Dict[str, Dict[str, List[float]]], List[str]]:
        """
        Collect per-CV JSON metrics for each ensemble id.
        Returns:
          metrics_by_ens: {ens_id: {metric_name: [values across CVs]}}
          metric_names: list of metric names (from first ensemble)
        """
        metrics_by_ens: Dict[str, Dict[str, List[float]]] = {}
        for fn in metrics_dir.glob("*.json"):
            # pattern: <ens_id>_CV_<cv>.json
            m = re.match(r"(.+?)_CV_(\d+)\.json$", fn.name)
            if not m:
                continue
            ens_id = m.group(1)
            with open(fn, "r") as f:
                data = json.load(f)
            md = metrics_by_ens.setdefault(ens_id, {})
            for k, v in data.items():
                try:
                    val = float(v)
                except Exception:
                    continue
                md.setdefault(k, []).append(val)

        if not metrics_by_ens:
            return {}, []

        # Metric names from first ensemble
        first_ens = next(iter(metrics_by_ens.keys()))
        metric_names = sorted(metrics_by_ens[first_ens].keys())
        return metrics_by_ens, metric_names

    def _write_ensemble_metric_summaries(
        self,
        ens_root: Path,
        metrics_by_ens: Dict[str, Dict[str, List[float]]],
        metric_names: List[str],
    ):
        """
        IO helper: write mean/median/std summary CSVs for ensemble metrics.
        """
        if not ens_root.exists():
            ens_root.mkdir(parents=True, exist_ok=True)
        out_mean = ens_root / "Ensembles_performance_mean.csv"
        out_median = ens_root / "Ensembles_performance_median.csv"
        out_std = ens_root / "Ensembles_performance_std.csv"

        # mean
        with out_mean.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Ensemble"] + metric_names)
            for ens_id, md in metrics_by_ens.items():
                row = [ens_id]
                for m in metric_names:
                    vals = [float(x) for x in md.get(m, [])]
                    row.append(str(mean(vals)) if vals else "nan")
                w.writerow(row)

        # median
        with out_median.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Ensemble"] + metric_names)
            for ens_id, md in metrics_by_ens.items():
                row = [ens_id]
                for m in metric_names:
                    vals = [float(x) for x in md.get(m, [])]
                    row.append(str(median(vals)) if vals else "nan")
                w.writerow(row)

        # std
        with out_std.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Ensemble"] + metric_names)
            for ens_id, md in metrics_by_ens.items():
                row = [ens_id]
                for m in metric_names:
                    vals = [float(x) for x in md.get(m, [])]
                    if len(vals) > 1:
                        row.append(str(stdev(vals)))
                    else:
                        row.append("nan")
                w.writerow(row)

    def _collect_ensemble_curves_core(
        self,
        curves_dir: Path,
        ensemble_ids,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Core: aggregate ROC/PRC curves over CV for each ensemble id.

        Returns:
          roc_summary: {ens_id: {"fpr": common_fpr, "tpr": mean_tpr, "auc": mean_auc}}
          prc_summary: {ens_id: {"recall": common_rec, "precision": mean_prec,
                                 "pr_auc": mean_pr_auc, "aps": mean_aps}}
        """
        roc_summary: Dict[str, Dict[str, Any]] = {}
        prc_summary: Dict[str, Dict[str, Any]] = {}

        common_fpr = np.linspace(0.0, 1.0, 200)
        common_rec = np.linspace(0.0, 1.0, 200)

        for ens_id in ensemble_ids:
            # ROC
            tprs = []
            aucs = []
            for roc_file in curves_dir.glob(f"{ens_id}_CV_*_roc.json"):
                with roc_file.open("r") as f:
                    roc_data = json.load(f)
                fpr = np.array(roc_data.get("fpr", []), dtype=float)
                tpr = np.array(roc_data.get("tpr", []), dtype=float)
                if fpr.size == 0 or tpr.size == 0:
                    continue
                tinterp = np.interp(common_fpr, fpr, tpr)
                tinterp[0] = 0.0
                tinterp[-1] = 1.0
                tprs.append(tinterp)
                aucs.append(auc(fpr, tpr))
            if tprs:
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                roc_summary[ens_id] = {
                    "fpr": common_fpr,
                    "tpr": mean_tpr,
                    "auc": float(np.mean(aucs)),
                }

            # PRC
            precs = []
            pr_aucs = []
            aps_list = []
            for prc_file in curves_dir.glob(f"{ens_id}_CV_*_prc.json"):
                with prc_file.open("r") as f:
                    prc_data = json.load(f)
                prec = np.array(prc_data.get("precision", []), dtype=float)
                rec = np.array(prc_data.get("recall", []), dtype=float)
                if rec.size == 0 or prec.size == 0:
                    continue
                sidx = np.argsort(rec)
                rec_sorted = rec[sidx]
                prec_sorted = prec[sidx]
                pinterp = np.interp(common_rec, rec_sorted, prec_sorted)
                precs.append(pinterp)
                pr_aucs.append(auc(rec_sorted, prec_sorted))
                # approximate APS by simple average (we don't have raw y/proba here)
                aps_list.append(float(np.mean(prec_sorted)))
            if precs:
                mean_prec = np.mean(precs, axis=0)
                prc_summary[ens_id] = {
                    "recall": common_rec,
                    "precision": mean_prec,
                    "pr_auc": float(np.mean(pr_aucs)),
                    "aps": float(np.mean(aps_list)) if aps_list else float("nan"),
                }

        return roc_summary, prc_summary

    def parse_runtime(self):
        """
        Loads runtime summaries from the entire pipeline and parses them into
        a single CSV runtime report.

        This implementation no longer relies on pickle; it just
        aggregates by the token after 'runtime_' in the filename. For model
        runtimes, this will be the algorithm small_name (e.g. 'LR', 'SVM').
        """
        dict_obj: Dict[str, float] = {}
        dict_obj["preprocessing"] = 0.0

        runtime_dir = Path(self.full_path) / "runtime"
        for file_path in glob.glob(str(runtime_dir / "runtime_*.txt")) \
            + glob.glob(str(runtime_dir / "models/runtime_*.txt")):
            file_path = str(Path(file_path).as_posix())
            with open(file_path, "r") as f:
                try:
                    val = float(f.readline())
                except Exception:
                    continue

            # file name: runtime_<ref>[_...].txt
            fname = os.path.basename(file_path)
            parts = fname.split("_")
            if len(parts) < 2:
                continue
            ref = parts[1].split(".")[0]  # e.g. 'exploratory', 'Stats', 'LR'

            if "preprocessing" in ref:
                dict_obj["preprocessing"] = dict_obj.get("preprocessing", 0.0) + val
            else:
                dict_obj[ref] = dict_obj.get(ref, 0.0) + val

        with open(self.full_path + "/runtimes.csv", mode="w", newline="") as file:
            writer = csv.writer(
                file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            writer.writerow(["Pipeline Component", "Phase", "Time (sec)"])

            # Phase 1 & 2 names are kept for backward compatibility
            if "exploratory" in dict_obj:
                writer.writerow(["Exploratory Analysis", 1, dict_obj["exploratory"]])
            writer.writerow(["Scale and Impute", 2, dict_obj.get("preprocessing", 0.0)])

            if "mutual" in dict_obj:
                writer.writerow(["Mutual Information (Feature Importance)", 3, dict_obj["mutual"]])
            if "multisurf" in dict_obj:
                writer.writerow(["MultiSURF (Feature Importance)", 3, dict_obj["multisurf"]])

            if "featureselection" in dict_obj:
                writer.writerow(["Feature Selection", 4, dict_obj["featureselection"]])

            # Any other keys that match algorithm small_names => Phase 6 Modeling
            for alg in self.algorithms:
                if alg in dict_obj:
                    writer.writerow(
                        [f"{alg} (Modeling)", 6, dict_obj[alg]]
                    )

            # Stats phase itself
            if "Stats" in dict_obj:
                writer.writerow(["Stats Summary", 8, dict_obj["Stats"]])

    def save_runtime(self):
        """
        Save phase runtime
        """
        os.makedirs(self.full_path + '/runtime/' , exist_ok=True)
        runtime_file = open(
            self.full_path + "/runtime/runtime_Stats.txt", "w"
        )
        runtime_file.write(str(time.time() - self.job_start_time))
        runtime_file.close()
