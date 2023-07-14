import csv
import glob
import os
import pickle
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from statistics import mean, median, stdev

from scipy import stats
from scipy.stats import kruskal, wilcoxon, mannwhitneyu
from streamline.utils.job import Job
import seaborn as sns

sns.set_theme()


class StatsJob(Job):
    """
    This 'Job' script creates summaries of ML classification evaluation statistics
    (means and standard deviations), ROC and PRC plots (comparing CV performance
    in the same ML algorithm and comparing average performance
    between ML algorithms), model feature importance averages over CV runs,
    boxplot comparing ML algorithms for each metric, Kruskal Wallis
    and Mann Whitney statistical comparisons between ML algorithms, model
    feature importance boxplot for each algorithm, and composite feature
    importance plots summarizing model feature importance across all ML algorithms.
    It is run for a single dataset from the original target
    dataset folder (data_path) in Phase 1 (i.e. stats summary completed for all cv datasets).
    """

    def __init__(self, full_path, outcome_label, outcome_type, instance_label,
                 scoring_metric='balanced_accuracy',
                 cv_partitions=5, top_features=40, sig_cutoff=0.05, metric_weight='balanced_accuracy', scale_data=True,
                 plot_roc=True, plot_prc=True, plot_fi_box=True, plot_metric_boxplots=True, show_plots=False):
        """

        Args:
            full_path:
            outcome_label:
            instance_label:
            scoring_metric:
            cv_partitions:
            top_features:
            sig_cutoff:
            metric_weight:
            scale_data:
            plot_roc:
            plot_prc:
            plot_fi_box:
            plot_metric_boxplots:
            show_plots:
        """
        super().__init__()
        self.full_path = full_path
        self.outcome_label = outcome_label
        self.outcome_type = outcome_type
        self.instance_label = instance_label
        self.data_name = self.full_path.split('/')[-1]
        self.experiment_path = '/'.join(self.full_path.split('/')[:-1])

        self.plot_roc = plot_roc
        self.plot_prc = plot_prc
        self.plot_fi_box = plot_fi_box
        self.cv_partitions = cv_partitions
        self.scale_data = scale_data
        self.plot_metric_boxplots = plot_metric_boxplots
        self.scoring_metric = scoring_metric
        self.top_features = top_features
        self.sig_cutoff = sig_cutoff
        self.metric_weight = metric_weight

        if self.outcome_type == "Continuous" and (self.scoring_metric == 'balanced_accuracy'
                                                  or self.metric_weight == 'balanced_accuracy'):
            self.metric_weight = 'explained_variance'
            self.scoring_metric = 'explained_variance'

        self.show_plots = show_plots
        if self.plot_fi_box:
            self.original_headers = pd.read_csv(self.full_path + "/exploratory/OriginalFeatureNames.csv",
                                                sep=',').columns.values.tolist()  # Get Original Headers
        else:
            try:
                self.original_headers = pd.read_csv(self.full_path
                                                    + "/exploratory/OriginalFeatureNames.csv",
                                                    sep=',').columns.values.tolist()  # Get Original Headers
            except Exception:
                self.original_headers = None
        if 'applymodel' not in full_path:
            self.partial_path = '/'.join(full_path.split('/')[:-1])
        else:
            self.partial_path = '/'.join(full_path.split('/')[:-3])
        pickle_in = open(self.partial_path + '/' + "algInfo.pickle", 'rb')
        alg_info = pickle.load(pickle_in)
        algorithms = list()
        abbrev = dict()
        colors = dict()
        for algorithm in alg_info.keys():
            if alg_info[algorithm][0]:
                algorithms.append(algorithm)
                abbrev[algorithm] = alg_info[algorithm][1]
                colors[algorithm] = alg_info[algorithm][2]
        self.algorithms = algorithms
        self.abbrev = abbrev
        self.colors = colors
        pickle_in.close()

    def run(self):
        self.job_start_time = time.time()  # for tracking phase runtime
        logging.info('Running Statistics Summary for ' + str(self.data_name))

        # Translate metric name from scikit-learn standard
        # (currently balanced accuracy is hardcoded for use in generating FI plots due to no-skill normalization)
        if self.outcome_type == "Categorical":
            metric_term_dict = {'balanced_accuracy': 'Balanced Accuracy', 'accuracy': 'Accuracy', 'f1': 'F1 Score',
                                'recall': 'Sensitivity (Recall)', 'precision': 'Precision (PPV)', 'roc_auc': 'ROC AUC'}
        elif self.outcome_type == "Continuous":
            metric_term_dict = {'max_error': 'Max Error', 'mean_absolute_error': 'Mean Absolute Error',
                                'mean_squared_error': 'Mean Squared Error',
                                'median_absolute_error': 'Median Absolute Error',
                                'explained_variance': 'Explained Variance',
                                'pearson_correlation': 'Pearson Correlation', 'f1': 'F1 Score'}

        self.metric_weight = metric_term_dict[self.metric_weight]

        # Get algorithms run, specify algorithm abbreviations, colors to use for
        # algorithms in plots, and original ordered feature name list
        self.preparation()

        if self.outcome_type == "Categorical":
            # Gather and summarize all evaluation metrics for each algorithm across all CVs.
            # Returns result_table used to plot average ROC and PRC plots and metric_dict
            # organizing all metrics over all algorithms and CVs.
            result_table, metric_dict = self.primary_stats_classification()
            # Plot ROC and PRC curves comparing average ML algorithm performance (averaged over all CVs)
            logging.info('Generating ROC and PRC plots...')
            self.do_plot_roc(result_table)
            self.do_plot_prc(result_table)
        elif self.outcome_type == "Continuous":
            result_table, metric_dict = self.primary_stats_regression()
            self.residuals_regression()

        # Make list of metric names
        logging.info('Saving Metric Summaries...')
        metrics = list(metric_dict[self.algorithms[0]].keys())

        # Save metric means, median and standard deviations
        self.save_metric_stats(metrics, metric_dict)

        # Generate boxplot comparing algorithm performance for each standard metric, if specified by user
        if self.plot_metric_boxplots:
            logging.info('Generating Metric Boxplots...')
            self.metric_boxplots(metrics, metric_dict)

        # Calculate and export Kruskal Wallis, Mann Whitney, and wilcoxon Rank sum stats
        # if more than one ML algorithm has been run (for the comparison) - note stats are based on
        # comparing the multiple CV models for each algorithm.
        if len(self.algorithms) > 1:
            logging.info('Running Non-Parametric Statistical Significance Analysis...')
            kruskal_summary = self.kruskal_wallis(metrics, metric_dict)
            self.wilcoxon_rank(metrics, metric_dict, kruskal_summary)
            self.mann_whitney_u(metrics, metric_dict, kruskal_summary)

        if self.outcome_type == "Categorical":
            ave_or_median = 'Median'
        else:
            ave_or_median = 'Mean'

        # Run FI Related stats and plots
        self.fi_stats(metric_dict, ave_or_median)

        # Export phase runtime
        self.save_runtime()

        # Parse all pipeline runtime files into a single runtime report
        self.parse_runtime()

        # Print phase completion
        logging.info(self.data_name + " phase 5 complete")
        job_file = open(self.experiment_path + '/jobsCompleted/job_stats_' + self.data_name + '.txt', 'w')
        job_file.write('complete')
        job_file.close()

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

    def fi_stats(self, metric_dict, ave_or_median='Median'):
        # Prepare for feature importance visualizations
        logging.info('Preparing for Model Feature Importance Plotting...')

        # old - 'Balanced Accuracy'
        fi_df_list, fi_med_list, fi_med_norm_list, med_metric_list, all_feature_list, \
            non_zero_union_features, \
            non_zero_union_indexes = self.prep_fi(metric_dict, ave_or_median)

        # Select 'top' features for composite visualisation
        features_to_viz = self.select_for_composite_viz(non_zero_union_features, non_zero_union_indexes,
                                                        med_metric_list, fi_med_norm_list)

        # Generate FI boxplots for each modeling algorithm if specified by user
        if self.plot_fi_box:
            logging.info('Generating Feature Importance Boxplot and Histograms...')
            self.do_fi_boxplots(fi_df_list, fi_med_list, ave_or_median)
            self.do_fi_histogram(fi_med_list, ave_or_median)

        # Visualize composite FI - Currently set up to only use Balanced Accuracy for composite FI plot visualization
        logging.info('Generating Composite Feature Importance Plots...')
        # Take top feature names to visualize and get associated feature importance values for each algorithm,
        # and original data ordered feature names list
        # If we want composite FI plots to be displayed in descending total bar height order.

        top_fi_med_norm_list, all_feature_list_to_viz = self.get_fi_to_viz_sorted(features_to_viz, all_feature_list,
                                                                                  fi_med_norm_list)

        # Generate Normalized composite FI plot
        self.composite_fi_plot(top_fi_med_norm_list, all_feature_list_to_viz, 'Norm',
                               "Normalized " + ave_or_median + " Feature Importance")

        # # Fractionate FI scores for normalized and fractionated composite FI plot
        # frac_lists = self.frac_fi(top_fi_med_norm_list)

        # # Generate Normalized and Fractionated composite FI plot
        # composite_fi_plot(frac_lists, algorithms, list(colors.values()),
        #                   all_feature_list_to_viz, 'Norm_Frac',
        #                   'Normalized and Fractionated Feature Importance')

        # Weight FI scores for normalized and (model performance) weighted composite FI plot
        weighted_lists, weights = self.weight_fi(med_metric_list, top_fi_med_norm_list)

        # Generate Normalized and Weighted Compound FI plot
        self.composite_fi_plot(weighted_lists, all_feature_list_to_viz,
                               'Norm_Weight', 'Normalized and Weighted ' + ave_or_median + ' Feature Importance')

        # Weight the Fractionated FI scores for normalized,fractionated, and weighted compound FI plot
        # weighted_frac_lists = self.weight_frac_fi(frac_lists,weights)

        # Generate Normalized, Fractionated, and Weighted Compound FI plot
        # self.composite_fi_plot(weighted_frac_lists, algorithms, list(colors.values()),
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
        """
        result_table = []
        metric_dict = {}
        for algorithm in self.algorithms:  # completed for each individual ML modeling algorithm
            # Define feature importance lists
            # used to save model feature importance individually for each cv within single summary file
            # (all original features in dataset prior to feature selection included)
            fi_all = []
            # Define evaluation stats variable lists
            mes = []  # store max error
            maes = []  # store mean absolute error
            mses = []  # store mean squared error
            mdaes = []  # store median absolute error
            evss = []  # store explained variance (r2)
            corrs = []  # store Pearson correlation
            # Gather statistics over all CV partitions
            for cv_count in range(0, self.cv_partitions):
                # Unpickle saved metrics from previous phase
                if master_list is None:
                    result_file = self.full_path + '/model_evaluation/pickled_metrics/' \
                                  + self.abbrev[algorithm] + "_CV_" + str(cv_count) + "_metrics.pickle"
                    file = open(result_file, 'rb')
                    results = pickle.load(file)
                    file.close()
                else:
                    results = master_list[cv_count][algorithm]

                # Separate pickled results
                me = results[0][0]
                mae = results[0][1]
                mse = results[0][2]
                mdae = results[0][3]
                evs = results[0][4]
                corr = results[0][5]
                fi = results[1]

                mes.append(me)
                maes.append(mae)
                mses.append(mse)
                mdaes.append(mdae)
                evss.append(evs)
                corrs.append(corr)
                # Format feature importance scores as list
                # (takes into account that all features are not in each CV partition)
                if master_list is None:
                    temp_list = []
                    j = 0
                    headers = pd.read_csv(self.full_path + '/CVDatasets/' + self.data_name
                                          + '_CV_' + str(cv_count) + '_Test.csv').columns.values.tolist()
                    if self.instance_label is not None:
                        headers.remove(self.instance_label)
                    headers.remove(self.outcome_label)
                    for each in self.original_headers:
                        if each in headers:  # Check if current feature from original dataset was in the partition
                            # Deal with features not being in original order (find index of current feature list.index()
                            f_index = headers.index(each)
                            temp_list.append(fi[f_index])
                        else:
                            temp_list.append(0)
                        j += 1
                    fi_all.append(temp_list)

            logging.info("Running stats for " + algorithm)

            mean_me = np.mean(mes, axis=0)
            mean_mae = np.mean(maes, axis=0)
            mean_mse = np.mean(mses, axis=0)
            mean_mdae = np.mean(mdaes, axis=0)
            mean_evs = np.mean(evss, axis=0)
            mean_corr = np.mean(corrs, axis=0)

            # Export and save all CV metric stats for each individual algorithm
            results = {'Max Error': mes, 'Mean Absolute Error': maes, 'Mean Squared Error': mses,
                       'Median Absolute Error': mdaes, 'Explained Variance': evss, 'Pearson Correlation': corrs}
            dr = pd.DataFrame(results)
            filepath = self.full_path + '/model_evaluation/' + self.abbrev[algorithm] + "_performance.csv"
            dr.to_csv(filepath, header=True, index=False)
            metric_dict[algorithm] = results

            # Save Average FI Stats
            if master_list is None:
                self.save_fi(fi_all, self.abbrev[algorithm], self.original_headers)

            result_dict = {'algorithm': algorithm, 'max_error': mean_me, 'mean_absolute_error': mean_mae,
                           'mean_squared_error': mean_mse, 'median_absolute_error': mean_mdae,
                           'explained_variance': mean_evs, 'pearson_correlation': mean_corr}
            result_table.append(result_dict)
        # Result table comparing average ML algorithm performance.
        result_table = pd.DataFrame.from_dict(result_table)
        result_table.set_index('algorithm', inplace=True)
        return result_table, metric_dict

    def primary_stats_classification(self, master_list=None, rep_data=None):
        """
        Combine classification metrics and model feature importance scores
        as well as ROC and PRC plot data across all CV datasets.
        Generate ROC and PRC plots comparing separate CV models for each individual modeling algorithm.
        """
        result_table = []
        metric_dict = {}

        # completed for each individual ML modeling algorithm
        for algorithm in self.algorithms:

            # stores values used in ROC and PRC plots
            alg_result_table = []

            # Define evaluation stats variable lists
            s_bac, s_ac, s_f1, s_re, s_sp, s_pr, s_tp, s_tn, s_fp, s_fn, s_npv, s_lrp, s_lrm = [[] for _ in range(13)]

            # Define feature importance lists
            # used to save model feature importance individually for
            # each cv within single summary file (all original features
            # in dataset prior to feature selection included)
            fi_all = []

            # Define ROC plot variable lists
            tprs = []  # stores interpolated true positive rates for average CV line in ROC
            aucs = []  # stores individual CV areas under ROC curve to calculate average

            mean_fpr = np.linspace(0, 1, 100)  # used to plot average of CV line in ROC plot
            mean_recall = np.linspace(0, 1, 100)  # used to plot average of CV line in PRC plot

            # Define PRC plot variable lists
            precs = []  # stores interpolated precision values for average CV line in PRC
            praucs = []  # stores individual CV areas under PRC curve to calculate average
            aveprecs = []  # stores individual CV average precisions for PRC to calculate CV average

            # Gather statistics over all CV partitions
            for cv_count in range(0, self.cv_partitions):

                if master_list is None:

                    # Unpickle saved metrics from previous phase
                    result_file = self.full_path + '/model_evaluation/pickled_metrics/' \
                                  + self.abbrev[algorithm] + "_CV_" + str(cv_count) + "_metrics.pickle"
                    file = open(result_file, 'rb')
                    results = pickle.load(file)
                    # [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi, probas_]
                    file.close()

                    # Separate pickled results
                    metric_list, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi, probas_ = results
                else:
                    results = master_list[cv_count][algorithm]
                    # grabs evalDict for a specific algorithm entry (with data values)
                    metric_list = results[0]
                    fpr = results[1]
                    tpr = results[2]
                    roc_auc = results[3]
                    prec = results[4]
                    recall = results[5]
                    prec_rec_auc = results[6]
                    ave_prec = results[7]
                    fi = results[8]
                    probas_ = results[9]

                # Separate metrics from metricList
                s_bac.append(metric_list[0])
                s_ac.append(metric_list[1])
                s_f1.append(metric_list[2])
                s_re.append(metric_list[3])
                s_sp.append(metric_list[4])
                s_pr.append(metric_list[5])
                s_tp.append(metric_list[6])
                s_tn.append(metric_list[7])
                s_fp.append(metric_list[8])
                s_fn.append(metric_list[9])
                s_npv.append(metric_list[10])
                s_lrp.append(metric_list[11])
                s_lrm.append(metric_list[12])

                # update list that stores values used in ROC and PRC plots
                alg_result_table.append([fpr, tpr, roc_auc, prec, recall, prec_rec_auc,
                                         ave_prec])

                # Update ROC plot variable lists needed to plot all CVs in one ROC plot
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                aucs.append(roc_auc)

                # Update PRC plot variable lists needed to plot all CVs in one PRC plot
                precs.append(np.interp(mean_recall, recall, prec))
                praucs.append(prec_rec_auc)
                aveprecs.append(ave_prec)

                if master_list is None:
                    # Format feature importance scores as list
                    # (takes into account that all features are not in each CV partition)
                    temp_list = []
                    j = 0
                    headers = pd.read_csv(
                        self.full_path + '/CVDatasets/' + self.data_name
                        + '_CV_' + str(cv_count) + '_Test.csv').columns.values.tolist()
                    if self.instance_label is not None:
                        if self.instance_label in headers:
                            headers.remove(self.instance_label)
                    headers.remove(self.outcome_label)
                    for each in self.original_headers:
                        # Check if current feature from original dataset was in the partition
                        if each in headers:
                            # Deal with features not being in original order (find index of current feature list.index()
                            f_index = headers.index(each)
                            temp_list.append(fi[f_index])
                        else:
                            temp_list.append(0)
                        j += 1
                    fi_all.append(temp_list)

            logging.info("Running stats on " + algorithm)

            # Define values for the mean ROC line (mean of individual CVs)
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(aucs)
            if self.plot_roc:
                self.do_model_roc(algorithm, tprs, aucs, mean_fpr, alg_result_table)

            # Define values for the mean PRC line (mean of individual CVs)
            mean_prec = np.mean(precs, axis=0)
            mean_pr_auc = np.mean(praucs)
            if self.plot_prc:
                if master_list is None:
                    self.do_model_prc(algorithm, precs, praucs, mean_recall, alg_result_table)
                else:
                    self.do_model_prc(algorithm, precs, praucs, mean_recall, alg_result_table, rep_data, True)

            # Export and save all CV metric stats for each individual algorithm
            results = {'Balanced Accuracy': s_bac, 'Accuracy': s_ac, 'F1 Score': s_f1, 'Sensitivity (Recall)': s_re,
                       'Specificity': s_sp, 'Precision (PPV)': s_pr, 'TP': s_tp, 'TN': s_tn, 'FP': s_fp, 'FN': s_fn,
                       'NPV': s_npv, 'LR+': s_lrp, 'LR-': s_lrm, 'ROC AUC': aucs, 'PRC AUC': praucs,
                       'PRC APS': aveprecs}
            dr = pd.DataFrame(results)
            filepath = self.full_path + '/model_evaluation/' + self.abbrev[algorithm] + "_performance.csv"
            dr.to_csv(filepath, header=True, index=False)
            metric_dict[algorithm] = results

            # Save Median FI Stats
            if master_list is None:
                self.save_fi(fi_all, self.abbrev[algorithm], self.original_headers)

            # Store ave metrics for creating global ROC and PRC plots later
            mean_ave_prec = np.mean(aveprecs)
            # result_dict = {'algorithm':algorithm,'fpr':mean_fpr, 'tpr':mean_tpr,
            #                'auc':mean_auc, 'prec':mean_prec, 'pr_auc':mean_pr_auc,
            #                'ave_prec':mean_ave_prec}
            result_dict = {'algorithm': algorithm, 'fpr': mean_fpr, 'tpr': mean_tpr,
                           'auc': mean_auc, 'prec': mean_prec, 'recall': mean_recall,
                           'pr_auc': mean_pr_auc, 'ave_prec': mean_ave_prec}
            result_table.append(result_dict)

        # Result table later used to create global ROC an PRC plots comparing average ML algorithm performance.
        result_table = pd.DataFrame.from_dict(result_table)
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

    def do_model_roc(self, algorithm, tprs, aucs, mean_fpr, alg_result_table):

        # Define values for the mean ROC line (mean of individual CVs)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)

        # Generate ROC Plot (including individual CV's lines, average line, and no skill line)
        # based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

        if self.plot_roc:
            # Set figure dimensions
            plt.rcParams["figure.figsize"] = (6, 6)
            # Plot individual CV ROC lines
            for i in range(self.cv_partitions):
                plt.plot(alg_result_table[i][0], alg_result_table[i][1], lw=1, alpha=0.3,
                         label='ROC fold %d (AUC = %0.3f)' % (i, alg_result_table[i][2]))
            # Plot no-skill line
            plt.plot([0, 1], [0, 1],
                     linestyle='--', lw=2, color='black', label='No-Skill', alpha=.8)
            # Plot average line for all CVs
            std_auc = np.std(aucs)  # AUC standard deviations across CVs
            plt.plot(mean_fpr, mean_tpr, color=self.colors[algorithm],
                     label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (float(mean_auc), float(std_auc)),
                     lw=2, alpha=.8)

            # Plot standard deviation grey zone of curves
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
            # Specify plot axes,labels, and legend
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
            # Export and/or show plot
            plt.savefig(self.full_path + '/model_evaluation/' +
                        self.abbrev[algorithm] + "_ROC.png", bbox_inches="tight")
            if self.show_plots:
                plt.show()
            else:
                plt.close('all')
                # plt.cla() # not required

    def do_model_prc(self, algorithm, precs, praucs, mean_recall, alg_result_table, rep_data=None, replicate=False):
        # Define values for the mean PRC line (mean of individual CVs)
        mean_prec = np.mean(precs, axis=0)
        mean_pr_auc = np.mean(praucs)

        # Generate PRC Plot (including individual CV's lines, average line, and no skill line)
        if self.plot_prc:
            # Set figure dimensions
            plt.rcParams["figure.figsize"] = (6, 6)
            # Plot individual CV PRC lines
            for i in range(self.cv_partitions):
                plt.plot(alg_result_table[i][4], alg_result_table[i][3], lw=1, alpha=0.3,
                         label='PRC fold %d (AUC = %0.3f)' % (i, alg_result_table[i][5]))
            # Estimate no skill line based on the fraction of cases found in the first test dataset
            # Technically there could be a unique no-skill line for each CV dataset based
            # on final class balance (however only one is needed, and stratified CV attempts
            # to keep partitions with similar/same class balance)

            if not replicate:
                # Estimate no skill line based on the fraction of cases found in the first test dataset
                test = pd.read_csv(
                    self.full_path + '/CVDatasets/' + self.data_name + '_CV_0_Test.csv')

                test_y = test[self.outcome_label].values
            else:
                test_y = rep_data[self.outcome_label].values

            no_skill = len(test_y[test_y == 1]) / len(test_y)  # Fraction of cases
            # Plot no-skill line
            plt.plot([0, 1], [no_skill, no_skill], color='black', linestyle='--', label='No-Skill', alpha=.8)
            # Plot average line for all CVs
            std_pr_auc = np.std(praucs)
            plt.plot(mean_recall, mean_prec, color=self.colors[algorithm],
                     label=r'Mean PRC (AUC = %0.3f $\pm$ %0.3f)' % (float(mean_pr_auc), float(std_pr_auc)),
                     lw=2, alpha=.8)
            # Plot standard deviation grey zone of curves
            std_prec = np.std(precs, axis=0)
            precs_upper = np.minimum(mean_prec + std_prec, 1)
            precs_lower = np.maximum(mean_prec - std_prec, 0)
            plt.fill_between(mean_recall, precs_lower, precs_upper, color='grey',
                             alpha=.2, label=r'$\pm$ 1 std. dev.')
            # Specify plot axes,labels, and legend
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('Recall (Sensitivity)')
            plt.ylabel('Precision (PPV)')
            plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
            # Export and/or show plot
            plt.savefig(self.full_path + '/model_evaluation/' +
                        self.abbrev[algorithm] + "_PRC.png", bbox_inches="tight")
            if self.show_plots:
                plt.show()
            else:
                plt.close('all')
                # plt.cla() # not required

    def do_plot_roc(self, result_table):
        """
        Generate ROC plot comparing average ML algorithm performance
        (over all CV training/testing sets)
        """
        count = 0
        # Plot curves for each individual ML algorithm
        for i in result_table.index:
            # plt.plot(result_table.loc[i]['fpr'],result_table.loc[i]['tpr'],
            #          color=colors[i],label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
            plt.plot(result_table.loc[i]['fpr'], result_table.loc[i]['tpr'], color=self.colors[i],
                     label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
            count += 1
        # Set figure dimensions
        plt.rcParams["figure.figsize"] = (6, 6)
        # Plot no-skill line
        plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='No-Skill', alpha=.8)
        # Specify plot axes,labels, and legend
        plt.xticks(np.arange(0.0, 1.1, step=0.1))
        plt.xlabel("False Positive Rate", fontsize=15)
        plt.yticks(np.arange(0.0, 1.1, step=0.1))
        plt.ylabel("True Positive Rate", fontsize=15)
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        # Export and/or show plot
        plt.savefig(self.full_path + '/model_evaluation/Summary_ROC.png', bbox_inches="tight")
        if self.show_plots:
            plt.show()
        else:
            plt.close('all')
            # plt.cla() # not required

    def do_plot_prc(self, result_table, rep_data=None, replicate=False):
        """
        Generate PRC plot comparing average ML algorithm performance
        (over all CV training/testing sets)
        """
        count = 0
        # Plot curves for each individual ML algorithm
        for i in result_table.index:
            plt.plot(result_table.loc[i]['recall'], result_table.loc[i]['prec'], color=self.colors[i],
                     label="{}, AUC={:.3f}, APS={:.3f}".format(i, result_table.loc[i]['pr_auc'],
                                                               result_table.loc[i]['ave_prec']))
            count += 1

        if not replicate:
            # Estimate no skill line based on the fraction of cases found in the first test dataset
            test = pd.read_csv(self.full_path + '/CVDatasets/' + self.data_name + '_CV_0_Test.csv')
            if self.instance_label is not None:
                test = test.drop(self.instance_label, axis=1)
            test_y = test[self.outcome_label].values
        else:
            test_y = rep_data[self.outcome_label].values

        no_skill = len(test_y[test_y == 1]) / len(test_y)  # Fraction of cases

        # Plot no-skill line
        plt.plot([0, 1], [no_skill, no_skill], color='black', linestyle='--', label='No-Skill', alpha=.8)
        # Specify plot axes,labels, and legend
        plt.xticks(np.arange(0.0, 1.1, step=0.1))
        plt.xlabel("Recall (Sensitivity)", fontsize=15)
        plt.yticks(np.arange(0.0, 1.1, step=0.1))
        plt.ylabel("Precision (PPV)", fontsize=15)
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        # Export and/or show plot
        plt.savefig(self.full_path + '/model_evaluation/Summary_PRC.png', bbox_inches="tight")
        if self.show_plots:
            plt.show()
        else:
            plt.close('all')
            # plt.cla() # not required

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

    def metric_boxplots(self, metrics, metric_dict):
        """
        Export boxplots comparing algorithm performance for each standard metric
        """
        if not os.path.exists(self.full_path + '/model_evaluation/metricBoxplots'):
            os.mkdir(self.full_path + '/model_evaluation/metricBoxplots')
        for metric in metrics:
            temp_list = []
            for algorithm in self.algorithms:
                temp_list.append(metric_dict[algorithm][metric])
            td = pd.DataFrame(temp_list)
            td = td.transpose().astype('float')

            td.columns = self.algorithms

            # Generate boxplot
            td.plot(kind='box', rot=90)
            # Specify plot labels
            plt.ylabel(str(metric))
            plt.xlabel('ML Algorithm')
            # Export and/or show plot
            plt.savefig(self.full_path +
                        '/model_evaluation/metricBoxplots/Compare_' + metric + '.png', bbox_inches="tight")
            if self.show_plots:
                plt.show()
            else:
                plt.close('all')
                # plt.cla() # not required

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

    def prep_fi(self, metric_dict, metric='Median'):
        """
        Organizes and prepares model feature importance
        data for boxplot and composite feature importance figure generation.
        """
        # Initialize required lists
        # algorithm feature importance dataframe list (used to generate FI boxplot for each algorithm)
        fi_df_list = []
        # algorithm feature importance medians list (used to generate composite FI barplots)
        fi_med_list = []
        # algorithm focus metric medians list (used in weighted FI viz)
        med_metric_list = []
        # list of pre-feature selection feature names as they appear in FI reports for each algorithm
        all_feature_list = []

        # Get necessary feature importance data and primary metric data
        # (currently only 'balanced accuracy' can be used for this)
        for algorithm in self.algorithms:
            # Get relevant feature importance info
            temp_df = pd.read_csv(self.full_path + '/model_evaluation/feature_importance/' + self.abbrev[
                algorithm] + "_FI.csv")  # CV FI scores for all original features in dataset.
            # Should be same for all algorithm files (i.e. all original features in standard CV dataset order)
            if algorithm == self.algorithms[0]:
                all_feature_list = temp_df.columns.tolist()
            fi_df_list.append(temp_df)
            if metric == 'Median':
                fi_med_list.append(temp_df.median().tolist())  # Saves median FI scores over CV runs
            elif metric == 'Mean':
                fi_med_list.append(temp_df.mean().tolist())  # Saves mean FI scores over CV runs
            # Get relevant metric info
            if metric == 'Median':
                med_ba = median(metric_dict[algorithm][self.metric_weight])
            elif metric == 'Mean':
                med_ba = mean(metric_dict[algorithm][self.metric_weight])
            med_metric_list.append(med_ba)
        # Normalize Median Feature importance scores, so they fall between (0 - 1)
        fi_med_norm_list = []
        for each in fi_med_list:  # each algorithm
            norm_list = []
            for i in range(len(each)):  # each feature (score) in original data order
                if each[i] <= 0:  # Feature importance scores assumed to be uninformative if at or below 0
                    norm_list.append(0)
                else:
                    norm_list.append((each[i]) / (max(each)))
            fi_med_norm_list.append(norm_list)

        # Identify features with non-zero medians
        # (step towards excluding features that had zero feature importance for all algorithms)
        alg_non_zero_fi_list = []  # stores list of feature name lists that are non-zero for each algorithm
        for each in fi_med_list:  # each algorithm
            temp_non_zero_list = []
            for i in range(len(each)):  # each feature
                if each[i] > 0.0:
                    # add feature names with positive values (doesn't need to be normalized for this)
                    temp_non_zero_list.append(all_feature_list[i])
            alg_non_zero_fi_list.append(temp_non_zero_list)
        non_zero_union_features = alg_non_zero_fi_list[0]  # grab first algorithm's list
        # Identify union of features with non-zero averages over all algorithms
        # (i.e. if any algorithm found a non-zero score it will be considered
        # for inclusion in top feature visualizations)
        for j in range(1, len(self.algorithms)):
            non_zero_union_features = list(set(non_zero_union_features) | set(alg_non_zero_fi_list[j]))
        non_zero_union_indexes = []
        for i in non_zero_union_features:
            non_zero_union_indexes.append(all_feature_list.index(i))
        # return fi_df_list, fi_ave_list, fi_ave_norm_list, ave_metric_list,\
        #     all_feature_list, non_zero_union_features, non_zero_union_indexes
        return fi_df_list, fi_med_list, fi_med_norm_list, med_metric_list, \
            all_feature_list, non_zero_union_features, non_zero_union_indexes

    def select_for_composite_viz(self, non_zero_union_features,
                                 non_zero_union_indexes,
                                 ave_metric_list, fi_ave_norm_list):
        """
        Identify list of top features over all algorithms to visualize
        (note that best features to visualize are chosen using algorithm
        performance weighting and normalization:
        frac plays no useful role here only for viz). All features included
        if there are fewer than 'top_model_features'. Top features are
        determined by the sum of performance
        (i.e. balanced accuracy) weighted feature importance over all algorithms.
        """
        # Create performance weighted score sum dictionary for all features
        score_sum_dict = {}
        i = 0
        for each in non_zero_union_features:  # for each non-zero feature
            for j in range(len(self.algorithms)):  # for each algorithm
                # grab target score from each algorithm
                score = fi_ave_norm_list[j][non_zero_union_indexes[i]]
                # multiply score by algorithm performance weight
                weight = ave_metric_list[j]
                # if weight <= .5:  # This is why this method is limited to balanced_accuracy and roc_auc
                #     weight = 0
                # if not weight == 0:
                #     weight = (weight - 0.5) / 0.5
                score = score * weight
                if not (each in score_sum_dict):
                    score_sum_dict[each] = score
                else:
                    score_sum_dict[each] += score
            i += 1
        # Sort features by decreasing score
        score_sum_dict_features = sorted(score_sum_dict, key=lambda x: score_sum_dict[x], reverse=True)
        # Keep all features if there are fewer than specified top results
        if len(non_zero_union_features) > self.top_features:
            features_to_viz = score_sum_dict_features[0:self.top_features]
        else:
            features_to_viz = score_sum_dict_features
        return features_to_viz  # list of feature names to visualize in composite FI plots.

    def do_fi_boxplots(self, fi_df_list, fi_med_list, metric='Median'):
        """
        Generate individual feature importance boxplot for each algorithm
        """
        algorithm_counter = 0
        for algorithm in self.algorithms:  # each algorithms
            # Make median feature importance score dictionary
            score_dict = {}
            counter = 0
            for med_score in fi_med_list[algorithm_counter]:  # each feature
                score_dict[self.original_headers[counter]] = med_score
                counter += 1
            # Sort features by decreasing score
            score_dict_features = sorted(score_dict, key=lambda x: score_dict[x], reverse=True)
            # Make list of feature names to visualize
            if len(self.original_headers) > self.top_features:
                features_to_viz = score_dict_features[0:self.top_features]
            else:
                features_to_viz = score_dict_features
            # FI score dataframe for current algorithm
            df = fi_df_list[algorithm_counter]
            # Subset of dataframe (in ranked order) to visualize
            viz_df = df[features_to_viz]
            # Generate Boxplot
            plt.figure(figsize=(15, 4))
            viz_df.boxplot(rot=90)
            plt.title(algorithm)
            plt.ylabel(metric + ' Feature Importance')
            plt.xlabel('Features')
            plt.xticks(np.arange(1, len(features_to_viz) + 1), features_to_viz, rotation='vertical')
            plt.savefig(self.full_path +
                        '/model_evaluation/'
                        'feature_importance/' + algorithm + '_boxplot', bbox_inches="tight")
            if self.show_plots:
                plt.show()
            else:
                plt.close('all')
                # plt.cla() # not required
            # Identify and sort (decreasing) features with top median FI
            algorithm_counter += 1

    def do_fi_histogram(self, fi_med_list, metric='Median'):
        """
        Generate histogram showing distribution of median feature importance scores for each algorithm.
        """
        algorithm_counter = 0
        for algorithm in self.algorithms:  # each algorithms
            med_scores = fi_med_list[algorithm_counter]
            # Plot a histogram of average feature importance
            plt.hist(med_scores, bins=100)
            plt.xlabel(metric + " Feature Importance")
            plt.ylabel("Frequency")
            plt.title("Histogram of" + metric + " Feature Importance for " + str(algorithm))
            plt.xticks(rotation='vertical')
            plt.savefig(self.full_path
                        + '/model_evaluation/'
                          'feature_importance/' + algorithm + '_histogram', bbox_inches="tight")
            if self.show_plots:
                plt.show()
            else:
                plt.close('all')
                # plt.cla() # not required

    def composite_fi_plot(self, fi_list, all_feature_list_to_viz, fig_name,
                          y_label_text):
        """
        Generate composite feature importance plot given list of feature names
        and associated feature importance scores for each algorithm.
        This is run for different transformations of the normalized feature importance scores.
        """
        alg_colors = list(self.colors.values())
        # Set basic plot properties
        rc('font', weight='bold', size=16)
        # The position of the bars on the x-axis
        r = all_feature_list_to_viz  # feature names
        # Set width of bars
        bar_width = 0.75
        # Set figure dimensions
        plt.figure(figsize=(24, 12))
        # Plot first algorithm FI scores (lowest) bar
        p1 = plt.bar(r, fi_list[0], color=alg_colors[0], edgecolor='white', width=bar_width)
        # Automatically calculate space needed to plot next bar on top of the one before it
        bottoms = []  # list of space used by previous
        # algorithms for each feature (so next bar can be placed directly above it)
        bottom = None
        for i in range(len(self.algorithms) - 1):
            for j in range(i + 1):
                if j == 0:
                    bottom = np.array(fi_list[0]).astype('float64')
                else:
                    bottom += np.array(fi_list[j]).astype('float64')
            bottoms.append(bottom)
        if not isinstance(bottoms, list):
            bottoms = bottoms.tolist()
        if len(self.algorithms) > 1:
            # Plot subsequent feature bars for each subsequent algorithm
            ps = [p1[0]]
            for i in range(len(self.algorithms) - 1):
                p = plt.bar(r, fi_list[i + 1], bottom=bottoms[i], color=alg_colors[i + 1], edgecolor='white',
                            width=bar_width)
                ps.append(p[0])
            lines = tuple(ps)
        else:
            ps = [p1[0]]
            lines = tuple(ps)
        # Specify axes info and legend
        plt.xticks(np.arange(len(all_feature_list_to_viz)), all_feature_list_to_viz, rotation='vertical')
        plt.xlabel("Feature", fontsize=20)
        plt.ylabel(y_label_text, fontsize=20)
        # plt.legend(lines[::-1], algorithms[::-1],loc="upper left", bbox_to_anchor=(1.01,1)) #legend outside plot
        plt.legend(lines[::-1], self.algorithms[::-1], loc="upper right")
        # Export and/or show plot
        plt.savefig(self.full_path + '/model_evaluation/feature_importance/Compare_FI_' + fig_name + '.png',
                    bbox_inches='tight')
        if self.show_plots:
            plt.show()
        else:
            plt.close('all')
            # plt.cla() # not required

    def get_fi_to_viz_sorted(self, features_to_viz, all_feature_list, fi_med_norm_list):
        """
        Takes a list of top features names for visualisation, gets their
        indexes. In every composite FI plot features are ordered the same way
        they are selected for visualisation (i.e. normalized and performance
        weighted). Because of this feature bars are only perfectly ordered in
        descending order for the normalized + performance weighted composite plot.
        """
        # Get original feature indexs for selected feature names
        feature_index_to_viz = []  # indexes of top features
        for i in features_to_viz:
            feature_index_to_viz.append(all_feature_list.index(i))
        # Create list of top feature importance values in original dataset feature order
        top_fi_med_norm_list = []  # feature importance values of top features for each algorithm (list of lists)
        for i in range(len(self.algorithms)):
            temp_list = []
            for j in feature_index_to_viz:  # each top feature index
                temp_list.append(fi_med_norm_list[i][j])  # add corresponding FI value
            top_fi_med_norm_list.append(temp_list)
        all_feature_list_to_viz = features_to_viz
        return top_fi_med_norm_list, all_feature_list_to_viz

    @staticmethod
    def frac_fi(top_fi_med_norm_list):
        """
        Transforms feature scores so that they sum to 1 over all features
        for a given algorithm.  This way the normalized and fracionated composit bar plot
        offers equal total bar area for every algorithm. The intuition
        here is that if an algorithm gives the same FI scores for all top features it won't be
        overly represented in the resulting plot (i.e. all features can
        have the same maximum feature importance which might lead to the impression that an
        algorithm is working better than it is.) Instead, that maximum
        'bar-real-estate' has to be divided by the total number of features. Notably, this
        transformation has the potential to alter total algorithm FI bar height ranking of features.
        """
        frac_lists = []
        for each in top_fi_med_norm_list:  # each algorithm
            frac_list = []
            for i in range(len(each)):  # each feature
                if sum(each) == 0:  # check that all feature scores are not zero to avoid zero division error
                    frac_list.append(0)
                else:
                    frac_list.append((each[i] / (sum(each))))
            frac_lists.append(frac_list)
        return frac_lists

    @staticmethod
    def weight_fi(med_metric_list, top_fi_med_norm_list):
        """
        Weights the feature importance scores by algorithm performance
        (intuitive because when interpreting feature importances we want
        to place more weight on better performing algorithms)
        """
        # Prepare weights
        weights = []
        # replace all balanced accuraces <=.5 with 0 (i.e. these are no better than random chance)
        for i in range(len(med_metric_list)):
            if med_metric_list[i] <= .5:
                med_metric_list[i] = 0
        # normalize balanced accuracies
        for i in range(len(med_metric_list)):
            if med_metric_list[i] == 0:
                weights.append(0)
            else:
                weights.append((med_metric_list[i] - 0.5) / 0.5)
        # Weight normalized feature importances
        weighted_lists = []
        for i in range(len(top_fi_med_norm_list)):  # each algorithm
            weight_list = np.multiply(weights[i], top_fi_med_norm_list[i]).tolist()
            weighted_lists.append(weight_list)
        return weighted_lists, weights

    @staticmethod
    def weight_frac_fi(frac_lists, weights):
        """ Weight normalized and fractionated feature importances. """
        weighted_frac_lists = []
        for i in range(len(frac_lists)):
            weight_list = np.multiply(weights[i], frac_lists[i]).tolist()
            weighted_frac_lists.append(weight_list)
        return weighted_frac_lists

    def parse_runtime(self):
        """
        Loads runtime summaries from entire pipeline and parses them into a single summary file.
        """
        dict_obj = dict()
        dict_obj['preprocessing'] = 0
        for file_path in glob.glob(self.full_path + '/runtime/*.txt'):
            file_path = str(Path(file_path).as_posix())
            f = open(file_path, 'r')
            val = float(f.readline())
            ref = file_path.split('/')[-1].split('_')[1].split('.')[0]
            if ref in self.abbrev:
                ref = self.abbrev[ref]
            if not (ref in dict_obj):
                if 'preprocessing' in ref:
                    dict_obj['preprocessing'] += val
                dict_obj[ref] = val
            else:
                dict_obj[ref] += val
        with open(self.full_path + '/runtimes.csv', mode='w', newline="") as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Pipeline Component", "Time (sec)"])
            writer.writerow(["Exploratory Analysis", dict_obj['exploratory']])
            writer.writerow(["Preprocessing", dict_obj['preprocessing']])
            try:
                writer.writerow(["Mutual Information", dict_obj['mutualinformation']])
            except KeyError:
                pass
            try:
                writer.writerow(["MultiSURF", dict_obj['multisurf']])
            except KeyError:
                pass
            writer.writerow(["Feature Selection", dict_obj['featureselection']])
            for algorithm in self.algorithms:  # Report runtimes for each algorithm
                writer.writerow(([algorithm, dict_obj[self.abbrev[algorithm]]]))
            writer.writerow(["Stats Summary", dict_obj['Stats']])

    def save_runtime(self):
        """
        Save phase runtime
        """
        runtime_file = open(self.full_path + '/runtime/runtime_Stats.txt', 'w')
        runtime_file.write(str(time.time() - self.job_start_time))
        runtime_file.close()
