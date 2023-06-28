import logging
import os
import shutil
import glob
import pickle
import time
import dask
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
import optuna
from streamline.runners.dataprocess_runner import DataProcessRunner
from streamline.runners.imputation_runner import ImputationRunner
from streamline.runners.feature_runner import FeatureImportanceRunner
from streamline.runners.feature_runner import FeatureSelectionRunner
from streamline.runners.model_runner import ModelExperimentRunner
from streamline.runners.stats_runner import StatsRunner

class AutoCleanRunner: 

    def __init__(self):

        #Dataprocess_runner 
        self.data_path = data_path
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.class_label = class_label
        self.instance_label = instance_label
        self.match_label = match_label
        self.ignore_features = ignore_features
        self.categorical_cutoff = 'Optuna'
        self.categorical_features = categorical_features
        self.quantitative_features = quantitative_features
        self.featureeng_missingness = 'Optuna'
        self.cleaning_missingness = 'Optuna'
        self.correlation_removal_threshold = 'Optuna'
        self.top_features = top_features
        self.exploration_list = 'Optuna'
        self.plot_list = plot_list
        self.n_splits = 'Optuna'
        self.partition_method = 'Optuna'
        self.run_cluster = run_cluster
        self.queue = queue
        self.reserved_memory = reserved_memory
        self.show_plots = show_plots

        #ImputationRunner
        self.scale_data = scale_data
        self.impute_data = impute_data
        self.multi_impute = multi_impute
        self.overwrite_cv = overwrite_cv
        self.random_state = random_state
        
        #FeatureImportanceRunner
        self.cv_count = None
        self.dataset = None
        self.instance_subset = instance_subset
        self.algorithms = list(algorithms)
        self.use_turf = use_turf
        self.turf_pct = turf_pct
        self.n_jobs = n_jobs

        #FeatureSelectionRunner
        self.cv_count = None
        self.dataset = None
        self.max_features_to_keep = max_features_to_keep
        self.filter_poor_features = filter_poor_features
        self.export_scores = export_scores

        #ModelExperimentRunner
        """
        Args:
            algorithms: list of str of ML models to run
            scoring_metric: primary scikit-learn specified scoring metric used for hyperparameter optimization and \
                            permutation-based model feature importance evaluation, default='balanced_accuracy'
            metric_direction: direction to optimize the scoring metric in optuna, \
                              either 'maximize' or 'minimize', default='maximize'
            training_subsample: for long running algos (XGB,SVM,ANN,KNN), option to subsample training set \
                                (0 for no subsample, default=0)
            use_uniform_fi: overrides use of any available feature importance estimate methods from models, \
                            instead using permutation_importance uniformly, default=True
            n_trials: number of bayesian hyperparameter optimization trials using optuna \
                      (specify an integer or None) default=200
            timeout: seconds until hyperparameter sweep stops running new trials \
                    (Note: it may run longer to finish last trial started) \
                    If set to None, STREAMLINE is completely replicable, but will take longer to run \
                    default=900 i.e. 900 sec = 15 minutes default \
            save_plots: export optuna-generated hyperparameter sweep plots, default False
            do_lcs_sweep: do LCS hyper-param tuning or use below params, default=False
            lcs_nu: fixed LCS nu param (recommended range 1-10), set to larger value for data with \
                    less or no noise, default=1
            lcs_iterations: fixed LCS number of learning iterations param, default=200000
            lcs_n: fixed LCS rule population maximum size param, default=2000
            lcs_timeout: seconds until hyperparameter sweep stops for LCS algorithms, default=1200

        """
        self.scoring_metric = scoring_metric
        self.metric_direction = metric_direction
        self.training_subsample = training_subsample
        self.uniform_fi = use_uniform_fi
        self.n_trials = n_trials
        self.timeout = timeout
        self.save_plots = save_plots
        self.do_lcs_sweep = do_lcs_sweep
        self.lcs_nu = lcs_nu
        self.lcs_n = lcs_n
        self.lcs_iterations = lcs_iterations
        self.lcs_timeout = lcs_timeout
        self.resubmit = resubmit

        #StatsRunner
        """
        Args:
            output_path: path to output directory
            experiment_name: name of experiment (no spaces)
            algorithms: list of str of ML models to run
            scoring_metric='balanced_accuracy'
            sig_cutoff: significance cutoff, default=0.05
            metric_weight='balanced_accuracy'
            scale_data=True
            plot_roc: Plot ROC curves individually for each algorithm including all CV results and averages,
                                default=True
            plot_prc: Plot PRC curves individually for each algorithm including all CV results and averages,
                                default=True
            plot_metric_boxplots: Plot box plot summaries comparing algorithms for each metric, default=True
            plot_fi_box: Plot feature importance boxplots and histograms for each algorithm, default=True
            metric_weight: ML model metric used as weight in composite FI plots \
                           (only supports balanced_accuracy or roc_auc as options). \
                           Recommend setting the same as primary_metric if possible, \
                           default='balanced_accuracy'
            top_features: number of top features to illustrate in figures, default=40
            show_plots: flag to show plots

        """
        self.scale_data = scale_data
        self.stats_scoring_metric = stats_scoring_metric
        self.plot_roc = plot_roc
        self.plot_prc = plot_prc
        self.plot_metric_boxplots = plot_metric_boxplots
        self.plot_fi_box = plot_fi_box
        self.metric_weight = metric_weight
        
    
    #def run(self):
