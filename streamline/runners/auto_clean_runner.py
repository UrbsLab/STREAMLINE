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

class AutoRunner: 

    def __init__(self, data_path: str = "./data/DemoData", output_path="./DemoOutput",
                experiment_name='demo_experiment', exploration_list=["Describe", "Univariate Analysis","Differentiate", "Feature Correlation"],
                plot_list=["Describe", "Univariate Analysis", "Feature Correlation"],
                class_label="Class", instance_label='InstanceID', match_label=None, n_splits=3, partition_method="Stratified",
                ignore_features=None, categorical_feature_headers=None, quantitative_feature_headers=None, top_features=20,
                categorical_cutoff=10, sig_cutoff=0.05, featureeng_missingness=0.5, cleaning_missingness=0.5,
                correlation_removal_threshold=1.0,
                random_state=None, run_cluster=False, queue='defq', reserved_memory=4, show_plots=False):
        
        #must input: 

        #Dataprocess_runner 
        self.data_path = data_path #(str) Data Folder Path
        self.output_path = output_path # (str) Ouput Folder Path (folder will be created by STREAMLINE automatically)
        self.experiment_name = experiment_name # (str) Experiment Name (change to save a new STREAMLINE run output folder instead of overwriting previous run)
        self.class_label = class_label  # (str) i.e. class outcome column label
        self.instance_label = instance_label # (str) If data includes instance labels, given respective column name here, otherwise put 'None'
        self.match_label = match_label # (str or None) Only applies when M selected for partition-method; indicates column label with matched instance ids'
        self.ignore_features = ignore_features # list of column names (given as string values) to exclude from the analysis (only insert column names if needed, otherwise leave empty)
        self.categorical_features = categorical_feature_headers # empty list for 'auto-detect' otherwise list feature names (given as string values) to be treated as categorical. Only impacts algorithms that can take variable type into account.
        self.quantitative_features = quantitative_feature_headers
        self.plot_list = plot_list
        self.top_features = top_features #(int) Number of top features to report in notebook for univariate analysis
        self.run_cluster = run_cluster
        self.queue = queue
        self.reserved_memory = reserved_memory
        self.show_plots = show_plots
        #auto_cleaning optuna_targets
        self.categorical_cutoff = categorical_cutoff  # (int) Bumber of unique values after which a variable is considered to be quantitative vs categorical 'Optuna'
        self.sig_cutoff = sig_cutoff # (float, 0-1) Significance cutoff used throughout pipeline
        self.featureeng_missingness = featureeng_missingness# (float, 0-1) Percentage of missing after which categorical featrure identifier is generated.'Optuna'
        self.cleaning_missingness = cleaning_missingness# (float, 0-1) Percentage of missing after instance and feature removal is performed. 'Optuna'
        self.correlation_removal_threshold = correlation_removal_threshold # (float, 0-1) 'Optuna'
        self.exploration_list = exploration_list  # (list of strings) Options:["Describe", "Differentiate", "Univariate Analysis"] 'Optuna'
        self.n_splits = n_splits# (int, > 1) Number of training/testing data partitions to create - and resulting number of models generated using each ML algorithm 'Optuna'
        self.partition_method = partition_method ## (str) for Stratified, Random, or Group, respectively 'Optuna'
    
        

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
