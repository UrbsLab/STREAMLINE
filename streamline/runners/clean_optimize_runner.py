import logging
import os
import pandas as pd
import optuna
from streamline.runners.auto_runner import AutoRunner

class OptimizeClean:


    def __init__(self, dataset_name: str, optimize_for: str = 'auroc', cv_folds: int = 1, opt_direction: str = 'maximize',
                data_path: str = "./data/DemoData", output_path: str="./DemoOutput", experiment_name: str='demo_experiment', sampler_type='TPE_sampler'):

        self.dataset = list(dataset_name)
        self.optimize_for = optimize_for
        self.cv_folds = cv_folds
        self.opt_direction = opt_direction
        self.data_path = data_path
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.sampler_type = sampler_type

    def run(self, run_para=False):
        


        
        '''
        self.categorical_cutoff = categorical_cutoff  # (int) Bumber of unique values after which a variable is considered to be quantitative vs categorical 'Optuna'
        self.sig_cutoff = sig_cutoff # (float, 0-1) Significance cutoff used throughout pipeline
        self.featureeng_missingness = featureeng_missingness# (float, 0-1) Percentage of missing after which categorical featrure identifier is generated.'Optuna'
        self.cleaning_missingness = cleaning_missingness# (float, 0-1) Percentage of missing after instance and feature removal is performed. 'Optuna'
        self.correlation_removal_threshold = correlation_removal_threshold # (float, 0-1) 'Optuna'
        self.exploration_list = exploration_list  # (list of strings) Options:["Describe", "Differentiate", "Univariate Analysis"] 'Optuna'
        self.n_splits = n_splits# (int, > 1) Number of training/testing data partitions to create - and resulting number of models generated using each ML algorithm 'Optuna'
        self.partition_method = partition_method ## (str) for Stratified, Random, or Group, respectively 'Optuna'
        '''
        