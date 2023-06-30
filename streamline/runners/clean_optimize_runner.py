import logging
import os
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from streamline.runners.auto_runner import AutoRunner
from streamline.runners.clean_runner import CleanRunner

class OptimizeClean:


    def __init__(self, dataset_name: str, optimize_for: str = 'ROC AUC', cv_folds: int = 1, opt_direction: str = 'maximize',
                data_path: str = "./data/DemoData", output_path: str="./DemoOutput", experiment_name: str='demo_experiment', sampler_type='TPE_sampler'):

        self.dataset = list(dataset_name)
        self.optimize_for = optimize_for #options: 'Balanced Accuracy', 'Accuracy', 'F1 Score', 'ROC AUC', 'PRC AUC', 'PRC APS' 
        self.cv_folds = cv_folds
        self.opt_direction = opt_direction
        self.data_path = data_path
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.sampler_type = sampler_type

    def run(self, run_para=False):
        
        def objective(trial):
            categorical_cutoff = trial.suggest_int('categorical_cutoff',2,10)
            sig_cutoff = trial.suggest_float('sig_cutoff',0.005, 0.05, log=True)
            featureeng_missingness = trial.suggest_float('featureeng_missingness', 0.05, 1, step=0.05)
            cleaning_missingness = trial.suggest_float('cleaning_missingness', 0.05, 1, step=0.05)
            correlation_removal_threshold = trial.suggest_float('correlation_removal_threshold', 0.5, 1, step=0.05)
            exploration_list = trial.suggest_categorical('exploration_list', [["Describe", "Univariate Analysis", "Feature Correlation"]])
            partition_method = trial.suggest_categorical('partition_method',['Stratified', 'Random']) #Group not included
            n_splits = trial.suggest_int('n_splits', 2, 10)
            self.params = {
                'categorical_cutoff': categorical_cutoff,
                'sig_cutoff': sig_cutoff,
                'featureeng_missingness': featureeng_missingness,
                'cleaning_missingness': cleaning_missingness,
                'correlation_removal_threshold': correlation_removal_threshold,
                'exploration_list': exploration_list,
                'partition_method': partition_method,
                'n_splits': n_splits
            }
            self.most_recent_run = AutoRunner(dataset_names=self.dataset, gen_report=False, clean=False,
                                            categorical_cutoff=categorical_cutoff, sig_cutoff=sig_cutoff, featureeng_missingness=featureeng_missingness,
                                            cleaning_missingness=cleaning_missingness, correlation_removal_threshold=correlation_removal_threshold,
                                            exploration_list=exploration_list, partition_method=partition_method, n_splits=n_splits)
            output_csv = self.most_recent_run.run(run_para=run_para)
            performance = pd.read_csv(output_csv)
            self.summary_chart = performance
            self.goal = performance[self.optimize_for].max()
            self.best_model = performance.loc[performance[self.optimize_for].idxmax()][0]
            png_out = output_csv.removesuffix('Summary_performance_mean.csv')
            png_out = png_out + 'Summary_ROC.png'
            self.final_model_comparison = plt.savefig(png_out)
            clean = CleanRunner(self.output_path, self.experiment_name, del_time=True, del_old_cv=True)
            # run_parallel is not used in clean
            clean.run()
            return self.goal
        
        study = optuna.create_study(direction=self.opt_direction)
        study.optimize(objective, n_trials=1)

        





        
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
        