import logging
import os
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
from streamline.runners.auto_runner import AutoRunner
from streamline.runners.clean_runner import CleanRunner



class OptimizeClean:


    def __init__(self, dataset_name: str, class_label: str ='y', instance_label: str ='InstanceId', optimize_for: str = 'ROC AUC', cv_folds: int = 1, opt_direction: str = 'maximize',
                data_path: str = "./data/DemoData", output_path: str="./DemoOutput", experiment_name: str='demo_experiment', sampler_type='TPE_sampler', n_trials=100):

        self.dataset = dataset_name
        self.optimize_for = optimize_for #options: 'Balanced Accuracy', 'Accuracy', 'F1 Score', 'ROC AUC', 'PRC AUC', 'PRC APS' 
        self.cv_folds = cv_folds
        self.opt_direction = opt_direction
        self.data_path = data_path
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.sampler_type = sampler_type
        self.class_label = class_label
        self.instance_label=instance_label
        self.n_trials = n_trials

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
            self.param = {
                'categorical_cutoff': categorical_cutoff,
                'sig_cutoff': sig_cutoff,
                'featureeng_missingness': featureeng_missingness,
                'cleaning_missingness': cleaning_missingness,
                'correlation_removal_threshold': correlation_removal_threshold,
                'exploration_list': exploration_list,
                'partition_method': partition_method,
                'n_splits': n_splits
            }
        
            try:
                self.most_recent_run = AutoRunner(dataset_names=self.dataset,output_path=self.output_path, experiment_name=self.experiment_name, gen_report=False, clean=False, categorical_cutoff=categorical_cutoff, sig_cutoff=sig_cutoff, featureeng_missingness=featureeng_missingness, cleaning_missingness=cleaning_missingness, correlation_removal_threshold=correlation_removal_threshold, exploration_list=exploration_list, partition_method=partition_method, n_splits=n_splits, class_label=self.class_label, instance_label=self.instance_label, ml_algorithms=["NB", "LR", "DT", "EN", "RF", "GB", "XGB", "LGB", "CGB", "SVM"], exclude=["ANN","KNN","GP", 'eLCS', 'XCS', "ExSTraCS"])
                output_csv = self.most_recent_run.run(run_para=False)
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
            except:
                print('EXCEPTION')
                self.goal = float(0.0)
                return self.goal
        
        study = optuna.create_study(sampler=TPESampler(), direction=self.opt_direction)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        