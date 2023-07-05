import logging
import os
import shutil
from streamline.runners.dataprocess_runner import DataProcessRunner
from streamline.runners.imputation_runner import ImputationRunner
from streamline.runners.feature_runner import FeatureImportanceRunner
from streamline.runners.feature_runner import FeatureSelectionRunner
from streamline.runners.model_runner import ModelExperimentRunner
from streamline.runners.stats_runner import StatsRunner
from streamline.runners.report_runner import ReportRunner
from streamline.runners.clean_runner import CleanRunner

class AutoRunner: 

    def __init__(self, dataset_names, gen_report=True, clean=True, data_path: str = "./data/DemoData", output_path: str="./DemoOutput",
                experiment_name: str='demo_experiment', exploration_list: list=["Describe", "Univariate Analysis", "Feature Correlation"],
                plot_list: list=["Describe", "Univariate Analysis", "Feature Correlation"],
                class_label:str="Class", instance_label:str='InstanceID', match_label=None, n_splits=3, partition_method="Stratified",
                ignore_features=None, categorical_feature_headers=None, quantitative_feature_headers=None, top_features=40,
                categorical_cutoff=10, sig_cutoff=0.05, featureeng_missingness=0.5, cleaning_missingness=0.5,
                correlation_removal_threshold=1.0,
                random_state=None, run_cluster=False, queue='defq', reserved_memory=4, show_plots=False,
                impute_scale_data=True, impute_data=True,
                impute_multi_impute=True, impute_overwrite_cv=True,
                do_mutual_info=True, do_multisurf=True,
                instance_subset=2000, algorithms=("MI", "MS"), use_turf=False, turf_pct=0.5,
                n_jobs=-1, max_features_to_keep=2000, filter_poor_features=True, export_scores=True,
                ml_algorithms=["NB", "LR", "DT", "XGB", "LGB", "CGB", "SVM","GB", "RF"], exclude=['eLCS', 'XCS'], scoring_metric='balanced_accuracy', metric_direction='maximize',
                training_subsample=0, use_uniform_fi=True, n_trials=200,
                timeout=900, do_lcs_sweep=False, lcs_nu=1, lcs_n=2000, lcs_iterations=200000,
                lcs_timeout=1200, resubmit=False,
                stats_scale_data=True, metric_weight='balanced_accuracy',
                plot_roc=True, plot_prc=True, plot_fi_box=True, plot_metric_boxplots=True, del_time=True, del_old_cv=True, save_plots=False):
        
        #must input: 

        #Dataprocess_runner 
        self.gen_report = gen_report
        self.clean = clean
        self.del_time = del_time
        self.del_old_cv = del_old_cv
        self.dataset_names = dataset_names
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
        self.impute_scale_data = impute_scale_data
        self.impute_data = impute_data
        self.impute_multi_impute = impute_multi_impute
        self.impute_overwrite_cv = impute_overwrite_cv
        self.random_state = random_state
        
        #FeatureImportanceRunner
        self.do_mutual_info = do_mutual_info
        self.do_multisurf = do_multisurf
        self.featureimp_cv_count = None
        self.featureimp_dataset = None
        self.featureimp_instance_subset = instance_subset # (int) Sample subset size to use with MultiSURF (since MultiSURF's compute time scales quadratically with instance count)
        self.featureimp_algorithms = list(algorithms)
        self.featureimp_use_turf = use_turf
        self.featureimp_turf_pct = turf_pct #future optuna
        self.featureimp_n_jobs = n_jobs # (int) Number of cores dedicated to running algorithm; setting to -1 will use all available cores when run locally


        #FeatureSelectionRunner
        self.cv_count = None
        self.dataset = None
        self.featuresel_max_features_to_keep = max_features_to_keep
        self.featuresel_filter_poor_features = filter_poor_features
        self.featuresel_export_scores = export_scores

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
        self.ml_algorithms = ml_algorithms
        self.exclude = exclude
        self.scoring_metric = scoring_metric
        self.metric_direction = metric_direction
        self.training_subsample = training_subsample
        self.use_uniform_fi = use_uniform_fi
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
        self.scale_data = stats_scale_data
        self.plot_roc = plot_roc
        self.plot_prc = plot_prc
        self.plot_metric_boxplots = plot_metric_boxplots
        self.plot_fi_box = plot_fi_box
        self.metric_weight = metric_weight
        
    def run(self, run_para=False):
        FORMAT = '%(levelname)s: %(message)s'
        logging.basicConfig(format=FORMAT)
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        if os.path.exists(self.output_path+'/'+self.experiment_name):
            shutil.rmtree(self.output_path+'/'+self.experiment_name)
        dpr = DataProcessRunner(data_path=self.data_path, output_path=self.output_path,
                experiment_name=self.experiment_name, exploration_list=self.exploration_list,
                plot_list=self.plot_list, class_label=self.class_label,
                instance_label=self.instance_label, match_label=self.match_label,
                n_splits=self.n_splits, partition_method=self.partition_method,
                ignore_features=self.ignore_features, categorical_features=self.categorical_features,
                quantitative_features=self.quantitative_features, top_features=self.top_features,
                categorical_cutoff=self.categorical_cutoff, sig_cutoff=self.sig_cutoff,
                featureeng_missingness=self.featureeng_missingness, cleaning_missingness=self.cleaning_missingness,
                correlation_removal_threshold=self.correlation_removal_threshold,
                random_state=self.random_state, run_cluster=self.run_cluster, queue=self.queue,
                reserved_memory=self.reserved_memory, show_plots=self.show_plots)
        dpr.run(run_parallel=run_para)
        ir = ImputationRunner(output_path=self.output_path, experiment_name=self.experiment_name, 
                        scale_data=self.impute_scale_data, impute_data=self.impute_data,
                        multi_impute=self.impute_multi_impute, overwrite_cv=self.impute_overwrite_cv, 
                        class_label=self.class_label, instance_label=self.instance_label, 
                        random_state=self.random_state)
        ir.run(run_parallel=run_para)
        self.featimp_algorithms = []
        if self.do_mutual_info:
            self.featimp_algorithms.append("MI")
        if self.do_multisurf:
            self.featimp_algorithms.append("MS")
        f_imp = FeatureImportanceRunner(output_path=self.output_path, experiment_name=self.experiment_name, 
                                class_label=self.class_label, 
                                instance_label=self.instance_label,
                                instance_subset=self.featureimp_instance_subset, 
                                algorithms=self.featimp_algorithms, 
                                use_turf=self.featureimp_use_turf, turf_pct=self.featureimp_turf_pct, 
                                random_state=self.random_state)
        f_imp.run(run_parallel=run_para)
        f_sel = FeatureSelectionRunner(output_path=self.output_path, experiment_name=self.experiment_name, 
                               algorithms=self.featimp_algorithms, class_label=self.class_label, 
                               instance_label=self.instance_label,
                               max_features_to_keep=self.featuresel_max_features_to_keep, 
                               filter_poor_features=self.featuresel_filter_poor_features, 
                               top_features=self.top_features, 
                               export_scores=self.featuresel_export_scores,
                               overwrite_cv=self.impute_overwrite_cv, 
                               random_state=self.random_state,
                               show_plots=self.show_plots)
        f_sel.run(run_parallel=run_para)
        model_exp = ModelExperimentRunner(
                                output_path=self.output_path, experiment_name=self.experiment_name, algorithms=self.ml_algorithms, 
                                exclude=self.exclude, class_label=self.class_label,
                                instance_label=self.instance_label, scoring_metric=self.scoring_metric, 
                                metric_direction=self.metric_direction,
                                training_subsample=self.training_subsample, 
                                use_uniform_fi=self.use_uniform_fi, n_trials=self.n_trials,
                                timeout=self.timeout, save_plots=self.save_plots, 
                                do_lcs_sweep=self.do_lcs_sweep, lcs_nu=self.lcs_nu, lcs_n=self.lcs_n, 
                                lcs_iterations=self.lcs_iterations,
                                lcs_timeout=self.lcs_timeout, resubmit=self.resubmit)
        model_exp.run(run_parallel=run_para)
        stats = StatsRunner(output_path=self.output_path, experiment_name=self.experiment_name, 
                    algorithms=self.ml_algorithms, exclude=self.exclude, 
                    class_label=self.class_label, instance_label=self.instance_label, 
                    scoring_metric=self.scoring_metric,
                    top_features=self.top_features, sig_cutoff=self.sig_cutoff, 
                    metric_weight=self.metric_weight, scale_data=self.scale_data,
                    plot_roc=self.plot_roc, plot_prc=self.plot_prc, 
                    plot_fi_box=self.plot_fi_box, 
                    plot_metric_boxplots=self.plot_metric_boxplots, 
                    show_plots=self.show_plots)
        stats.run(run_parallel=run_para)     
        if self.gen_report:
            rep = ReportRunner(self.output_path, self.experiment_name, 
                   algorithms=self.ml_algorithms, exclude=self.exclude)
            rep.run(run_parallel=run_para)
        if self.clean:
            dataset_paths = os.listdir(self.output_path + "/" + self.experiment_name)
            #only working with one dataset at a time as of now.
            for dataset_directory_path in dataset_paths:
                full_path = self.output_path + "/" + self.experiment_name + "/" + dataset_directory_path
                for i in self.dataset_names:
                    if full_path == self.output_path + "/" + self.experiment_name + "/" + i:
                        return full_path + '/model_evaluation/Summary_performance_mean.csv'
            clean = CleanRunner(self.output_path, self.experiment_name, del_time=self.del_time, del_old_cv=self.del_old_cv)
            # run_parallel is not used in clean
            clean.run()
            raise Exception('Performance Not Found')
        
        dataset_paths = os.listdir(self.output_path + "/" + self.experiment_name)
    
        #only working with one dataset at a time as of now.
        for dataset_directory_path in dataset_paths:
            full_path = self.output_path + "/" + self.experiment_name + "/" + dataset_directory_path
            for i in self.dataset_names:
                print(dataset_directory_path, i)
                if full_path == self.output_path + "/" + self.experiment_name + "/" + i:
                    return full_path + '/model_evaluation/Summary_performance_mean.csv'
        raise Exception('Performance Not Found')