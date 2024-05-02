import os
import sys
import time
import logging
from streamline.utils.parser import parser_function
from streamline.utils.checker import check_phase
from streamline.utils.runners import check_if_single_phase

from streamline.runners.dataprocess_runner import DataProcessRunner
from streamline.runners.imputation_runner import ImputationRunner
from streamline.runners.feature_runner import FeatureImportanceRunner
from streamline.runners.feature_runner import FeatureSelectionRunner
from streamline.runners.model_runner import ModelExperimentRunner
from streamline.runners.stats_runner import StatsRunner
from streamline.runners.report_runner import ReportRunner
from streamline.runners.clean_runner import CleanRunner

class STREAMLINERunner: 

    def __init__(self, dataset_names, gen_report=True, clean=True, data_path: str = "./data/DemoData", output_path: str="./DemoOutput",
                experiment_name: str='demo_experiment', exploration_list: list=["Describe", "Univariate Analysis", "Feature Correlation"],
                plot_list: list=["Describe", "Univariate Analysis", "Feature Correlation"],
                outcome_label:str="Class", instance_label:str='InstanceID', match_label=None, n_splits=3, partition_method="Stratified",
                ignore_features=None, categorical_feature_headers=None, quantitative_feature_headers=None, top_features=40,
                categorical_cutoff=10, sig_cutoff=0.05, featureeng_missingness=0.5, cleaning_missingness=0.5,
                correlation_removal_threshold=1.0,
                random_state=None, run_cluster=False, queue='defq', reserved_memory=4, show_plots=False,
                impute_scale_data=True, impute_data=True,
                impute_multi_impute=True, impute_overwrite_cv=True,
                do_mutual_info=True, do_multisurf=True,
                instance_subset=2000, algorithms=("MI", "MS"), use_turf=False, turf_pct=0.5,
                n_jobs=-1, max_features_to_keep=2000, filter_poor_features=True, export_scores=True,
                ml_algorithms=["NB", "LR", "DT", "EN"], exclude=['eLCS', 'XCS'], scoring_metric='balanced_accuracy', metric_direction='maximize',
                training_subsample=0, use_uniform_fi=True, n_trials=1,
                timeout=15, do_lcs_sweep=False, lcs_nu=1, lcs_n=2000, lcs_iterations=200000,
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
        self.outcome_label = outcome_label  # (str) i.e. class outcome column label
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
        logger.setLevel(logging.INFO)
        # if os.path.exists(self.output_path+'/'+self.experiment_name):
        #     shutil.rmtree(self.output_path+'/'+self.experiment_name)
        dpr = DataProcessRunner(data_path=self.data_path, output_path=self.output_path,
                experiment_name=self.experiment_name, outcome_label=self.outcome_label,
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
                        outcome_label=self.outcome_label, instance_label=self.instance_label, 
                        random_state=self.random_state)
        ir.run(run_parallel=run_para)
        self.featimp_algorithms = []
        if self.do_mutual_info:
            self.featimp_algorithms.append("MI")
        if self.do_multisurf:
            self.featimp_algorithms.append("MS")
        f_imp = FeatureImportanceRunner(output_path=self.output_path, experiment_name=self.experiment_name, 
                                outcome_label=self.outcome_label, 
                                instance_label=self.instance_label,
                                instance_subset=self.featureimp_instance_subset, 
                                algorithms=self.featimp_algorithms, 
                                use_turf=self.featureimp_use_turf, turf_pct=self.featureimp_turf_pct, 
                                random_state=self.random_state)
        f_imp.run(run_parallel=run_para)
        f_sel = FeatureSelectionRunner(output_path=self.output_path, experiment_name=self.experiment_name, 
                               algorithms=self.featimp_algorithms, outcome_label=self.outcome_label, 
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
                                exclude=self.exclude, outcome_label=self.outcome_label,
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
                    outcome_label=self.outcome_label, instance_label=self.instance_label, 
                    scoring_metric=self.scoring_metric,
                    top_features=self.top_features, sig_cutoff=self.sig_cutoff, 
                    metric_weight=self.metric_weight, scale_data=self.scale_data,
                    show_plots=self.show_plots)
        stats.run(run_parallel=run_para)     
        if self.gen_report:
            rep = ReportRunner(self.output_path, self.experiment_name)
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
                #print(dataset_directory_path, i)
                if full_path == self.output_path + "/" + self.experiment_name + "/" + i:
                    return full_path + '/model_evaluation/Summary_performance_mean.csv'
        raise Exception('Performance Not Found')

# warnings.filterwarnings("ignore")

# optuna.logging.set_verbosity(optuna.logging.WARNING)

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

# phase_list = ["", "Exploratory", "Data Process", "Feature Imp.",
#               "Feature Sel.", "Modeling", "Post-Analysis", "Dataset Compare",
#               "Testing Evaluation Report", "Replication",
#               "Replication Evaluation Report", "Cleaning"]

# phase_number = [' ', 1, 2, 3, 4, 5, 6, 7, 9, 8, 9, ' ']
# for idx in range(len(phase_number)):
#     if type(phase_number[idx]) == int:
#         phase_number[idx] = " (" + str(phase_number[idx]) + ") "


# def runner(obj, phase, run_parallel=True, params=None):
#     start = time.time()

#     phase_str = phase_list[phase]
#     phase_nu = phase_number[phase]
#     print()
#     if params['run_cluster'] and phase != 11:
#         print("Running " + phase_str + " Phase " + str(phase_nu)
#               + " with " + str(params['run_cluster']) + " Setup")
#     else:
#         print("Running " + phase_str + " Stage" + str(phase_nu)
#               + "with " + "Local" + " Setup")
#     how = "with " + str(params['run_cluster']) + " Manual Jobs"
#     if params['run_cluster'] == "SLURMOld" or params['run_cluster'] == "LSFOld":
#         obj.run(run_parallel=run_parallel)
#         try:
#             rep_data_path = params['rep_data_path']
#             dataset_for_rep = params['dataset_for_rep']
#         except KeyError:
#             rep_data_path = None
#             dataset_for_rep = None
#         if phase == 1:
#             time.sleep(5)
#         while len(check_phase(params['output_path'], params['experiment_name'],
#                               phase=phase, len_only=True,
#                               rep_data_path=rep_data_path,
#                               dataset_for_rep=dataset_for_rep,
#                               output=True)) != 0:
#             print()
#             if check_if_single_phase(params):
#                 print("Only one phase submitted using bash scripts, the runner can submit jobs and exit")
#                 print("Exiting")
#                 sys.exit()
#             print("Waiting for " + phase_str + " Manual Jobs to Finish")
#             time.sleep(5)
#         print()
#     else:
#         obj.run(run_parallel=run_parallel)
#         if not run_parallel or run_parallel == "False":
#             how = "serially"
#         elif run_parallel in ["multiprocessing", "True", True] \
#                 and str(params['run_cluster']) == "False":
#             how = "parallely"
#         if str(params['run_cluster']) != "False":
#             how = "with " + str(params['run_cluster']) + " dask cluster"

#     print("Ran " + phase_str + " Phase " + how + " in " + str(time.time() - start))
#     if str(params['run_cluster']) == "LSF":
#         time.sleep(2)
#     del obj


# def len_datasets(output_path, experiment_name):
#     datasets = os.listdir(output_path + '/' + experiment_name)
#     remove_list = ['.DS_Store', 'metadata.pickle', 'metadata.csv', 'algInfo.pickle',
#                    'jobsCompleted', 'logs', 'jobs', 'DatasetComparisons',
#                    'UsefulNotebooks', 'dask_logs',
#                    experiment_name + '_STREAMLINE_Report.pdf']
#     for text in remove_list:
#         if text in datasets:
#             datasets.remove(text)
#     return len(datasets)


# def run(params):
#     start_g = time.time()

#     if params['do_eda']:
#         from streamline.runners.dataprocess_runner import DataProcessRunner
#         eda = DataProcessRunner(params['dataset_path'], params['output_path'], params['experiment_name'],
#                                 exclude_eda_output=params['exclude_eda_output'],
#                                 outcome_label=params['outcome_label'], instance_label=params['instance_label'],
#                                 match_label=params['match_label'],
#                                 n_splits=params['cv_partitions'],
#                                 partition_method=params['partition_method'],
#                                 ignore_features=params['ignore_features_path'],
#                                 categorical_features=params['categorical_feature_path'],
#                                 quantitative_features=params['quantitative_feature_path'],
#                                 top_features=params['top_uni_features'],
#                                 categorical_cutoff=params['categorical_cutoff'],
#                                 sig_cutoff=params['sig_cutoff'],
#                                 featureeng_missingness=params['featureeng_missingness'],
#                                 cleaning_missingness=params['cleaning_missingness'],
#                                 correlation_removal_threshold=params['correlation_removal_threshold'],
#                                 random_state=params['random_state'],
#                                 run_cluster=params['run_cluster'],
#                                 queue=params['queue'],
#                                 reserved_memory=params['reserved_memory'])

#         runner(eda, 1, run_parallel=params['run_parallel'], params=params)
#         params['outcome_type'] = eda.outcome_type

#     if params['do_dataprep']:
#         from streamline.runners.imputation_runner import ImputationRunner
#         dpr = ImputationRunner(params['output_path'], params['experiment_name'], scale_data=params['scale_data'],
#                                impute_data=params['impute_data'],
#                                multi_impute=params['multi_impute'], overwrite_cv=params['overwrite_cv'],
#                                outcome_label=params['outcome_label'],
#                                instance_label=params['instance_label'], random_state=params['random_state'],
#                                run_cluster=params['run_cluster'],
#                                queue=params['queue'],
#                                reserved_memory=params['reserved_memory'])
#         runner(dpr, 2, run_parallel=params['run_parallel'], params=params)

#     if params['do_feat_imp']:
#         from streamline.runners.feature_runner import FeatureImportanceRunner
#         f_imp = FeatureImportanceRunner(params['output_path'], params['experiment_name'],
#                                         outcome_label=params['outcome_label'],
#                                         instance_label=params['instance_label'],
#                                         instance_subset=params['instance_subset'], algorithms=params['feat_algorithms'],
#                                         use_turf=params['use_turf'],
#                                         turf_pct=params['turf_pct'],
#                                         random_state=params['random_state'], n_jobs=params['n_jobs'],
#                                         run_cluster=params['run_cluster'],
#                                         queue=params['queue'],
#                                         reserved_memory=params['reserved_memory'])
#         runner(f_imp, 3, run_parallel=params['run_parallel'], params=params)

#     if params['do_feat_sel']:
#         from streamline.runners.feature_runner import FeatureSelectionRunner
#         f_sel = FeatureSelectionRunner(params['output_path'], params['experiment_name'],
#                                        algorithms=params['feat_algorithms'],
#                                        outcome_label=params['outcome_label'],
#                                        instance_label=params['instance_label'],
#                                        max_features_to_keep=params['max_features_to_keep'],
#                                        filter_poor_features=params['filter_poor_features'],
#                                        top_features=params['top_fi_features'], export_scores=params['export_scores'],
#                                        overwrite_cv=params['overwrite_cv_feat'], random_state=params['random_state'],
#                                        n_jobs=params['n_jobs'],
#                                        run_cluster=params['run_cluster'],
#                                        queue=params['queue'],
#                                        reserved_memory=params['reserved_memory'])
#         runner(f_sel, 4, run_parallel=params['run_parallel'], params=params)

#     if params['do_model']:
#         from streamline.runners.model_runner import ModelExperimentRunner
#         model = ModelExperimentRunner(params['output_path'], params['experiment_name'],
#                                       algorithms=params['algorithms'], exclude=params['exclude'],
#                                       outcome_label=params['outcome_label'], outcome_type=params['outcome_type'],
#                                       instance_label=params['instance_label'], scoring_metric=params['primary_metric'],
#                                       metric_direction=params['metric_direction'],
#                                       training_subsample=params['training_subsample'],
#                                       use_uniform_fi=params['use_uniform_fi'],
#                                       n_trials=params['n_trials'],
#                                       timeout=params['timeout'], save_plots=False, do_lcs_sweep=params['do_lcs_sweep'],
#                                       lcs_nu=params['lcs_nu'],
#                                       lcs_n=params['lcs_n'],
#                                       lcs_iterations=params['lcs_iterations'],
#                                       lcs_timeout=params['lcs_timeout'], resubmit=params['model_resubmit'],
#                                       random_state=params['random_state'], n_jobs=params['n_jobs'],
#                                       run_cluster=params['run_cluster'],
#                                       queue=params['queue'],
#                                       reserved_memory=params['reserved_memory'])

#         runner(model, 5, run_parallel=params['run_parallel'], params=params)

#     if params['do_stats']:
#         from streamline.runners.stats_runner import StatsRunner
#         stats = StatsRunner(params['output_path'], params['experiment_name'],
#                             outcome_label=params['outcome_label'], outcome_type=params['outcome_type'],
#                             instance_label=params['instance_label'],
#                             scoring_metric=params['primary_metric'],
#                             top_features=params['top_model_fi_features'], sig_cutoff=params['sig_cutoff'],
#                             metric_weight=params['metric_weight'],
#                             scale_data=params['scale_data'],
#                             exclude_plots=params['exclude_plots'], show_plots=False,
#                             run_cluster=params['run_cluster'],
#                             queue=params['queue'],
#                             reserved_memory=params['reserved_memory'])
#         runner(stats, 6, run_parallel=params['run_parallel'], params=params)

#     if params['do_compare_dataset']:
#         if len_datasets(params['output_path'], params['experiment_name']) > 1:
#             from streamline.runners.compare_runner import CompareRunner
#             compare = CompareRunner(params['output_path'], params['experiment_name'], experiment_path=None,
#                                     outcome_label=params['outcome_label'], instance_label=params['instance_label'],
#                                     sig_cutoff=params['sig_cutoff'],
#                                     show_plots=False,
#                                     run_cluster=params['run_cluster'],
#                                     queue=params['queue'],
#                                     reserved_memory=params['reserved_memory'])
#             runner(compare, 7, run_parallel=params['run_parallel'], params=params)

#     if params['do_report']:
#         from streamline.runners.report_runner import ReportRunner
#         report = ReportRunner(output_path=params['output_path'], experiment_name=params['experiment_name'],
#                               experiment_path=None,
#                               run_cluster=params['run_cluster'],
#                               queue=params['queue'],
#                               reserved_memory=params['reserved_memory'])
#         runner(report, 8, run_parallel=params['run_parallel'], params=params)

#     if params['do_replicate']:
#         from streamline.runners.replicate_runner import ReplicationRunner
#         replicate = ReplicationRunner(params['rep_data_path'], params['dataset_for_rep'], params['output_path'],
#                                       params['experiment_name'],
#                                       outcome_label=params['outcome_label'], instance_label=params['instance_label'],
#                                       match_label=params['match_label'],
#                                       exclude=params['exclude'],
#                                       exclude_plots=params['exclude_rep_plots'],
#                                       run_cluster=params['run_cluster'],
#                                       queue=params['queue'],
#                                       reserved_memory=params['reserved_memory'])
#         runner(replicate, 9, run_parallel=params['run_parallel'], params=params)

#     if params['do_rep_report']:
#         from streamline.runners.report_runner import ReportRunner
#         report = ReportRunner(output_path=params['output_path'], experiment_name=params['experiment_name'],
#                               experiment_path=None, training=False,
#                               rep_data_path=params['rep_data_path'],
#                               dataset_for_rep=params['dataset_for_rep'],
#                               run_cluster=params['run_cluster'],
#                               queue=params['queue'],
#                               reserved_memory=params['reserved_memory'])
#         runner(report, 10, run_parallel=params['run_parallel'], params=params)

#     if params['do_cleanup']:
#         from streamline.runners.clean_runner import CleanRunner
#         clean = CleanRunner(params['output_path'], params['experiment_name'],
#                             del_time=params['del_time'], del_old_cv=params['del_old_cv'])
#         runner(clean, 11, run_parallel=params['run_parallel'], params=params)

#     print("DONE!!!")
#     print("Ran in " + str(time.time() - start_g))


# if __name__ == '__main__':

#     # NOTE: All keys must be small
#     config_dict = parser_function(sys.argv)

#     if not os.path.exists(config_dict['output_path']):
#         os.mkdir(str(config_dict['output_path']))

#     if config_dict['verbose']:
#         stdout_handler = logging.StreamHandler(sys.stdout)
#         stdout_handler.setLevel(logging.INFO)
#         stdout_handler.setFormatter(formatter)
#         logger.addHandler(stdout_handler)
#     else:
#         file_handler = logging.FileHandler(str(config_dict['output_path']) + '/logs.log')
#         file_handler.setLevel(logging.INFO)
#         file_handler.setFormatter(formatter)
#         logger.addHandler(file_handler)

#     sys.exit(run(config_dict))
