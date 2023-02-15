import os
import sys
import time
import optuna
import logging
import configparser
from streamline.runners.eda_runner import EDARunner
from streamline.runners.dataprocess_runner import DataProcessRunner
from streamline.runners.feature_runner import FeatureImportanceRunner
from streamline.runners.feature_runner import FeatureSelectionRunner
from streamline.runners.model_runner import ModelExperimentRunner
from streamline.runners.stats_runner import StatsRunner
from streamline.runners.compare_runner import CompareRunner
from streamline.runners.report_runner import ReportRunner
from streamline.runners.replicate_runner import ReplicationRunner
from streamline.runners.clean_runner import CleanRunner
from streamline.utils.parser import parser_function
from streamline.utils.checker import check_phase
from streamline.utils.runners import num_cores
import warnings
warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')


def runner(obj, phase_str, run_parallel=True, params=None):
    start = time.time()
    obj.run(run_parallel=run_parallel)
    if not run_parallel or run_parallel == "False":
        how = "serially"
    elif run_parallel in ["multiprocessing", "True"]:
        how = "parallely"
    else:
        how = "with " + run_parallel + " dask cluster"
    if params and (params['run_cluster'] == "SLURMOld" and phase_str == "Modeling"):
        while check_phase(params['output_path'], params['experiment_name'], 
                          phase=5, output=False) != 0:
            time.sleep(5)
            how = "with SLURM Manual Jobs"

    print("Ran " + phase_str + " Phase " + how + " in " + str(time.time() - start))
    del obj


def run(params):
    start_g = time.time()

    if 'feat_algorithms' not in params:
        feat_algorithms = list()
        if params['do_mutual_info']:
            feat_algorithms.append("MI")
        if params['do_multisurf']:
            feat_algorithms.append("MS")
        params['feat_algorithms'] = feat_algorithms

    if params['do_all']:
        params['algorithms'] = None

    if params['ignore_features_path'] == '':
        params['ignore_features_path'] = None
    if params['categorical_feature_path'] == '':
        params['categorical_feature_path'] = None
    if params['match_label'] == '':
        params['match_label'] = None
    if params['instance_label'] == '':
        params['instance_label'] = None

    if params['do_eda']:
        eda = EDARunner(params['dataset_path'], params['output_path'], params['experiment_name'], exploration_list=None,
                        plot_list=None,
                        class_label=params['class_label'], instance_label=params['instance_label'],
                        match_label=params['match_label'],
                        n_splits=params['cv_partitions'],
                        partition_method=params['partition_method'],
                        ignore_features=params['ignore_features_path'],
                        categorical_features=params['categorical_feature_path'],
                        top_features=params['top_features'],
                        categorical_cutoff=params['categorical_cutoff'], sig_cutoff=params['sig_cutoff'],
                        random_state=params['random_state'])

        runner(eda, "Exploratory", run_parallel=params['run_parallel'], params=params)

    if params['do_dataprep']:
        dpr = DataProcessRunner(params['output_path'], params['experiment_name'], scale_data=params['scale_data'],
                                impute_data=params['impute_data'],
                                multi_impute=params['multi_impute'], overwrite_cv=params['overwrite_cv'],
                                class_label=params['class_label'],
                                instance_label=params['instance_label'], random_state=params['random_state'])
        runner(dpr, "Data Process", run_parallel=params['run_parallel'], params=params)

    if params['do_feat_imp']:
        f_imp = FeatureImportanceRunner(params['output_path'], params['experiment_name'],
                                        class_label=params['class_label'],
                                        instance_label=params['instance_label'],
                                        instance_subset=params['instance_subset'], algorithms=params['feat_algorithms'],
                                        use_turf=params['use_turf'],
                                        turf_pct=params['turf_pct'],
                                        random_state=params['random_state'], n_jobs=params['n_jobs'])
        runner(f_imp, "Feature Imp.", run_parallel=params['run_parallel'], params=params)

    if params['do_feat_sel']:
        f_sel = FeatureSelectionRunner(params['output_path'], params['experiment_name'],
                                       algorithms=params['feat_algorithms'],
                                       class_label=params['class_label'],
                                       instance_label=params['instance_label'],
                                       max_features_to_keep=params['max_features_to_keep'],
                                       filter_poor_features=params['filter_poor_features'],
                                       top_features=params['top_features'], export_scores=params['export_scores'],
                                       overwrite_cv=params['overwrite_cv_feat'], random_state=params['random_state'],
                                       n_jobs=params['n_jobs'])
        runner(f_sel, "Feature Sel.", run_parallel=params['run_parallel'], params=params)

    if params['do_model']:
        model = ModelExperimentRunner(params['output_path'], params['experiment_name'],
                                      algorithms=params['algorithms'], exclude=params['exclude'],
                                      class_label=params['class_label'],
                                      instance_label=params['instance_label'], scoring_metric=params['primary_metric'],
                                      metric_direction=params['metric_direction'],
                                      training_subsample=params['training_subsample'],
                                      use_uniform_fi=params['use_uniform_fi'],
                                      n_trials=params['n_trials'],
                                      timeout=params['timeout'], save_plots=False, do_lcs_sweep=params['do_lcs_sweep'],
                                      lcs_nu=params['lcs_nu'],
                                      lcs_n=params['lcs_n'],
                                      lcs_iterations=params['lcs_iterations'],
                                      lcs_timeout=params['lcs_timeout'], resubmit=False,
                                      random_state=params['random_state'], n_jobs=params['n_jobs'],
                                      run_cluster=params['run_cluster'],
                                      queue=params['queue'],
                                      reserved_memory=params['reserved_memory'])

        runner(model, "Modeling", run_parallel=params['run_parallel'], params=params)

    if params['do_stats']:
        stats = StatsRunner(params['output_path'], params['experiment_name'], algorithms=params['algorithms'],
                            exclude=params['exclude'],
                            class_label=params['class_label'], instance_label=params['instance_label'],
                            scoring_metric=params['primary_metric'],
                            top_features=params['top_model_features'], sig_cutoff=params['sig_cutoff'],
                            metric_weight=params['metric_weight'],
                            scale_data=params['scale_data'],
                            plot_roc=params['plot_roc'], plot_prc=params['plot_prc'], plot_fi_box=params['plot_fi_box'],
                            plot_metric_boxplots=params['plot_metric_boxplots'], show_plots=False)
        runner(stats, "Stats", run_parallel=params['run_parallel'], params=params)

    if params['do_compare_dataset']:
        compare = CompareRunner(params['output_path'], params['experiment_name'], experiment_path=None,
                                algorithms=params['algorithms'],
                                exclude=params['exclude'],
                                class_label=params['class_label'], instance_label=params['instance_label'],
                                sig_cutoff=params['sig_cutoff'],
                                show_plots=False)
        runner(compare, "Dataset Compare", run_parallel=params['run_parallel'], params=params)

    if params['do_report']:
        report = ReportRunner(output_path=params['output_path'], experiment_name=params['experiment_name'],
                              experiment_path=None,
                              algorithms=params['algorithms'], exclude=params['exclude'])
        runner(report, "Reporting", run_parallel=params['run_parallel'], params=params)

    if params['do_replicate']:
        replicate = ReplicationRunner(params['rep_data_path'], params['dataset_for_rep'], params['output_path'],
                                      params['experiment_name'],
                                      class_label=params['class_label'], instance_label=params['instance_label'],
                                      match_label=params['match_label'],
                                      algorithms=params['algorithms'], load_algo=True,
                                      exclude=params['exclude'],
                                      export_feature_correlations=params['rep_export_feature_correlations'],
                                      plot_roc=params['rep_plot_roc'], plot_prc=params['rep_plot_prc'],
                                      plot_metric_boxplots=params['rep_plot_metric_boxplots'])
        runner(replicate, "Replication", run_parallel=params['run_parallel'], params=params)

    if params['do_rep_report']:
        report = ReportRunner(output_path=params['output_path'], experiment_name=params['experiment_name'],
                              experiment_path=None,
                              algorithms=params['algorithms'], exclude=params['exclude'], training=False,
                              rep_data_path=params['rep_data_path'],
                              dataset_for_rep=params['dataset_for_rep'])
        runner(report, "Replicate Report", run_parallel=params['run_parallel'], params=params)

    if params['do_cleanup']:
        clean = CleanRunner(params['output_path'], params['experiment_name'],
                            del_time=params['del_time'], del_old_cv=params['del_old_cv'])
        runner(clean, "Cleaning", run_parallel=params['run_parallel'], params=params)

    print("DONE!!!")
    print("Ran in " + str(time.time() - start_g))


if __name__ == '__main__':

    # NOTE: All keys must be small

    if sys.argv[1] == '-c' or sys.argv[1] == '--config':
        config_file = sys.argv[2]
        config = configparser.ConfigParser()
        config.read(config_file)
        config_dict = dict()
        for s in config.sections():
            config_dict.update({k: eval(v) for k, v in config.items(s)})
    else:
        config_dict = vars(parser_function(sys.argv))

    print("Running with " + str(num_cores) + "CPUs")

    if not os.path.exists(config_dict['output_path']):
        os.mkdir(config_dict['output_path'])

    if config_dict['verbose']:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
    else:
        file_handler = logging.FileHandler(config_dict['output_path'] + '/logs.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    sys.exit(run(config_dict))
