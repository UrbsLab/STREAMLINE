import os
import sys
import time
import optuna
import logging
import warnings
import configparser
from streamline.utils.checker import check_phase
from streamline.utils.runners import check_if_single_phase
from streamline.utils.parser_helpers import save_config
from streamline.utils.config_loader import load_default_config
from streamline.utils.parser import parser_function_definition
from streamline.utils.parser import parser_function_all, single_parse, process_params
from streamline.utils.parser_helpers import parse_checker, parse_general
from streamline.utils.checker import check_phase


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class STREAMLINERunner:
    phase_names = ["", "Exploratory", "Data Process", "Feature Imp.",
              "Feature Sel.", "Modeling", "Post-Analysis", "Dataset Compare",
              "Testing Evaluation Report", "Replication",
              "Replication Evaluation Report", "Cleaning"]
    phase_number_idx = [' ', ' (1) ', ' (2) ', ' (3) ', ' (4) ', ' (5) ',
                        ' (6) ', ' (7) ', ' (9) ', ' (8) ', ' (9) ', ' ']
    essential_params = ['output_path', 'experiment_name']

    def __init__(self, argv=None, run=False):
        self.params = dict()
        self.runner = run
        self.checker = None
        if argv:
            self.process_argv(argv)
        if run:
            self.set_logger()
            self.run()

    def process_argv(self, argv):
        self.load_default_params()
        mode_params = parser_function_definition(argv)
        # self.params.update(mode_params)
        # print(mode_params)

        if mode_params['config'] != "":
            self.params['config'] = mode_params['config']
            self.load_config_params()

        def check_cli(mode_params):
            flag = False
            if mode_params['do_till_report']:
                flag = True
            if mode_params['checker']:
                flag = True
            for key in mode_params:
                if mode_params[key] and key not in ['config', 'do_till_report', 'verbose', 'checker']:
                    flag = True
            return flag
        
        if check_cli(mode_params):
            self.load_cli_params(mode_params, argv)

        if not self.checker:
            self.params = process_params(self.params)

        for param in self.essential_params:
            if param not in self.params:
                raise Exception("Essential params not in config: " + str(param))
        if not os.path.exists(self.params['output_path']):
            os.mkdir(str(self.params['output_path']))
        if not os.path.exists(self.params['output_path'] + '/' + self.params['experiment_name']):
            os.mkdir(str(self.params['output_path']))
        self.save_params()
    
    def load_default_params(self):
        self.params.update(load_default_config())

    def load_config_params(self):
        config_file = self.params['config']
        config = configparser.ConfigParser()
        config.read(config_file)
        for s in config.sections():
            self.params.update({k: eval(v) for k, v in config.items(s)})
        save_config(self.params['output_path'],
                    self.params['experiment_name'],
                    self.params)

    def load_cli_params(self, mode_params, argv):
        if mode_params['do_till_report']:
            print("Running till Report Generation Stage")
            config = parser_function_all(argv)
            self.params.update(config)
            save_config(self.params['output_path'],
                        self.params['experiment_name'],
                        self.params)
        if mode_params['checker']:
            print("Checking Progress")
            config = parse_general(argv, self.params)
            config = parse_checker(argv, config)
            self.params.update(config)
            self.checker = True
            return

        for key in mode_params:
            if mode_params[key] and key not in ['config', 'do_till_report', 'verbose', 'checker']:
                config = single_parse(mode_params, argv, self.params)
                self.params.update(config)
                self.params.update(mode_params)
                save_config(self.params['output_path'],
                            self.params['experiment_name'],
                            self.params)

    def save_params(self):
        save_config(self.params['output_path'], self.params['experiment_name'], self.params)

    def set_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

        if self.params['verbose']:
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(logging.INFO)
            stdout_handler.setFormatter(formatter)
            logger.addHandler(stdout_handler)
        else:
            file_handler = logging.FileHandler(str(self.params['output_path']) + '/' +
                                               str(self.params['experiment_name'])
                                               + '/overview_log.log')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger

    def get_len_datasets(self):
        output_path, experiment_name  = self.params['output_path'], self.params['experiment_name']
        datasets = os.listdir(output_path + '/' + experiment_name)
        remove_list = ['.DS_Store', 'metadata.pickle', 'metadata.csv', 'algInfo.pickle', 'runparams.pickle', 'overview_log.log', 
                    'jobsCompleted', 'logs', 'jobs', 'DatasetComparisons',
                    'UsefulNotebooks', 'dask_logs',
                    experiment_name + '_STREAMLINE_Report.pdf']
        for text in remove_list:
            if text in datasets:
                datasets.remove(text)
        return len(datasets)
        
    def run_phase(self, obj, phase):
        start = time.time()

        run_parallel = self.params['run_parallel']

        phase_str = self.phase_names[phase]
        phase_nu = self.phase_number_idx[phase]
        print()
        if self.params['run_cluster'] and phase != 11:
            print("Running " + phase_str + " Phase " + str(phase_nu)
                + " with " + str(self.params['run_cluster']) + " Setup")
        else:
            print("Running " + phase_str + " Stage" + str(phase_nu)
                + "with " + "Local" + " Setup")
        how = "with " + str(self.params['run_cluster']) + " Manual Jobs"
        if self.params['run_cluster'] == "SLURMOld" or self.params['run_cluster'] == "LSFOld":
            obj.run(run_parallel=run_parallel)
            try:
                rep_data_path = self.params['rep_data_path']
                dataset_for_rep = self.params['dataset_for_rep']
            except KeyError:
                rep_data_path = None
                dataset_for_rep = None
            if phase == 1:
                time.sleep(5)
            while len(check_phase(self.params['output_path'], self.params['experiment_name'],
                                phase=phase, len_only=True,
                                rep_data_path=rep_data_path,
                                dataset_for_rep=dataset_for_rep,
                                output=True)) != 0:
                print()
                if check_if_single_phase(self.params):
                    print("Only one phase submitted using bash scripts, the self.run_phase can submit jobs and exit")
                    print("Exiting")
                    sys.exit()
                print("Waiting for " + phase_str + " Manual Jobs to Finish")
                time.sleep(5)
            print()
        else:
            obj.run(run_parallel=run_parallel)
            if not run_parallel or run_parallel == "False":
                how = "serially"
            elif run_parallel in ["multiprocessing", "True", True] \
                    and str(self.params['run_cluster']) == "False":
                how = "parallely"
            if str(self.params['run_cluster']) != "False":
                how = "with " + str(self.params['run_cluster']) + " dask cluster"

        print("Ran " + phase_str + " Phase " + how + " in " + str(time.time() - start))
        if str(self.params['run_cluster']) == "LSF":
            time.sleep(2)
        del obj

    def check_progress(self):
        check_phase(self.params['output_path'], self.params['experiment_name'], 
                    self.params['phase'], self.params['len_only'],
                    self.params['rep_data_path'], self.params['dataset_for_rep'])

    def run(self):
        start_g = time.time()

        if self.params['do_eda']:
            from streamline.runners.dataprocess_runner import DataProcessRunner
            eda = DataProcessRunner(self.params['dataset_path'], self.params['output_path'], self.params['experiment_name'],
                                    exclude_eda_output=self.params['exclude_eda_output'],
                                    outcome_label=self.params['outcome_label'], instance_label=self.params['instance_label'],
                                    match_label=self.params['match_label'],
                                    n_splits=self.params['cv_partitions'],
                                    partition_method=self.params['partition_method'],
                                    ignore_features=self.params['ignore_features_path'],
                                    categorical_features=self.params['categorical_feature_path'],
                                    quantitative_features=self.params['quantitative_feature_path'],
                                    top_features=self.params['top_uni_features'],
                                    categorical_cutoff=self.params['categorical_cutoff'],
                                    sig_cutoff=self.params['sig_cutoff'],
                                    featureeng_missingness=self.params['featureeng_missingness'],
                                    cleaning_missingness=self.params['cleaning_missingness'],
                                    correlation_removal_threshold=self.params['correlation_removal_threshold'],
                                    random_state=self.params['random_state'],
                                    run_cluster=self.params['run_cluster'],
                                    queue=self.params['queue'],
                                    reserved_memory=self.params['reserved_memory'],
                                    walltime=self.params['walltime'])

            self.run_phase(eda, 1)
            self.params['outcome_type'] = eda.outcome_type

        if self.params['do_dataprep']:
            from streamline.runners.imputation_runner import ImputationRunner
            dpr = ImputationRunner(self.params['output_path'], self.params['experiment_name'], scale_data=self.params['scale_data'],
                                impute_data=self.params['impute_data'],
                                multi_impute=self.params['multi_impute'], overwrite_cv=self.params['overwrite_cv'],
                                outcome_label=self.params['outcome_label'],
                                instance_label=self.params['instance_label'], random_state=self.params['random_state'],
                                run_cluster=self.params['run_cluster'],
                                queue=self.params['queue'],
                                reserved_memory=self.params['reserved_memory'],
                                walltime=self.params['walltime'])
            self.run_phase(dpr, 2)

        if self.params['do_feat_imp']:
            from streamline.runners.feature_runner import FeatureImportanceRunner
            f_imp = FeatureImportanceRunner(self.params['output_path'], self.params['experiment_name'],
                                            outcome_label=self.params['outcome_label'],
                                            instance_label=self.params['instance_label'],
                                            instance_subset=self.params['instance_subset'], algorithms=self.params['feat_algorithms'],
                                            use_turf=self.params['use_turf'],
                                            turf_pct=self.params['turf_pct'],
                                            random_state=self.params['random_state'], n_jobs=self.params['n_jobs'],
                                            run_cluster=self.params['run_cluster'],
                                            queue=self.params['queue'],
                                            reserved_memory=self.params['reserved_memory'],
                                            walltime=self.params['walltime'])
            self.run_phase(f_imp, 3)

        if self.params['do_feat_sel']:
            from streamline.runners.feature_runner import FeatureSelectionRunner
            f_sel = FeatureSelectionRunner(self.params['output_path'], self.params['experiment_name'],
                                        algorithms=self.params['feat_algorithms'],
                                        outcome_label=self.params['outcome_label'],
                                        instance_label=self.params['instance_label'],
                                        max_features_to_keep=self.params['max_features_to_keep'],
                                        filter_poor_features=self.params['filter_poor_features'],
                                        top_features=self.params['top_fi_features'], export_scores=self.params['export_scores'],
                                        overwrite_cv=self.params['overwrite_cv_feat'], random_state=self.params['random_state'],
                                        n_jobs=self.params['n_jobs'],
                                        run_cluster=self.params['run_cluster'],
                                        queue=self.params['queue'],
                                        reserved_memory=self.params['reserved_memory'],
                                        walltime=self.params['walltime'])
            self.run_phase(f_sel, 4)

        if self.params['do_model']:
            from streamline.runners.model_runner import ModelExperimentRunner
            model = ModelExperimentRunner(self.params['output_path'], self.params['experiment_name'],
                                        algorithms=self.params['algorithms'], exclude=self.params['exclude'],
                                        outcome_label=self.params['outcome_label'], outcome_type=self.params['outcome_type'],
                                        instance_label=self.params['instance_label'], scoring_metric=self.params['primary_metric'],
                                        metric_direction=self.params['metric_direction'],
                                        training_subsample=self.params['training_subsample'],
                                        use_uniform_fi=self.params['use_uniform_fi'],
                                        n_trials=self.params['n_trials'],
                                        timeout=self.params['timeout'], save_plots=False, do_lcs_sweep=self.params['do_lcs_sweep'],
                                        lcs_nu=self.params['lcs_nu'],
                                        lcs_n=self.params['lcs_n'],
                                        lcs_iterations=self.params['lcs_iterations'],
                                        lcs_timeout=self.params['lcs_timeout'], resubmit=self.params['model_resubmit'],
                                        random_state=self.params['random_state'], n_jobs=self.params['n_jobs'],
                                        run_cluster=self.params['run_cluster'],
                                        queue=self.params['queue'],
                                        reserved_memory=self.params['reserved_memory'],
                                        walltime=self.params['walltime'])

            self.run_phase(model, 5)

        if self.params['do_stats']:
            from streamline.runners.stats_runner import StatsRunner
            stats = StatsRunner(self.params['output_path'], self.params['experiment_name'],
                                outcome_label=self.params['outcome_label'], outcome_type=self.params['outcome_type'],
                                instance_label=self.params['instance_label'],
                                scoring_metric=self.params['primary_metric'],
                                top_features=self.params['top_model_fi_features'], sig_cutoff=self.params['sig_cutoff'],
                                metric_weight=self.params['metric_weight'],
                                scale_data=self.params['scale_data'],
                                exclude_plots=self.params['exclude_plots'], show_plots=False,
                                run_cluster=self.params['run_cluster'],
                                queue=self.params['queue'],
                                reserved_memory=self.params['reserved_memory'],
                                walltime=self.params['walltime'])
            self.run_phase(stats, 6)

        if self.params['do_compare_dataset']:
            if self.get_len_datasets() > 1:
                from streamline.runners.compare_runner import CompareRunner
                compare = CompareRunner(self.params['output_path'], self.params['experiment_name'], experiment_path=None,
                                        outcome_label=self.params['outcome_label'], instance_label=self.params['instance_label'],
                                        sig_cutoff=self.params['sig_cutoff'],
                                        show_plots=False,
                                        run_cluster=self.params['run_cluster'],
                                        queue=self.params['queue'],
                                        reserved_memory=self.params['reserved_memory'], 
                                        walltime=self.params['walltime'])
                self.run_phase(compare, 7)

        if self.params['do_report']:
            from streamline.runners.report_runner import ReportRunner
            report = ReportRunner(output_path=self.params['output_path'], experiment_name=self.params['experiment_name'],
                                experiment_path=None,
                                run_cluster=self.params['run_cluster'],
                                queue=self.params['queue'],
                                reserved_memory=self.params['reserved_memory'],
                                walltime=self.params['walltime'])
            self.run_phase(report, 8)

        if self.params['do_replicate']:
            from streamline.runners.replicate_runner import ReplicationRunner
            replicate = ReplicationRunner(self.params['rep_data_path'], self.params['dataset_for_rep'], self.params['output_path'],
                                        self.params['experiment_name'],
                                        outcome_label=self.params['outcome_label'], instance_label=self.params['instance_label'],
                                        match_label=self.params['match_label'],
                                        exclude_plots=self.params['exclude_rep_plots'],
                                        run_cluster=self.params['run_cluster'],
                                        queue=self.params['queue'],
                                        reserved_memory=self.params['reserved_memory'], 
                                        walltime=self.params['walltime'])
            self.run_phase(replicate, 9)

        if self.params['do_rep_report']:
            from streamline.runners.report_runner import ReportRunner
            report = ReportRunner(output_path=self.params['output_path'], experiment_name=self.params['experiment_name'],
                                experiment_path=None, training=False,
                                rep_data_path=self.params['rep_data_path'],
                                dataset_for_rep=self.params['dataset_for_rep'],
                                run_cluster=self.params['run_cluster'],
                                queue=self.params['queue'],
                                reserved_memory=self.params['reserved_memory'],
                                walltime=self.params['walltime'])
            self.run_phase(report, 10)

        if self.params['do_cleanup']:
            from streamline.runners.clean_runner import CleanRunner
            clean = CleanRunner(self.params['output_path'], self.params['experiment_name'],
                                del_time=self.params['del_time'], del_old_cv=self.params['del_old_cv'])
            self.run_phase(clean, 11)

        print("DONE!!!")
        print("Ran in " + str(time.time() - start_g))


if __name__ == '__main__':
    sys.exit(STREAMLINERunner(sys.argv))
