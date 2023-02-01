import logging
import os
import glob
import multiprocessing
from streamline.modeling.modeljob import ModelJob
from streamline.modeling.utils import model_str_to_obj
from streamline.modeling.utils import SUPPORTED_MODELS
from streamline.modeling.utils import is_supported_model


class ModelExperimentRunner:
    """
    Runner Class for running all the model jobs for
    cross-validation splits.
    """

    def __init__(self, output_path, experiment_name, algorithms=None, exclude=None, class_label="Class",
                 instance_label=None, scoring_metric='balanced_accuracy', metric_direction='maximize',
                 training_subsample=0, use_uniform_fi=True, n_trials=200,
                 timeout=900, save_plots=False, do_lcs_sweep=False, lcs_nu=1, lcs_n=2000, lcs_iterations=200000,
                 lcs_timeout=1200, random_state=None, n_jobs=None):

        """
        Args:
            output_path: path to output directory
            experiment_name: name of experiment (no spaces)
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
        # TODO: What does training subsample do
        self.cv_count = None
        self.dataset = None
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.class_label = class_label
        self.instance_label = instance_label

        if algorithms is None:
            self.algorithms = SUPPORTED_MODELS
            if exclude is not None:
                for algorithm in exclude:
                    try:
                        self.algorithms.remove(algorithm)
                    except Exception:
                        logging.error("Unknown algorithm in exclude: " + str(algorithm))
        else:
            for algorithm in algorithms:
                assert is_supported_model(algorithm)
            self.algorithms = algorithms

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

        self.random_state = random_state
        self.n_jobs = n_jobs

        # Argument checks
        if not os.path.exists(self.output_path):
            raise Exception("Output path must exist (from phase 1) before phase 4 can begin")
        if not os.path.exists(self.output_path + '/' + self.experiment_name):
            raise Exception("Experiment must exist (from phase 1) before phase 4 can begin")

    def run(self, run_parallel):

        # Iterate through datasets, ignoring common folders
        dataset_paths = os.listdir(self.output_path + "/" + self.experiment_name)
        remove_list = ['jobsCompleted', 'logs', 'jobs', 'DatasetComparisons', 'UsefulNotebooks']

        for text in remove_list:
            if text in dataset_paths:
                dataset_paths.remove(text)

        job_list = list()

        for dataset_directory_path in dataset_paths:
            full_path = self.output_path + "/" + self.experiment_name + "/" + dataset_directory_path
            if not os.path.exists(full_path + '/models'):
                os.mkdir(full_path + '/models')
            if not os.path.exists(full_path + '/model_evaluation'):
                os.mkdir(full_path + '/model_evaluation')
            if not os.path.exists(full_path + '/models/pickledModels'):
                os.mkdir(full_path + '/models/pickledModels')
            cv_dataset_paths = list(glob.glob(full_path + "/CVDatasets/*_CV_*Train.csv"))
            cv_partitions = len(cv_dataset_paths)
            for cv_count in range(cv_partitions):
                for algorithm in self.algorithms:
                    # logging.info("Running Model "+str(algorithm))
                    if (not self.do_lcs_sweep) or (algorithm not in ['eLCS', 'XCS', 'ExSTraCS']):
                        model = model_str_to_obj(algorithm)(cv_folds=3,
                                                            scoring_metric=self.scoring_metric,
                                                            metric_direction=self.metric_direction,
                                                            random_state=self.random_state,
                                                            cv=None, n_jobs=self.n_jobs)
                    else:
                        model = model_str_to_obj(algorithm)(cv_folds=3,
                                                            scoring_metric=self.scoring_metric,
                                                            metric_direction=self.metric_direction,
                                                            random_state=self.random_state,
                                                            cv=None, n_jobs=self.n_jobs,
                                                            iterations=self.lcs_iterations,
                                                            N=self.lcs_n, nu=self.lcs_nu)

                    job_obj = ModelJob(full_path, self.output_path, self.experiment_name, cv_count, self.class_label,
                                       self.instance_label, self.scoring_metric, self.metric_direction, self.n_trials,
                                       self.timeout, self.uniform_fi, self.save_plots, self.random_state)
                    if run_parallel:
                        p = multiprocessing.Process(target=runner_fn, args=(job_obj, model))
                        job_list.append(p)
                    else:
                        job_obj.run(model)
        if run_parallel:
            self.run_jobs(job_list)

    @staticmethod
    def run_jobs(job_list):
        for j in job_list:
            j.start()
        for j in job_list:
            j.join()


def runner_fn(job_obj, model):
    job_obj.run(model)
