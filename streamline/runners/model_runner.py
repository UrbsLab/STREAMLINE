import logging
import os
import glob
import multiprocessing
import pickle

from streamline.modeling.utils import ABBREVIATION
from streamline.modeling.modeljob import ModelJob
from streamline.modeling.utils import model_str_to_obj
from streamline.modeling.utils import SUPPORTED_MODELS
from streamline.modeling.utils import is_supported_model
from streamline.utils.runners import model_runner_fn, run_jobs


class ModelExperimentRunner:
    """
    Runner Class for running all the model jobs for
    cross-validation splits.
    """

    def __init__(self, output_path, experiment_name, algorithms=None, exclude=None, class_label="Class",
                 instance_label=None, scoring_metric='balanced_accuracy', metric_direction='maximize',
                 training_subsample=0, use_uniform_fi=True, n_trials=200,
                 timeout=900, save_plots=False, do_lcs_sweep=False, lcs_nu=1, lcs_n=2000, lcs_iterations=200000,
                 lcs_timeout=1200, resubmit=False, random_state=None, n_jobs=None):

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
            self.algorithms = list()
            for algorithm in algorithms:
                self.algorithms.append(is_supported_model(algorithm))

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
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Argument checks
        if not os.path.exists(self.output_path):
            raise Exception("Output path must exist (from phase 1) before phase 4 can begin")
        if not os.path.exists(self.output_path + '/' + self.experiment_name):
            raise Exception("Experiment must exist (from phase 1) before phase 4 can begin")

        self.save_metadata()

    def run(self, run_parallel):

        # Iterate through datasets, ignoring common folders
        dataset_paths = os.listdir(self.output_path + "/" + self.experiment_name)
        remove_list = ['metadata.pickle', 'metadata.csv', 'algInfo.pickle', 'jobsCompleted',
                       'logs', 'jobs', 'DatasetComparisons']

        for text in remove_list:
            if text in dataset_paths:
                dataset_paths.remove(text)

        job_list = list()

        if self.resubmit:
            phase5completed = []
            for filename in glob.glob(self.output_path + "/" + self.experiment_name + '/jobsCompleted/job_model*'):
                ref = filename.split('/')[-1]
                phase5completed.append(ref)
        else:
            phase5completed = []

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
                    abbrev = ABBREVIATION[algorithm]
                    target_file = 'job_model_' + dataset_directory_path + '_' + str(cv_count) + '_' + \
                                  abbrev + '.txt'
                    if target_file in phase5completed:
                        continue
                        # target for a re-submit

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
                        p = multiprocessing.Process(target=model_runner_fn, args=(job_obj, model))
                        job_list.append(p)
                    else:
                        job_obj.run(model)
        if run_parallel:
            run_jobs(job_list)

    def save_metadata(self):
        # Load metadata
        file = open(self.output_path + '/' + self.experiment_name + '/' + "metadata.pickle", 'rb')
        metadata = pickle.load(file)
        file.close()
        # Update metadata
        metadata['Naive Bayes'] = str('Naive Bayes' in self.algorithms)
        metadata['Logistic Regression'] = str('Logistic Regression' in self.algorithms)
        metadata['Decision Tree'] = str('Decision Tree' in self.algorithms)
        metadata['Random Forest'] = str('Random Forest' in self.algorithms)
        metadata['Gradient Boosting'] = str('Gradient Boosting' in self.algorithms)
        metadata['Extreme Gradient Boosting'] = str('Extreme Gradient Boosting' in self.algorithms)
        metadata['Light Gradient Boosting'] = str('Light Gradient Boosting' in self.algorithms)
        metadata['Category Gradient Boosting'] = str('Category Gradient Boosting' in self.algorithms)
        metadata['Support Vector Machine'] = str('Support Vector Machine' in self.algorithms)
        metadata['Artificial Neural Network'] = str('Artificial Neural Network' in self.algorithms)
        metadata['K-Nearest Neighbors'] = str('K-Nearest Neighbors' in self.algorithms)
        metadata['Genetic Programming'] = str('Genetic Programming' in self.algorithms)
        metadata['eLCS'] = str('eLCS' in self.algorithms)
        metadata['XCS'] = str('XCS' in self.algorithms)
        metadata['ExSTraCS'] = str('ExSTraCS' in self.algorithms)
        # Add new algorithms here...
        metadata['Primary Metric'] = self.scoring_metric
        metadata['Training Subsample for KNN,ANN,SVM,and XGB'] = self.training_subsample
        metadata['Uniform Feature Importance Estimation (Models)'] = self.uniform_fi
        metadata['Hyperparameter Sweep Number of Trials'] = self.n_trials
        metadata['Hyperparameter Timeout'] = self.timeout
        metadata['Export Hyperparameter Sweep Plots'] = self.save_plots
        metadata['Do LCS Hyperparameter Sweep'] = self.do_lcs_sweep
        metadata['nu'] = self.lcs_nu
        metadata['Training Iterations'] = self.lcs_iterations
        metadata['N (Rule Population Size)'] = self.lcs_n
        metadata['LCS Hyperparameter Sweep Timeout'] = self.lcs_timeout
        # Pickle the metadata for future use
        pickle_out = open(self.output_path + '/' + self.experiment_name + '/' + "metadata.pickle", 'wb')
        pickle.dump(metadata, pickle_out)
        pickle_out.close()
