import os
import glob
import multiprocessing
from streamline.featurefns.selection import FeatureSelection
from streamline.featurefns.importance import FeatureImportance


class FeatureImportanceRunner:
    """
    Runner Class for running feature importance jobs for
    cross-validation splits.
    """
    def __init__(self, output_path, experiment_name, class_label="Class", instance_label=None,
                 instance_subset=None, algorithm="MS", use_turf=True, turf_pct=True,
                 random_state=None, n_jobs=None):
        """

        Args:
            output_path:
            experiment_name:
            class_label:
            instance_label:
            instance_subset:
            algorithm:
            use_turf:
            turf_pct:
            random_state:
            n_jobs:

        Returns: None

        """
        self.cv_count = None
        self.dataset = None
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.class_label = class_label
        self.instance_label = instance_label
        self.instance_subset = instance_subset
        self.algorithm = algorithm
        assert (algorithm in ["MI", "MS"])
        self.use_turf = use_turf
        self.turf_pct = turf_pct
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Argument checks
        if not os.path.exists(self.output_path):
            raise Exception("Output path must exist (from phase 1) before phase 3 can begin")
        if not os.path.exists(self.output_path + '/' + self.experiment_name):
            raise Exception("Experiment must exist (from phase 1) before phase 3 can begin")

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
            experiment_path = self.output_path + '/' + self.experiment_name

            if self.algorithm:
                if not os.path.exists(full_path + "/feature_selection"):
                    os.mkdir(full_path + "/feature_selection")

            if self.algorithm == "MI":
                if not os.path.exists(full_path + "/feature_selection/mutual_information"):
                    os.mkdir(full_path + "/feature_selection/mutual_information")
                for cv_train_path in glob.glob(full_path + "/CVDatasets/*_CV_*Train.csv"):
                    job_obj = FeatureImportance(cv_train_path, experiment_path, self.class_label,
                                                self.instance_label, self.instance_subset, self.algorithm,
                                                self.use_turf, self.turf_pct, self.random_state, self.n_jobs)
                    if run_parallel:
                        p = multiprocessing.Process(target=runner_fn, args=(job_obj,))
                        job_list.append(p)
                    else:
                        job_obj.run()

            if self.algorithm == "MS":
                if not os.path.exists(full_path + "/feature_selection/multisurf"):
                    os.mkdir(full_path + "/feature_selection/multisurf")
                for cv_train_path in glob.glob(full_path + "/CVDatasets/*_CV_*Train.csv"):
                    job_obj = FeatureImportance(cv_train_path, experiment_path, self.class_label,
                                                self.instance_label, self.instance_subset, self.algorithm,
                                                self.use_turf, self.turf_pct, self.random_state, self.n_jobs)
                    if run_parallel:
                        p = multiprocessing.Process(target=runner_fn, args=(job_obj,))
                        job_list.append(p)
                    else:
                        job_obj.run()
        if run_parallel:
            self.run_jobs(job_list)

    @staticmethod
    def run_jobs(job_list):
        for j in job_list:
            j.start()
        for j in job_list:
            j.join()


def runner_fn(job_obj):
    job_obj.run