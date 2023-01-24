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
                 instance_subset=None, algorithms=("MI", "MS"), n_splits=10, use_turf=True, turf_pct=True,
                 random_state=None, n_jobs=None):
        """

        Args:
            output_path:
            experiment_name:
            class_label:
            instance_label:
            instance_subset:
            algorithms:
            n_splits:
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
        self.algorithms = list(algorithms)
        # assert (algorithms in ["MI", "MS"])
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

            if self.algorithms is not None or self.algorithms != []:
                if not os.path.exists(full_path + "/feature_selection"):
                    os.mkdir(full_path + "/feature_selection")

            if "MI" in self.algorithms:
                if not os.path.exists(full_path + "/feature_selection/mutual_information"):
                    os.mkdir(full_path + "/feature_selection/mutual_information")
                for cv_train_path in glob.glob(full_path + "/CVDatasets/*_CV_*Train.csv"):
                    job_obj = FeatureImportance(cv_train_path, experiment_path, self.class_label,
                                                self.instance_label, self.instance_subset, "MI",
                                                self.use_turf, self.turf_pct, self.random_state, self.n_jobs)
                    if run_parallel:
                        p = multiprocessing.Process(target=runner_fn, args=(job_obj,))
                        job_list.append(p)
                    else:
                        job_obj.run()

            if "MS" in self.algorithms:
                if not os.path.exists(full_path + "/feature_selection/multisurf"):
                    os.mkdir(full_path + "/feature_selection/multisurf")
                for cv_train_path in glob.glob(full_path + "/CVDatasets/*_CV_*Train.csv"):
                    job_obj = FeatureImportance(cv_train_path, experiment_path, self.class_label,
                                                self.instance_label, self.instance_subset, "MS",
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


class FeatureSelectionRunner:
    """
    Runner Class for running feature selection jobs for
    cross-validation splits.
    """
    def __init__(self, output_path, experiment_name, algorithms, class_label="Class", instance_label=None,
                 max_features_to_keep=2000, filter_poor_features=True, top_features=40, export_scores=True,
                 overwrite_cv=True, random_state=None, n_jobs=None):
        """

        Args:
            output_path: path other the output folder
            experiment_name: name for the current experiment
            algorithms: feature selection algorithms from last phase
            max_features_to_keep: max features to keep (only applies if filter_poor_features is True), default=2000
            filter_poor_features: filter out the worst performing features prior to modeling,default='True'
            top_features: number of top features to illustrate in figures, default=40)
            export_scores: export figure summarizing average fi scores over cv partitions, default='True'
            overwrite_cv: overwrites working cv datasets with new feature subset datasets,default="True"
            random_state: random seed for reproducibility
            n_jobs: n_jobs param for multiprocessing

        Returns: None

        """
        self.cv_count = None
        self.dataset = None
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.class_label = class_label
        self.instance_label = instance_label

        self.max_features_to_keep = max_features_to_keep
        self.filter_poor_features = filter_poor_features
        self.top_features = top_features
        self.export_scores = export_scores
        self.overwrite_cv = overwrite_cv

        self.algorithms = algorithms
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
            experiment_path = self.output_path + '/' + self.experiment_name
            cv_dataset_paths = list(glob.glob(full_path + "/CVDatasets/*_CV_*Train.csv"))
            job_obj = FeatureSelection(full_path, len(cv_dataset_paths), self.algorithms,
                                       self.class_label, self.instance_label, self.export_scores,
                                       self.top_features, self.max_features_to_keep,
                                       self.filter_poor_features, self.overwrite_cv)
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
    job_obj.run()
