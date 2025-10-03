import os
import csv
import time
import random
import pickle
import logging
import numpy as np
from streamline.utils.job import Job
from streamline.utils.dataset import Dataset
from streamline.featurefns.utils import LABELS
from streamline.featurefns.utils import algorithm_str_to_obj


class FeatureImportance(Job):
    """
    Initializer for Feature Importance Job
    """

    def __init__(self, cv_train_path, experiment_path, outcome_label, instance_label=None, instance_subset=2000,
                 algorithm="MS", use_turf=True, turf_pct=True, random_state=None, n_jobs=None):
        """

        Args:
            cv_train_path: path for the cross-validation dataset created
            experiment_path:
            outcome_label:
            instance_label:
            instance_subset:
            algorithm:
            use_turf:
            turf_pct:
            random_state:
            n_jobs:

        """
        super().__init__()
        self.name = algorithm
        params = (('use_turf', use_turf), ('turf_pct', turf_pct))
        if not (self.name in LABELS):
            raise Exception("Feature importance algorithm not found")
        self.algorithm = algorithm_str_to_obj(algorithm)(cv_train_path, experiment_path, outcome_label,
                                                         instance_label=instance_label, instance_subset=instance_subset,
                                                         params=params, random_state=random_state, n_jobs=n_jobs)
        self.random_state = random_state

    def run(self):
        """
        Run all elements of the feature importance evaluation:
        applies either mutual information and multisurf and saves a sorted dictionary
        of features with associated scores

        """

        self.job_start_time = time.time()
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        self.algorithm.prepare_data()
        logging.info('Prepared Train and Test for: ' + str(self.algorithm.dataset.name)
                     + "_CV_" + str(self.algorithm.cv_count))

        scores, output_path = self.algorithm.run_algorithm()

        logging.info('Sort and pickle feature importance scores...')
        header = self.algorithm.dataset.data.columns.values.tolist()
        header.remove(self.algorithm.outcome_label)
        if self.algorithm.instance_label is not None:
            header.remove(self.algorithm.instance_label)
        # Save sorted feature importance scores:
        score_dict, score_sorted_features = self.algorithm.sort_save_fi_scores(scores, header,
                                                                               self.algorithm.path_name)
        # Pickle feature importance information to be used in Phase 4 (feature selection)
        self.algorithm.pickle_scores(self.algorithm.path_name, scores, score_dict, score_sorted_features)
        # Save phase runtime
        self.save_runtime(self.algorithm.path_name)
        # Print phase completion
        logging.info(self.algorithm.dataset.name + " CV " + str(self.algorithm.cv_count) + " phase 3 "
                     + self.algorithm.model_name + " evaluation complete")
        job_file = open(
            self.algorithm.experiment_path + '/jobsCompleted/job_' + self.algorithm.path_name + '_'
            + self.algorithm.dataset.name + '_' + str(self.algorithm.cv_count) + '.txt', 'w')
        job_file.write('complete')
        job_file.close()

    def save_runtime(self, output_name):
        """
        Save phase runtime
        Args:
            output_name: name of the output tag
        """
        runtime_file = open(
            self.algorithm.experiment_path + '/' + self.algorithm.dataset.name
            + '/runtime/runtime_' + output_name + '_CV_'
            + str(self.algorithm.cv_count) + '.txt', 'w')
        runtime_file.write(str(time.time() - self.job_start_time))
        runtime_file.close()
