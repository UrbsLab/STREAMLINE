import os
import csv
import time
import random
import pickle
import logging
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from skrebate import MultiSURF, TURF
from streamline.utils.job import Job
from streamline.utils.dataset import Dataset
from streamline.modeling.utils import num_cores


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
        self.cv_count = None
        self.dataset = None
        self.cv_train_path = cv_train_path
        self.experiment_path = experiment_path
        self.outcome_label = outcome_label
        self.instance_label = instance_label
        self.instance_subset = instance_subset
        self.algorithm = algorithm
        self.use_turf = use_turf
        self.turf_pct = turf_pct
        self.random_state = random_state
        self.n_jobs = n_jobs

    def run(self):
        """
        Run all elements of the feature importance evaluation:
        applies either mutual information and multisurf and saves a sorted dictionary
        of features with associated scores

        """

        self.job_start_time = time.time()
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        self.prepare_data()
        logging.info('Prepared Train and Test for: ' + str(self.dataset.name) + "_CV_" + str(self.cv_count))

        assert (self.algorithm == 'MI' or self.algorithm == 'MS')
        # Apply mutual information if specified by user
        if self.algorithm == 'MI':
            logging.info('Running Mutual Information...')
            scores, output_path, alg_name = self.run_mutual_information()
        # Apply MultiSURF if specified by user
        elif self.algorithm == 'MS':
            logging.info('Running MultiSURF...')
            scores, output_path, alg_name = self.run_multi_surf()
        else:
            raise Exception("Feature importance algorithm not found")

        logging.info('Sort and pickle feature importance scores...')
        header = self.dataset.data.columns.values.tolist()
        header.remove(self.outcome_label)
        if self.instance_label is not None:
            header.remove(self.instance_label)
        # Save sorted feature importance scores:
        score_dict, score_sorted_features = self.sort_save_fi_scores(scores, header, alg_name)
        # Pickle feature importance information to be used in Phase 4 (feature selection)
        self.pickle_scores(alg_name, scores, score_dict, score_sorted_features)
        # Save phase runtime
        self.save_runtime(alg_name)
        # Print phase completion
        logging.info(self.dataset.name + " CV" + str(self.cv_count) + " phase 3 "
                     + alg_name + " evaluation complete")
        job_file = open(
            self.experiment_path + '/jobsCompleted/job_' + alg_name + '_'
            + self.dataset.name + '_' + str(self.cv_count) + '.txt', 'w')
        job_file.write('complete')
        job_file.close()

    def prepare_data(self):
        """
        Loads target cv training dataset, separates class from features and removes instance labels.
        """
        self.dataset = Dataset(self.cv_train_path, self.outcome_label, instance_label=self.instance_label)
        self.dataset.name = self.cv_train_path.split('/')[-3]
        self.dataset.instance_label = self.instance_label
        self.dataset.outcome_label = self.outcome_label
        self.cv_count = self.cv_train_path.split('/')[-1].split("_")[-2]

    def run_mutual_information(self):
        """
        Run mutual information on target training dataset and return scores as well as file path/name information.
        """
        alg_name = "mutual_information"
        output_path = self.experiment_path + '/' + self.dataset.name + "/feature_selection/" \
                      + alg_name + '/' + alg_name + "_scores_cv_" + str(self.cv_count) + '.csv'
        if not os.path.exists(self.experiment_path + '/' + self.dataset.name + "/feature_selection/" + alg_name + "/"):
            os.makedirs(self.experiment_path + '/' + self.dataset.name + "/feature_selection/" + alg_name + "/")
        scores = mutual_info_classif(self.dataset.feature_only_data(), self.dataset.get_outcome(),
                                     random_state=self.random_state)
        return scores, output_path, alg_name

    def run_multi_surf(self):
        """
        Run multiSURF (a Relief-based feature importance algorithm able to detect both univariate
        and interaction effects) and return scores as well as file path/name information
        """
        # Format instance sampled dataset (prevents MultiSURF from running a very long time in large instance spaces)

        #############
        # Code portion that's problematic
        # TODO: Debug
        # data_features = self.dataset.feature_only_data()
        # print(data_features.shape, self.dataset.get_outcome().shape, self.dataset.data.shape)
        # print(len(self.dataset.data.columns))
        # print(len(data_features.columns))
        # print(data_features.shape, self.dataset.get_outcome().shape)
        # formatted = np.insert(data_features, data_features.shape[1], self.dataset.get_outcome(), 1)
        #
        # choices = np.random.choice(formatted.shape[0], min(self.instance_subset, formatted.shape[0]), replace=False)
        # new_l = list()
        # for i in choices:
        #     new_l.append(formatted[i])
        # formatted = np.array(new_l)
        # data_features = np.delete(formatted, -1, axis=1)
        # data_phenotypes = formatted[:, -1]
        ##############

        # New code
        headers = list(self.dataset.data.columns)
        if self.instance_label:
            headers.remove(self.instance_label)
        headers.remove(self.outcome_label)
        data_features = self.dataset.data[headers + [self.outcome_label, ]]
        n = data_features.shape[0]
        if self.instance_subset is not None:
            n = min(data_features.shape[0], self.instance_subset)
        data_features = data_features.sample(n)
        data_phenotypes = data_features[self.outcome_label]
        data_features = data_features.drop(self.outcome_label, axis=1)

        # Run MultiSURF
        alg_name = "multisurf"
        if not os.path.exists(self.experiment_path + '/' + self.dataset.name + "/feature_selection/" + alg_name + "/"):
            os.makedirs(self.experiment_path + '/' + self.dataset.name + "/feature_selection/" + alg_name + "/")
        output_path = self.experiment_path + '/' + self.dataset.name + "/feature_selection/" + alg_name + "/" \
                    + alg_name + "_scores_cv_" + str(self.cv_count) + '.csv'

        if self.n_jobs is None:
            self.n_jobs = 1

        if self.use_turf:
            try:
                clf = TURF(MultiSURF(n_jobs=self.n_jobs), pct=self.turf_pct).fit(data_features.values,
                                                                                 data_phenotypes.values)
            except ModuleNotFoundError:
                raise Exception("sk-rebate version error")
        else:
            clf = MultiSURF(n_jobs=self.n_jobs).fit(data_features.values, data_phenotypes.values)
        scores = clf.feature_importances_
        return scores, output_path, alg_name

    def pickle_scores(self, output_name, scores, score_dict, score_sorted_features):
        """
        Pickle the scores, score dictionary and features sorted by score to be used primarily
        in phase 4 (feature selection) of pipeline
        """
        # Save Scores to pickled file for later use
        outfile = open(
            self.experiment_path + '/' + self.dataset.name + "/feature_selection/" + output_name
            + "/pickledForPhase4/" + str(self.cv_count) + '.pickle', 'wb')
        pickle.dump([scores, score_dict, score_sorted_features], outfile)
        outfile.close()

    def save_runtime(self, output_name):
        """
        Save phase runtime
        Args:
            output_name: name of the output tag
        """
        runtime_file = open(
            self.experiment_path + '/' + self.dataset.name + '/runtime/runtime_' + output_name + '_CV_'
            + str(self.cv_count) + '.txt', 'w')
        runtime_file.write(str(time.time() - self.job_start_time))
        runtime_file.close()

    def sort_save_fi_scores(self, scores, ordered_feature_names, alg_name):
        """
        Creates a feature score dictionary and a dictionary sorted by decreasing feature importance scores.

        Args:
            scores:
            ordered_feature_names:
            alg_name:

        Returns: score_dict, score_sorted_features - dictionary of scores and score sorted name of features

        """
        # Put list of scores in dictionary
        score_dict = {}
        i = 0
        for each in ordered_feature_names:
            score_dict[each] = scores[i]
            i += 1
        # Sort features by decreasing score
        filename = self.experiment_path + '/' \
                   + self.dataset.name + "/feature_selection/" \
                   + alg_name + '/' + alg_name + "_scores_cv_" + str(self.cv_count) + '.csv'

        score_sorted_features = sorted(score_dict, key=lambda x: score_dict[x], reverse=True)
        # Save scores to 'formatted' file
        with open(filename, mode='w', newline="") as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Sorted " + alg_name + " Scores"])
            for k in score_sorted_features:
                writer.writerow([k, score_dict[k]])
        file.close()
        return score_dict, score_sorted_features

    # def __getstate__(self):
    #     """called when pickling - this hack allows subprocesses to
    #        be spawned without the AuthenticationString raising an error"""
    #     state = self.__dict__.copy()
    #     conf = state['_config']
    #     if 'authkey' in conf:
    #         # del conf['authkey']
    #         conf['authkey'] = bytes(conf['authkey'])
    #     return state
    #
    # def __setstate__(self, state):
    #     """for unpickling"""
    #     state['_config']['authkey'] = state['_config']['authkey']
    #     self.__dict__.update(state)
