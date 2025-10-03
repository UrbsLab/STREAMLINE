import os
import csv
import pickle
from streamline.utils.job import Job
from streamline.utils.dataset import Dataset


class FeatureAlgorithm(Job):
    """
    Initializer for Feature Importance Job
    """

    def __init__(self, cv_train_path, experiment_path, outcome_label, instance_label=None, params=None,
                 instance_subset=2000, random_state=None, n_jobs=None):
        """
        Args:
            cv_train_path: path for the cross-validation dataset created
            experiment_path:
            outcome_label:
            instance_label:
            instance_subset:
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
        self.params = params
        self.instance_subset = instance_subset
        self.random_state = random_state
        self.n_jobs = n_jobs

    def prepare_data(self):
        """
        Loads target cv training dataset, separates class from features and removes instance labels.
        """
        self.dataset = Dataset(self.cv_train_path, self.outcome_label, instance_label=self.instance_label)
        self.dataset.name = self.cv_train_path.split('/')[-3]
        self.dataset.instance_label = self.instance_label
        self.dataset.outcome_label = self.outcome_label
        self.cv_count = self.cv_train_path.split('/')[-1].split("_")[-2]

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

    def run_algorithm(self):
        pass
