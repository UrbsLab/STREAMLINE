import os
from sklearn.feature_selection import mutual_info_classif
from streamline.featurefns.feature_algorithm import FeatureAlgorithm


class MutualInformation(FeatureAlgorithm):
    """
    Initializer for Feature Importance Job
    """
    model_name = "Mutual Information"
    small_name = "MI"
    path_name = "mutual_information"

    def __init__(self, cv_train_path, experiment_path, outcome_label, instance_label=None, instance_subset=2000,
                 params=None, random_state=None, n_jobs=None):
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
        super().__init__(cv_train_path, experiment_path, outcome_label, instance_label=instance_label,
                         instance_subset=instance_subset,
                         random_state=random_state, n_jobs=n_jobs)
        self.cv_count = None
        self.params = params
        self.n_jobs = n_jobs

    def run_algorithm(self):
        """
        Run mutual information on target training dataset and return scores as well as file path/name information.
        """
        alg_name = self.path_name
        output_path = self.experiment_path + '/' + self.dataset.name + "/feature_selection/" \
                      + alg_name + '/' + alg_name + "_scores_cv_" + str(self.cv_count) + '.csv'
        if not os.path.exists(self.experiment_path + '/' + self.dataset.name + "/feature_selection/" + alg_name + "/"):
            os.makedirs(self.experiment_path + '/' + self.dataset.name + "/feature_selection/" + alg_name + "/")
        scores = mutual_info_classif(self.dataset.feature_only_data(), self.dataset.get_outcome(),
                                     random_state=self.random_state)
        return scores, output_path
