import os
from skrebate import TURF
from skrebate import MultiSURFstar as MultiSURFstarClass
from streamline.featurefns.feature_algorithm import FeatureAlgorithm


class MultiSURFStar(FeatureAlgorithm):
    """
    Initializer for Feature Importance Job
    """
    model_name = "MultiSURFstar"
    small_name = "MSS"
    path_name = "multisurfstar"

    def __init__(self, cv_train_path, experiment_path, outcome_label, instance_label=None, instance_subset=2000,
                 params=(('use_turf', True), ('turf_pct', True)), random_state=None, n_jobs=None):
        """

        Args:
            cv_train_path: path for the cross-validation dataset created
            experiment_path:
            outcome_label:
            instance_label:
            instance_subset:
            algorithm:
            random_state:
            n_jobs:

        """
        super().__init__(cv_train_path, experiment_path, outcome_label, instance_label=instance_label,
                         instance_subset=instance_subset,
                         random_state=random_state, n_jobs=n_jobs)
        self.cv_count = None
        params = dict(params)
        self.use_turf = params['use_turf']
        self.turf_pct = params['turf_pct']
        self.n_jobs = n_jobs

    def run_algorithm(self):
        """
        Run multiSURF (a Relief-based feature importance algorithm able to detect both univariate
        and interaction effects) and return scores as well as file path/name information
        """
        # Format instance sampled dataset (prevents MultiSURF from running a very long time in large instance spaces)

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

        # Run MultiSURFstar
        alg_name = self.path_name
        if not os.path.exists(self.experiment_path + '/' + self.dataset.name + "/feature_selection/" + alg_name + "/"):
            os.makedirs(self.experiment_path + '/' + self.dataset.name + "/feature_selection/" + alg_name + "/")
        output_path = self.experiment_path + '/' + self.dataset.name + "/feature_selection/" + alg_name + "/" \
                      + alg_name + "_scores_cv_" + str(self.cv_count) + '.csv'

        if self.n_jobs is None:
            self.n_jobs = 1

        if self.use_turf:
            try:
                clf = TURF(MultiSURFstarClass(n_jobs=self.n_jobs), pct=self.turf_pct).fit(data_features.values,
                                                                                          data_phenotypes.values)
            except ModuleNotFoundError:
                raise Exception("sk-rebate version error")
        else:
            clf = MultiSURFstarClass(n_jobs=self.n_jobs).fit(data_features.values, data_phenotypes.values)
        scores = clf.feature_importances_
        return scores, output_path
