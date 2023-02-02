from abc import ABC
from streamline.modeling.basemodel import BaseModel
from streamline.modeling.parameters import get_parameters
from sklearn.neighbors import KNeighborsClassifier as KNN


class KNNClassifier(BaseModel, ABC):
    def __init__(self, cv_folds=3, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(KNN, "K-Nearest Neighbors", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = get_parameters(self.model_name)
        self.param_grid['random_state'] = [random_state, ]
        self.small_name = "KNN"
        self.color = "chocolate"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {
            'n_neighbors': trial.suggest_int('n_neighbors', self.param_grid['n_neighbors'][0],
                                             self.param_grid['n_neighbors'][1]),
            'weights': trial.suggest_categorical('weights', self.param_grid['weights']),
            'p': trial.suggest_int('p', self.param_grid['p'][0], self.param_grid['p'][1]),
            'metric': trial.suggest_categorical('metric', self.param_grid['metric'])}
        mean_cv_score = self.hyper_eval()
        return mean_cv_score
