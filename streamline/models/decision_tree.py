from abc import ABC
from streamline.modeling.basemodel import BaseModel
from streamline.modeling.parameters import get_parameters
from sklearn.tree import DecisionTreeClassifier as DT


class DecisionTreeClassifier(BaseModel, ABC):
    def __init__(self, cv_folds=3, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(DT, "Decision Tree", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = get_parameters(self.model_name)
        self.param_grid['random_state'] = [random_state, ]
        self.small_name = "DT"
        self.color = "yellow"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {'criterion': trial.suggest_categorical('criterion', self.param_grid['criterion']),
                       'splitter': trial.suggest_categorical('splitter', self.param_grid['splitter']),
                       'max_depth': trial.suggest_int('max_depth', self.param_grid['max_depth'][0],
                                                      self.param_grid['max_depth'][1]),
                       'min_samples_split': trial.suggest_int('min_samples_split',
                                                              self.param_grid['min_samples_split'][0],
                                                              self.param_grid['min_samples_split'][1]),
                       'min_samples_leaf': trial.suggest_int('min_samples_leaf', self.param_grid['min_samples_leaf'][0],
                                                             self.param_grid['min_samples_leaf'][1]),
                       'max_features': trial.suggest_categorical('max_features', self.param_grid['max_features']),
                       'class_weight': trial.suggest_categorical('class_weight', self.param_grid['class_weight']),
                       'random_state': trial.suggest_categorical('random_state', self.param_grid['random_state'])}

        mean_cv_score = self.hyper_eval()
        return mean_cv_score
