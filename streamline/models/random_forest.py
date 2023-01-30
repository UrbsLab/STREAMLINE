from abc import ABC
from streamline.modeling.basemodel import BaseModel
from streamline.modeling.parameters import get_parameters
from sklearn.ensemble import RandomForestClassifier as RF


class RandomForest(BaseModel, ABC):
    def __init__(self, cv_folds=5, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(RF, "Random Forest", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = get_parameters(self.model_name)
        self.param_grid['random_state'] = [random_state, ]
        self.small_name = "RF"
        self.n_jobs = n_jobs

    def objective(self, trial):
        self.params = {'n_estimators': trial.suggest_int('n_estimators', self.param_grid['n_estimators'][0],
                                                         self.param_grid['n_estimators'][1]),
                       'criterion': trial.suggest_categorical('criterion', self.param_grid['criterion']),
                       'max_depth': trial.suggest_int('max_depth', self.param_grid['max_depth'][0],
                                                      self.param_grid['max_depth'][1]),
                       'min_samples_split': trial.suggest_int('min_samples_split',
                                                              self.param_grid['min_samples_split'][0],
                                                              self.param_grid['min_samples_split'][1]),
                       'min_samples_leaf': trial.suggest_int('min_samples_leaf', self.param_grid['min_samples_leaf'][0],
                                                             self.param_grid['min_samples_leaf'][1]),
                       'max_features': trial.suggest_categorical('max_features', self.param_grid['max_features']),
                       'bootstrap': trial.suggest_categorical('bootstrap', self.param_grid['bootstrap']),
                       'oob_score': trial.suggest_categorical('oob_score', self.param_grid['oob_score']),
                       'class_weight': trial.suggest_categorical('class_weight', self.param_grid['class_weight']),
                       'random_state': trial.suggest_categorical('random_state', self.param_grid['random_state'])}

        mean_cv_score = self.hypereval(trial)
        return mean_cv_score
