from abc import ABC
from streamline.modeling.basemodel import BaseModel
from streamline.modeling.parameters import get_parameters
from sklearn.svm import SVC as SVC


class SupportVectorClassifier(BaseModel, ABC):
    def __init__(self, cv_folds=3, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(SVC, "Support Vector Machine", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = get_parameters(self.model_name)
        self.param_grid['random_state'] = [random_state, ]
        self.small_name = "SVM"
        self.color = "orange"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {'kernel': trial.suggest_categorical('kernel', self.param_grid['kernel']),
                       'C': trial.suggest_loguniform('C', self.param_grid['C'][0], self.param_grid['C'][1]),
                       'gamma': trial.suggest_categorical('gamma', self.param_grid['gamma']),
                       'degree': trial.suggest_int('degree', self.param_grid['degree'][0],
                                                   self.param_grid['degree'][1]),
                       'probability': trial.suggest_categorical('probability', self.param_grid['probability']),
                       'class_weight': trial.suggest_categorical('class_weight', self.param_grid['class_weight']),
                       'random_state': trial.suggest_categorical('random_state', self.param_grid['random_state'])}
        mean_cv_score = self.hypereval(trial)
        return mean_cv_score
