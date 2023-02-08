from abc import ABC
from streamline.modeling.basemodel import BaseModel
from streamline.modeling.parameters import get_parameters
from skeLCS import eLCS
from skXCS import XCS
from skExSTraCS import ExSTraCS


class eLCSClassifier(BaseModel, ABC):
    model_name = "eLCS"
    small_name = "eLCS"
    color = "green"

    def __init__(self, cv_folds=3, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None,
                 iterations=None, N=None, nu=None):
        super().__init__(eLCS, "eLCS", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = get_parameters(self.model_name)
        if iterations:
            self.param_grid['learning_iterations'] = [iterations, ]
        if N:
            self.param_grid['N'] = [N, ]
        if nu:
            self.param_grid['nu'] = [nu, ]
        self.small_name = "eLCS"
        self.color = "green"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {}
        mean_cv_score = self.hyper_eval()
        return mean_cv_score


class XCSClassifier(BaseModel, ABC):
    model_name = "XCS"
    small_name = "XCS"
    color = "olive"

    def __init__(self, cv_folds=3, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None,
                 iterations=None, N=None, nu=None):
        super().__init__(XCS, "XCS", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = get_parameters(self.model_name)
        if iterations:
            self.param_grid['learning_iterations'] = [iterations, ]
        if N:
            self.param_grid['N'] = [N, ]
        if nu:
            self.param_grid['nu'] = [nu, ]
        self.small_name = "XCS"
        self.color = "olive"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {}
        mean_cv_score = self.hyper_eval()
        return mean_cv_score


class ExSTraCSClassifier(BaseModel, ABC):
    model_name = "ExSTraCS"
    small_name = "ExSTraCS"
    color = "lawngreen"

    def __init__(self, cv_folds=3, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None,
                 iterations=None, N=None, nu=None):
        super().__init__(ExSTraCS, "ExSTraCS", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = get_parameters(self.model_name)
        if iterations:
            self.param_grid['learning_iterations'] = [iterations, ]
        if N:
            self.param_grid['N'] = [N, ]
        if nu:
            self.param_grid['nu'] = [nu, ]
        self.small_name = "ExSTraCS"
        self.color = "lawngreen"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {}
        mean_cv_score = self.hyper_eval()
        return mean_cv_score
