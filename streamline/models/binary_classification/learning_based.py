import logging
from abc import ABC
from streamline.modeling.submodels import BinaryClassificationModel
from streamline.modeling.parameters import get_parameters
from skeLCS import eLCS
from skXCS import XCS
from skExSTraCS import ExSTraCS


class eLCSClassifier(BinaryClassificationModel, ABC):
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
        self.param_grid['random_state'] = [random_state, ]
        self.small_name = "eLCS"
        self.color = "green"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {
            'learning_iterations': trial.suggest_categorical('learning_iterations',
                                                             self.param_grid['learning_iterations']),
            'N': trial.suggest_categorical('N', self.param_grid['N']),
            'nu': trial.suggest_categorical('nu', self.param_grid['nu']),
            'random_state': trial.suggest_categorical('random_state', self.param_grid['random_state'])}
        mean_cv_score = self.hyper_eval()
        return mean_cv_score


class XCSClassifier(BinaryClassificationModel, ABC):
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
        self.param_grid['random_state'] = [random_state, ]
        self.small_name = "XCS"
        self.color = "olive"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = params = {
            'learning_iterations': trial.suggest_categorical('learning_iterations',
                                                             self.param_grid['learning_iterations']),
            'N': trial.suggest_categorical('N', self.param_grid['N']),
            'nu': trial.suggest_categorical('nu', self.param_grid['nu']),
            'random_state': trial.suggest_categorical('random_state', self.param_grid['random_state'])}

        mean_cv_score = self.hyper_eval()
        return mean_cv_score


class ExSTraCSClassifier(BinaryClassificationModel, ABC):
    model_name = "ExSTraCS"
    small_name = "ExSTraCS"
    color = "lawngreen"

    def __init__(self, cv_folds=3, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None,
                 iterations=None, N=None, nu=None, expert_knowledge=None):
        super().__init__(ExSTraCS, "ExSTraCS", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = get_parameters(self.model_name)
        if iterations:
            self.param_grid['learning_iterations'] = [iterations, ]
        if N:
            self.param_grid['N'] = [N, ]
        if nu:
            self.param_grid['nu'] = [nu, ]
        self.param_grid['expert_knowledge'] = expert_knowledge
        self.param_grid['random_state'] = [random_state, ]
        self.small_name = "ExSTraCS"
        self.color = "lawngreen"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {'learning_iterations': trial.suggest_categorical('learning_iterations',
                                                                        self.param_grid['learning_iterations']),
                       'N': trial.suggest_categorical('N', self.param_grid['N']),
                       'nu': trial.suggest_categorical('nu', self.param_grid['nu']),
                       'random_state': trial.suggest_categorical('random_state', self.param_grid['random_state']),
                       'expert_knowledge': self.param_grid['expert_knowledge'],
                       'rule_compaction': trial.suggest_categorical('rule_compaction',
                                                                    self.param_grid['rule_compaction'])}

        mean_cv_score = self.hyper_eval()
        return mean_cv_score
