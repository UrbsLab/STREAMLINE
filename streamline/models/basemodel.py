import copy
import logging
import optuna
from sklearn import clone
from sklearn.model_selection import StratifiedKFold


class BaseModel:
    def __init__(self, model, model_name,
                 cv_folds=5, scoring_metric='balanced_accuracy', metric_direction='maximize',
                 random_state=None, cv=None, sampler=None):
        self.is_single = True
        self.model = model
        self.model_name = model_name
        self.y_train = None
        self.x_train = None
        self.param_grid = None
        self.params = None
        self.random_state = random_state
        self.scoring_metric = scoring_metric
        self.metric_direction = metric_direction
        if cv is None:
            self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        else:
            self.cv = cv

        if sampler is None:
            self.sampler = optuna.samplers.TPESampler(seed=self.random_state)
        else:
            self.sampler = sampler
        self.study = None
        optuna.logging.set_verbosity(optuna.logging.INFO)

    def objective(self, trail):
        return 0

    def optimize(self, x_train, y_train, n_trails, timeout):
        self.x_train = x_train
        self.y_train = y_train
        for key, value in self.param_grid.items():
            if len(value) > 1:
                self.is_single = False
                break

        if not self.is_single:
            self.study = optuna.create_study(direction=self.metric_direction, sampler=self.sampler)
            self.study.optimize(lambda trial: self.objective(trial), n_trials=n_trails, timeout=timeout,
                                catch=(ValueError,))


        return list()

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x_in):
        self.model.predict(x_in)
