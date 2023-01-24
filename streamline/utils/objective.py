import numpy as np
from sklearn import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score


class Objective(object):
    def __init__(self, model, params, x_train, y_train, cv_folds,
                 scoring_metric, random_state=None):
        self.model = model
        self.params = params
        self.x_train = x_train
        self.y_train = y_train
        self.scoring_metric = scoring_metric
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

    def __call__(self, trial):
        model = clone(self.model).set_params(**self.params)

        mean_cv_score = cross_val_score(model, self.x_train, self.y_train,
                                        scoring=self.scoring_metric,
                                        cv=self.cv, n_jobs=-1).mean()
        return mean_cv_score
