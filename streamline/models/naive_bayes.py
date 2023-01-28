import copy
import logging
from abc import ABC
from sklearn.model_selection import cross_val_score
from streamline.modeling.basemodel import BaseModel
from streamline.modeling.parameters import get_parameters
from sklearn.naive_bayes import GaussianNB as NB


class NaiveBayes(BaseModel, ABC):
    def __init__(self, cv_folds=5, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(NB, "Naive Bayes", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = get_parameters(self.model_name)
        self.small_name = "NB"
        self.n_jobs = n_jobs

    def objective(self, trial):
        self.params = {}
        model = copy.deepcopy(self.model).set_params(**self.params)

        mean_cv_score = cross_val_score(model, self.x_train, self.y_train,
                                        scoring=self.scoring_metric,
                                        cv=self.cv, n_jobs=self.n_jobs).mean()
        logging.debug("Trail Completed")
        return mean_cv_score
