from abc import ABC
from streamline.modeling.submodels import RegressionModel
from sklearn.linear_model import ElasticNet as EN


class ElasticNet(RegressionModel, ABC):
    model_name = "Elastic Net"
    small_name = "EN"
    color = "steelblue"

    def __init__(self, cv_folds=3, scoring_metric='explained_variance',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(EN, "Elastic Net", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = {'alpha': [1e-3, 1], 'l1_ratio': [0, 1], 'max_iter': [2000, 2500],
                           'random_state': [random_state, ]}
        self.small_name = "EN"
        self.color = "steelblue"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {'alpha': trial.suggest_float('alpha', self.param_grid['alpha'][0], self.param_grid['alpha'][1]),
                       'l1_ratio': trial.suggest_float('l1_ratio', self.param_grid['l1_ratio'][0],
                                                       self.param_grid['l1_ratio'][1]),
                       'max_iter': trial.suggest_int('max_iter', self.param_grid['max_iter'][0],
                                                     self.param_grid['max_iter'][1])}

        mean_cv_score = self.hyper_eval()
        return mean_cv_score

    def residual_record(self, x_train, y_train, x_test, y_test):
        y_train_pred = self.predict(x_train)
        y_pred = self.predict(x_test)
        residual_train = y_train - y_train_pred
        residual_test = y_test - y_pred
        return residual_train, residual_test, y_train_pred, y_pred
