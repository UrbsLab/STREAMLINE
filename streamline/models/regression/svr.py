from abc import ABC
from streamline.modeling.submodels import RegressionModel
from streamline.modeling.parameters import get_parameters
from sklearn.svm import SVR as SVRModel


class SVR(RegressionModel, ABC):
    model_name = "Support Vector Regression"
    small_name = "SVR"
    color = "dimgrey"

    def __init__(self, cv_folds=3, scoring_metric='explained_variance',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(SVRModel, "Support Vector Regression", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = get_parameters(self.model_name, model_type="Regression")
        self.param_grid['random_state'] = [random_state, ]
        self.small_name = "SVR"
        self.color = "dimgrey"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {'kernel': trial.suggest_categorical('kernel', self.param_grid['kernel']),
                       'C': trial.suggest_float('C', self.param_grid['C'][0], self.param_grid['C'][1]),
                       'gamma': trial.suggest_categorical('gamma', self.param_grid['gamma']),
                       'degree': trial.suggest_int('degree', self.param_grid['degree'][0],
                                                   self.param_grid['degree'][1])}

        mean_cv_score = self.hyper_eval()
        return mean_cv_score

    def residual_record(self, x_train, y_train, x_test, y_test):
        y_train_pred = self.predict(x_train)
        y_pred = self.predict(x_test)
        residual_train = y_train - y_train_pred
        residual_test = y_test - y_pred
        return residual_train, residual_test, y_train_pred, y_pred
