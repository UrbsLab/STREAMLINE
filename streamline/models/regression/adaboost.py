from abc import ABC
from streamline.modeling.submodels import RegressionModel
from streamline.modeling.parameters import get_parameters
from sklearn.ensemble import AdaBoostRegressor


class SVR(RegressionModel, ABC):
    model_name = "AdaBoost"
    small_name = "AB"
    color = "teal"

    def __init__(self, cv_folds=3, scoring_metric='explained_variance',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(AdaBoostRegressor, "AdaBoost", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = get_parameters(self.model_name, model_type="Regression")
        self.param_grid['random_state'] = [random_state, ]
        self.small_name = "AB"
        self.color = "teal"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {'n_estimators': [10, 1000], 'learning_rate': [.0001, 0.3],
                       'loss': ['linear', 'square', 'exponential']}


        mean_cv_score = self.hyper_eval()
        return mean_cv_score

    def residual_record(self, x_train, y_train, x_test, y_test):
        y_train_pred = self.predict(x_train)
        y_pred = self.predict(x_test)
        residual_train = y_train - y_train_pred
        residual_test = y_test - y_pred
        return residual_train, residual_test, y_train_pred, y_pred
