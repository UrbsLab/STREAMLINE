from abc import ABC
from streamline.modeling.submodels import RegressionModel
from streamline.modeling.parameters import get_parameters
from sklearn.ensemble import GradientBoostingRegressor


class SVR(RegressionModel, ABC):
    model_name = "GradBoost"
    small_name = "GB"
    color = "olive"

    def __init__(self, cv_folds=3, scoring_metric='explained_variance',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(GradientBoostingRegressor, "GradBoost", cv_folds, scoring_metric, metric_direction,
                         random_state, cv)
        self.param_grid = get_parameters(self.model_name, model_type="Regression")
        self.param_grid['random_state'] = [random_state, ]
        self.small_name = "GB"
        self.color = "olive"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {'learning_rate': trial.suggest_float('learning_rate', self.param_grid['learning_rate'][0],
                                                            self.param_grid['learning_rate'][1]),
                       'n_estimators': trial.suggest_int('n_estimators', self.param_grid['n_estimators'][0],
                                                         self.param_grid['n_estimators'][1]),
                       'min_samples_leaf': trial.suggest_int('min_samples_leaf', self.param_grid['min_samples_leaf'][0],
                                                             self.param_grid['min_samples_leaf'][1]),
                       'min_samples_split': trial.suggest_int('min_samples_split', self.param_grid['min_samples_split'][0],
                                                              self.param_grid['min_samples_split'][1]),
                       'max_depth': trial.suggest_int('max_depth', self.param_grid['max_depth'][0],
                                                      self.param_grid['max_depth'][1])}

        mean_cv_score = self.hyper_eval()
        return mean_cv_score

    def residual_record(self, x_train, y_train, x_test, y_test):
        y_train_pred = self.predict(x_train)
        y_pred = self.predict(x_test)
        residual_train = y_train - y_train_pred
        residual_test = y_test - y_pred
        return residual_train, residual_test, y_train_pred, y_pred
