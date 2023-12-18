from abc import ABC
from streamline.modeling.submodels import RegressionModel
from sklearn.ensemble import AdaBoostRegressor as ABR


class AdaBoostRegressor(RegressionModel, ABC):
    model_name = "AdaBoost"
    small_name = "AB"
    color = "teal"

    def __init__(self, cv_folds=3, scoring_metric='explained_variance',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(ABR, "AdaBoost", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = {'n_estimators': [10, 1000], 'learning_rate': [.0001, 0.3],
                           'loss': ['linear', 'square', 'exponential'], 'random_state': [random_state, ]}
        self.small_name = "AB"
        self.color = "teal"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {'n_estimators': trial.suggest_int('n_estimators', self.param_grid['n_estimators'][0],
                                                         self.param_grid['n_estimators'][1]),
                       'learning_rate': trial.suggest_float('learning_rate', self.param_grid['learning_rate'][0],
                                                            self.param_grid['learning_rate'][1]),
                       'loss': trial.suggest_categorical('loss', self.param_grid['loss'])}

        mean_cv_score = self.hyper_eval()
        return mean_cv_score

    def residual_record(self, x_train, y_train, x_test, y_test):
        y_train_pred = self.predict(x_train)
        y_pred = self.predict(x_test)
        residual_train = y_train - y_train_pred
        residual_test = y_test - y_pred
        return residual_train, residual_test, y_train_pred, y_pred
