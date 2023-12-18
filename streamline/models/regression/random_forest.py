from abc import ABC
from streamline.modeling.submodels import RegressionModel
from sklearn.ensemble import RandomForestRegressor


class RFRegressor(RegressionModel, ABC):
    model_name = "Random Forest"
    small_name = "RF"
    color = "navy"

    def __init__(self, cv_folds=3, scoring_metric='explained_variance',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(RandomForestRegressor, "Random Forest", cv_folds,
                         scoring_metric, metric_direction, random_state, cv)
        self.param_grid = {'n_estimators': [10, 1000], 'max_depth': [1, 30], 'min_samples_split': [2, 50],
                           'min_samples_leaf': [1, 50], 'max_features': [None, 'auto', 'log2'], 'bootstrap': [True],
                           'oob_score': [False, True], 'random_state': [random_state, ]}
        self.small_name = "RF"
        self.color = "navy"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {'n_estimators': trial.suggest_int('n_estimators', self.param_grid['n_estimators'][0],
                                                         self.param_grid['n_estimators'][1]),
                       'max_depth': trial.suggest_int('max_depth', self.param_grid['max_depth'][0],
                                                      self.param_grid['max_depth'][1]),
                       'min_samples_split': trial.suggest_int('min_samples_split',
                                                              self.param_grid['min_samples_split'][0],
                                                              self.param_grid['min_samples_split'][1]),
                       'min_samples_leaf': trial.suggest_int('min_samples_leaf',
                                                             self.param_grid['min_samples_leaf'][0],
                                                             self.param_grid['min_samples_leaf'][1]),
                       'max_features': trial.suggest_categorical('max_features',
                                                                 self.param_grid['max_features']),
                       'bootstrap': trial.suggest_categorical('bootstrap', self.param_grid['bootstrap']),
                       'oob_score': trial.suggest_categorical('oob_score', self.param_grid['oob_score'])}

        mean_cv_score = self.hyper_eval()
        return mean_cv_score

    def residual_record(self, x_train, y_train, x_test, y_test):
        y_train_pred = self.predict(x_train)
        y_pred = self.predict(x_test)
        residual_train = y_train - y_train_pred
        residual_test = y_test - y_pred
        return residual_train, residual_test, y_train_pred, y_pred
