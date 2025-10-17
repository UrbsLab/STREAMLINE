from abc import ABC
from streamline.modeling.submodels import RegressionModel
from group_lasso import GroupLasso as GL


class GroupLasso(RegressionModel, ABC):
    model_name = "Group Lasso"
    small_name = "GL"
    color = "orange"

    def __init__(self, cv_folds=3, scoring_metric='explained_variance',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(GL, "Group Lasso", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = {'group_reg': [1e-3, 1], 'n_iter': [2000, 2500],
                           'scale_reg': ['group_size', 'none', 'inverse_group_size'], 'random_state': [random_state, ],
                           # 'subsampling_scheme': [0.1,0.9],
                           # 'frobenius_lipschitz': [True],
                           }
        self.small_name = "GL"
        self.color = "orange"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {'group_reg': trial.suggest_float('group_reg', self.param_grid['group_reg'][0],
                                                        self.param_grid['group_reg'][1]),
                       'n_iter': trial.suggest_int('n_iter', self.param_grid['n_iter'][0],
                                                   self.param_grid['n_iter'][1]),
                       'scale_reg': trial.suggest_categorical('scale_reg', self.param_grid['scale_reg']),
                       'random_state': trial.suggest_categorical('random_state', self.param_grid['random_state'])}

        mean_cv_score = self.hyper_eval()
        return mean_cv_score

    def residual_record(self, x_train, y_train, x_test, y_test):
        y_train_pred = self.predict(x_train)
        y_pred = self.predict(x_test)
        residual_train = y_train - y_train_pred
        residual_test = y_test - y_pred
        return residual_train, residual_test, y_train_pred, y_pred
