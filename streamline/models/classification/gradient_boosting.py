from abc import ABC
from streamline.modeling.submodels import ClassificationModel
from streamline.modeling.parameters import get_parameters
from sklearn.ensemble import GradientBoostingClassifier as GB
from xgboost import XGBClassifier as XGB
from lightgbm import LGBMClassifier as LGB
from catboost import CatBoostClassifier as CGB


class GBClassifier(ClassificationModel, ABC):
    model_name = "Gradient Boosting"
    small_name = "GB"
    color = "cornflowerblue"

    def __init__(self, cv_folds=3, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(GB, "Gradient Boosting", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = get_parameters(self.model_name)
        self.param_grid['random_state'] = [random_state, ]
        self.small_name = "GB"
        self.color = "cornflowerblue"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {'n_estimators': trial.suggest_int('n_estimators', self.param_grid['n_estimators'][0],
                                                         self.param_grid['n_estimators'][1]),
                       'loss': trial.suggest_categorical('loss', self.param_grid['loss']),
                       'learning_rate': trial.suggest_float('learning_rate', self.param_grid['learning_rate'][0],
                                                            self.param_grid['learning_rate'][1], log=True),
                       'min_samples_leaf': trial.suggest_int('min_samples_leaf', self.param_grid['min_samples_leaf'][0],
                                                             self.param_grid['min_samples_leaf'][1]),
                       'min_samples_split': trial.suggest_int('min_samples_split',
                                                              self.param_grid['min_samples_split'][0],
                                                              self.param_grid['min_samples_split'][1]),
                       'max_depth': trial.suggest_int('max_depth', self.param_grid['max_depth'][0],
                                                      self.param_grid['max_depth'][1]),
                       'random_state': trial.suggest_categorical('random_state', self.param_grid['random_state'])}

        mean_cv_score = self.hyper_eval()
        return mean_cv_score


class XGBClassifier(ClassificationModel, ABC):
    model_name = "Extreme Gradient Boosting"
    small_name = "XGB"
    color = "cyan"

    def __init__(self, cv_folds=3, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(XGB, "Extreme Gradient Boosting", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = get_parameters(self.model_name)
        self.param_grid['random_state'] = [random_state, ]
        self.small_name = "XGB"
        self.color = "cyan"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        class_weight = params['class_weight']
        param_grid = self.param_grid
        self.params = {'booster': trial.suggest_categorical('booster', param_grid['booster']),
                       'objective': trial.suggest_categorical('objective', param_grid['objective']),
                       'verbosity': trial.suggest_categorical('verbosity', param_grid['verbosity']),
                       'reg_lambda': trial.suggest_float('reg_lambda', param_grid['reg_lambda'][0],
                                                         param_grid['reg_lambda'][1], log=True),
                       'alpha': trial.suggest_float('alpha', param_grid['alpha'][0], param_grid['alpha'][1], log=True),
                       'eta': trial.suggest_float('eta', param_grid['eta'][0], param_grid['eta'][1], log=True),
                       'gamma': trial.suggest_float('gamma', param_grid['gamma'][0], param_grid['gamma'][1], log=True),
                       'max_depth': trial.suggest_int('max_depth', param_grid['max_depth'][0],
                                                      param_grid['max_depth'][1]),
                       'grow_policy': trial.suggest_categorical('grow_policy', param_grid['grow_policy']),
                       'n_estimators': trial.suggest_int('n_estimators', param_grid['n_estimators'][0],
                                                         param_grid['n_estimators'][1]),
                       'min_samples_split': trial.suggest_int('min_samples_split', param_grid['min_samples_split'][0],
                                                              param_grid['min_samples_split'][1]),
                       'min_samples_leaf': trial.suggest_int('min_samples_leaf', param_grid['min_samples_leaf'][0],
                                                             param_grid['min_samples_leaf'][1]),
                       'subsample': trial.suggest_uniform('subsample', param_grid['subsample'][0],
                                                          param_grid['subsample'][1]),
                       'min_child_weight': trial.suggest_float('min_child_weight',
                                                               param_grid['min_child_weight'][0],
                                                               param_grid['min_child_weight'][1], log=True),
                       'colsample_bytree': trial.suggest_uniform('colsample_bytree', param_grid['colsample_bytree'][0],
                                                                 param_grid['colsample_bytree'][1]),
                       'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [1.0, class_weight]),
                       'nthread': trial.suggest_categorical('nthread', param_grid['nthread']),
                       'random_state': trial.suggest_categorical('random_state', param_grid['random_state']), }

        mean_cv_score = self.hyper_eval()
        return mean_cv_score


class LGBClassifier(ClassificationModel, ABC):
    model_name = "Light Gradient Boosting"
    small_name = "LGB"
    color = "pink"

    def __init__(self, cv_folds=3, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(LGB, "Light Gradient Boosting", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = get_parameters(self.model_name)
        self.param_grid['random_state'] = [random_state, ]
        self.small_name = "LGB"
        self.color = "pink"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        class_weight = params['class_weight']
        param_grid = self.param_grid
        self.params = {'objective': trial.suggest_categorical('objective', param_grid['objective']),
                       'metric': trial.suggest_categorical('metric', param_grid['metric']),
                       'verbosity': trial.suggest_categorical('verbosity', param_grid['verbosity']),
                       'boosting_type': trial.suggest_categorical('boosting_type', param_grid['boosting_type']),
                       'num_leaves': trial.suggest_int('num_leaves', param_grid['num_leaves'][0],
                                                       param_grid['num_leaves'][1]),
                       'max_depth': trial.suggest_int('max_depth', param_grid['max_depth'][0],
                                                      param_grid['max_depth'][1]),
                       'reg_alpha': trial.suggest_float('reg_alpha', param_grid['reg_alpha'][0],
                                                        param_grid['reg_alpha'][1], log=True),
                       'reg_lambda': trial.suggest_float('reg_lambda', param_grid['reg_lambda'][0],
                                                         param_grid['reg_lambda'][1], log=True),
                       'colsample_bytree': trial.suggest_uniform('colsample_bytree', param_grid['colsample_bytree'][0],
                                                                 param_grid['colsample_bytree'][1]),
                       'subsample': trial.suggest_uniform('subsample', param_grid['subsample'][0],
                                                          param_grid['subsample'][1]),
                       'subsample_freq': trial.suggest_int('subsample_freq', param_grid['subsample_freq'][0],
                                                           param_grid['subsample_freq'][1]),
                       'min_child_samples': trial.suggest_int('min_child_samples', param_grid['min_child_samples'][0],
                                                              param_grid['min_child_samples'][1]),
                       'n_estimators': trial.suggest_int('n_estimators', param_grid['n_estimators'][0],
                                                         param_grid['n_estimators'][1]),
                       'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [1.0, class_weight]),
                       'random_state': trial.suggest_categorical('random_state', param_grid['random_state']),
                       }
        # print(self.model.get_params())
        mean_cv_score = self.hyper_eval()
        return mean_cv_score


class CGBClassifier(ClassificationModel, ABC):
    model_name = "Category Gradient Boosting"
    small_name = "CGB"
    color = "magenta"

    def __init__(self, cv_folds=3, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(CGB, "Category Gradient Boosting", cv_folds, scoring_metric, metric_direction, random_state,
                         cv)
        self.param_grid = get_parameters(self.model_name)
        self.param_grid['random_state'] = [random_state, ]
        self.small_name = "CGB"
        self.color = "magenta"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {'learning_rate': trial.suggest_float('learning_rate', self.param_grid['learning_rate'][0],
                                                            self.param_grid['learning_rate'][1], log=True),
                       'iterations': trial.suggest_int('iterations', self.param_grid['iterations'][0],
                                                       self.param_grid['iterations'][1]),
                       'depth': trial.suggest_int('depth', self.param_grid['depth'][0], self.param_grid['depth'][1]),
                       'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', self.param_grid['l2_leaf_reg'][0],
                                                        self.param_grid['l2_leaf_reg'][1]),
                       'loss_function': trial.suggest_categorical('loss_function', self.param_grid['loss_function']),
                       'random_state': trial.suggest_categorical('random_state', self.param_grid['random_state']),
                       'verbose': trial.suggest_categorical('verbose', self.param_grid['verbose']),
                       }

        mean_cv_score = self.hyper_eval()
        return mean_cv_score
