from abc import ABC
from streamline.modeling.basemodel import BaseModel
from streamline.modeling.parameters import get_parameters
from sklearn.ensemble import GradientBoostingClassifier as GB
from xgboost import XGBClassifier as XGB
from lightgbm import LGBMClassifier as LGB
from catboost import CatBoostClassifier as CGB


class GBClassifier(BaseModel, ABC):
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
                       'learning_rate': trial.suggest_loguniform('learning_rate', self.param_grid['learning_rate'][0],
                                                                 self.param_grid['learning_rate'][1]),
                       'min_samples_leaf': trial.suggest_int('min_samples_leaf', self.param_grid['min_samples_leaf'][0],
                                                             self.param_grid['min_samples_leaf'][1]),
                       'min_samples_split': trial.suggest_int('min_samples_split',
                                                              self.param_grid['min_samples_split'][0],
                                                              self.param_grid['min_samples_split'][1]),
                       'max_depth': trial.suggest_int('max_depth', self.param_grid['max_depth'][0],
                                                      self.param_grid['max_depth'][1]),
                       'random_state': trial.suggest_categorical('random_state', self.param_grid['random_state'])}

        mean_cv_score = self.hypereval(trial)
        return mean_cv_score


class XGBClassifier(BaseModel, ABC):
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
                       'reg_lambda': trial.suggest_loguniform('reg_lambda', param_grid['reg_lambda'][0],
                                                              param_grid['reg_lambda'][1]),
                       'alpha': trial.suggest_loguniform('alpha', param_grid['alpha'][0], param_grid['alpha'][1]),
                       'eta': trial.suggest_loguniform('eta', param_grid['eta'][0], param_grid['eta'][1]),
                       'gamma': trial.suggest_loguniform('gamma', param_grid['gamma'][0], param_grid['gamma'][1]),
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
                       'min_child_weight': trial.suggest_loguniform('min_child_weight',
                                                                    param_grid['min_child_weight'][0],
                                                                    param_grid['min_child_weight'][1]),
                       'colsample_bytree': trial.suggest_uniform('colsample_bytree', param_grid['colsample_bytree'][0],
                                                                 param_grid['colsample_bytree'][1]),
                       'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [1.0, class_weight]),
                       'nthread': trial.suggest_categorical('nthread', param_grid['nthread']),
                       'random_state': trial.suggest_categorical('seed', param_grid['seed'])}

        mean_cv_score = self.hypereval(trial)
        return mean_cv_score


class LGBClassifier(BaseModel, ABC):
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
                       'lambda_l1': trial.suggest_loguniform('lambda_l1', param_grid['lambda_l1'][0],
                                                             param_grid['lambda_l1'][1]),
                       'lambda_l2': trial.suggest_loguniform('lambda_l2', param_grid['lambda_l2'][0],
                                                             param_grid['lambda_l2'][1]),
                       'feature_fraction': trial.suggest_uniform('feature_fraction', param_grid['feature_fraction'][0],
                                                                 param_grid['feature_fraction'][1]),
                       'bagging_fraction': trial.suggest_uniform('bagging_fraction', param_grid['bagging_fraction'][0],
                                                                 param_grid['bagging_fraction'][1]),
                       'bagging_freq': trial.suggest_int('bagging_freq', param_grid['bagging_freq'][0],
                                                         param_grid['bagging_freq'][1]),
                       'min_child_samples': trial.suggest_int('min_child_samples', param_grid['min_child_samples'][0],
                                                              param_grid['min_child_samples'][1]),
                       'n_estimators': trial.suggest_int('n_estimators', param_grid['n_estimators'][0],
                                                         param_grid['n_estimators'][1]),
                       'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [1.0, class_weight]),
                       'num_threads': trial.suggest_categorical('num_threads', param_grid['num_threads']),
                       'random_state': trial.suggest_categorical('seed', param_grid['seed'])}

        mean_cv_score = self.hypereval(trial)
        return mean_cv_score


class CGBClassifier(BaseModel, ABC):
    def __init__(self, cv_folds=3, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(CGB, "Category Gradient Boosting", cv_folds, scoring_metric, metric_direction, random_state,
                         cv)
        self.param_grid = get_parameters(self.model_name)
        self.param_grid['random_state'] = [random_state, ]
        self.small_name = "LGB"
        self.color = "magenta"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {'learning_rate': trial.suggest_loguniform('learning_rate', self.param_grid['learning_rate'][0],
                                                                 self.param_grid['learning_rate'][1]),
                       'iterations': trial.suggest_int('iterations', self.param_grid['iterations'][0],
                                                       self.param_grid['iterations'][1]),
                       'depth': trial.suggest_int('depth', self.param_grid['depth'][0], self.param_grid['depth'][1]),
                       'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', self.param_grid['l2_leaf_reg'][0],
                                                        self.param_grid['l2_leaf_reg'][1]),
                       'loss_function': trial.suggest_categorical('loss_function', self.param_grid['loss_function']),
                       'random_seed': trial.suggest_categorical('random_seed', self.param_grid['random_seed'])}

        mean_cv_score = self.hypereval(trial)
        return mean_cv_score
