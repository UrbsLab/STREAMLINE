from abc import ABC
from streamline.modeling.submodels import ClassificationModel
from sklearn.linear_model import SGDClassifier as SGD


class ElasticNetClassifier(ClassificationModel, ABC):
    model_name = "Elastic Net"
    small_name = "EN"
    color = "aquamarine"

    def __init__(self, cv_folds=3, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(SGD, "Elastic Net", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = {'penalty': ['elasticnet'], 'loss': ['log_loss', 'modified_huber'], 'alpha': [0.04, 0.05],
                           'max_iter': [1000, 2000], 'l1_ratio': [0.001, 0.1], 'class_weight': [None, 'balanced'],
                           'random_state': [random_state, ]}
        self.small_name = "EN"
        self.color = "aquamarine"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {'penalty': trial.suggest_categorical('penalty', self.param_grid['penalty']),
                       'loss': trial.suggest_categorical('loss', self.param_grid['loss']),
                       'alpha': trial.suggest_float('alpha', self.param_grid['alpha'][0],
                                                    self.param_grid['l1_ratio'][1]),
                       'max_iter': trial.suggest_int('max_iter', self.param_grid['max_iter'][0],
                                                     self.param_grid['max_iter'][1]),
                       'l1_ratio': trial.suggest_float('l1_ratio', self.param_grid['l1_ratio'][0],
                                                       self.param_grid['l1_ratio'][1]),
                       'class_weight': trial.suggest_categorical('class_weight', self.param_grid['class_weight']),
                       'random_state': trial.suggest_categorical('random_state', self.param_grid['random_state'])}

        mean_cv_score = self.hyper_eval()
        return mean_cv_score
