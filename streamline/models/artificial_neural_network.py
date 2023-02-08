from abc import ABC
from streamline.modeling.basemodel import BaseModel
from streamline.modeling.parameters import get_parameters
from sklearn.neural_network import MLPClassifier as MLP


class MLPClassifier(BaseModel, ABC):
    model_name = "Artificial Neural Network"
    small_name = "ANN"
    color = "red"

    def __init__(self, cv_folds=3, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(MLP, "Artificial Neural Network", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = get_parameters(self.model_name)
        self.param_grid['random_state'] = [random_state, ]
        self.small_name = "ANN"
        self.color = "red"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {'activation': trial.suggest_categorical('activation', self.param_grid['activation']),
                       'learning_rate': trial.suggest_categorical('learning_rate', self.param_grid['learning_rate']),
                       'momentum': trial.suggest_uniform('momentum', self.param_grid['momentum'][0],
                                                         self.param_grid['momentum'][1]),
                       'solver': trial.suggest_categorical('solver', self.param_grid['solver']),
                       'batch_size': trial.suggest_categorical('batch_size', self.param_grid['batch_size']),
                       'alpha': trial.suggest_loguniform('alpha', self.param_grid['alpha'][0],
                                                         self.param_grid['alpha'][1]),
                       'max_iter': trial.suggest_categorical('max_iter', self.param_grid['max_iter']),
                       'random_state': trial.suggest_categorical('random_state', self.param_grid['random_state'])}
        mean_cv_score = self.hyper_eval()
        return mean_cv_score
