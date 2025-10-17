from abc import ABC
from streamline.modeling.submodels import BinaryClassificationModel
from sklearn.neural_network import MLPClassifier as MLP


class MLPClassifier(BinaryClassificationModel, ABC):
    model_name = "Artificial Neural Network"
    small_name = "ANN"
    color = "red"

    def __init__(self, cv_folds=3, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(MLP, "Artificial Neural Network", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = {'n_layers': [1, 3], 'layer_size': [1, 100],
                           'activation': ['identity', 'logistic', 'tanh', 'relu'],
                           'learning_rate': ['constant', 'invscaling', 'adaptive'], 'momentum': [0.1, 0.9],
                           'solver': ['sgd', 'adam'], 'batch_size': ['auto'], 'alpha': [0.0001, 0.05],
                           'max_iter': [200], 'random_state': [random_state, ]}
        self.small_name = "ANN"
        self.color = "red"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {'activation': trial.suggest_categorical('activation', self.param_grid['activation']),
                       'learning_rate': trial.suggest_categorical('learning_rate', self.param_grid['learning_rate']),
                       'momentum': trial.suggest_float('momentum', self.param_grid['momentum'][0],
                                                       self.param_grid['momentum'][1]),
                       'solver': trial.suggest_categorical('solver', self.param_grid['solver']),
                       'batch_size': trial.suggest_categorical('batch_size', self.param_grid['batch_size']),
                       'alpha': trial.suggest_float('alpha', self.param_grid['alpha'][0],
                                                    self.param_grid['alpha'][1], log=True),
                       'max_iter': trial.suggest_categorical('max_iter', self.param_grid['max_iter']),
                       'random_state': trial.suggest_categorical('random_state', self.param_grid['random_state'])}
        n_layers = trial.suggest_int('n_layers', self.param_grid['n_layers'][0], self.param_grid['n_layers'][1])
        layers = []
        for i in range(n_layers):
            layers.append(
                trial.suggest_int('n_units_l{}'.format(i), self.param_grid['layer_size'][0],
                                  self.param_grid['layer_size'][1]))
            self.params['hidden_layer_sizes'] = tuple(layers)
        mean_cv_score = self.hyper_eval()
        return mean_cv_score
