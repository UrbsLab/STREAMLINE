from abc import ABC
from streamline.modeling.submodels import MulticlassClassificationModel
from gplearn.genetic import SymbolicClassifier as GP


class GPClassifier(MulticlassClassificationModel, ABC):
    model_name = "Genetic Programming"
    small_name = "GP"
    color = "purple"

    def __init__(self, cv_folds=3, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(GP, "Genetic Programming", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = {'population_size': [100, 1000], 'generations': [10, 500], 'tournament_size': [3, 50],
                           'init_method': ['grow', 'full', 'half and half'],
                           'function_set': [['add', 'sub', 'mul', 'div'],
                                            ['add', 'sub', 'mul', 'div', 'sqrt', 'log',
                                             'abs', 'neg', 'inv', 'max', 'min'],
                                            ['add', 'sub', 'mul', 'div', 'sqrt', 'log',
                                             'abs', 'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan']],
                           'parsimony_coefficient': [0.001, 0.01], 'low_memory': [True],
                           'random_state': [random_state, ]}
        self.small_name = "GP"
        self.color = "purple"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        feature_names = params['feature_names']
        self.params = {'population_size': trial.suggest_int('population_size', self.param_grid['population_size'][0],
                                                            self.param_grid['population_size'][1]),
                       'generations': trial.suggest_int('generations', self.param_grid['generations'][0],
                                                        self.param_grid['generations'][1]),
                       'tournament_size': trial.suggest_int('tournament_size', self.param_grid['tournament_size'][0],
                                                            self.param_grid['tournament_size'][1]),
                       'function_set': trial.suggest_categorical('function_set', self.param_grid['function_set']),
                       'init_method': trial.suggest_categorical('init_method', self.param_grid['init_method']),
                       'parsimony_coefficient': trial.suggest_float('parsimony_coefficient',
                                                                    self.param_grid['parsimony_coefficient'][0],
                                                                    self.param_grid['parsimony_coefficient'][1]),
                       'feature_names': trial.suggest_categorical('feature_names', [feature_names]),
                       'low_memory': trial.suggest_categorical('low_memory', self.param_grid['low_memory']),
                       'random_state': trial.suggest_categorical('random_state', self.param_grid['random_state'])}
        mean_cv_score = self.hyper_eval()
        return mean_cv_score
