from abc import ABC
from streamline.p6_modeling.utils.submodels import MulticlassClassificationModel
from skheros.heros import HEROS


class HEROSMulticlassClassifier(MulticlassClassificationModel, ABC):
    model_name = "HEROS"
    small_name = "HEROS"
    color = "darkgreen"
    subsampling_allowed = True

    def __init__(self, cv_folds=3, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None,
                 iterations=100000, pop_size=1000, model_iterations=500,
                 model_pop_size=100, nu=1):
        super().__init__(HEROS, "HEROS", cv_folds, scoring_metric, metric_direction, random_state, cv)

        self.param_grid = {
            'iterations': [100000, 200000, 500000] if iterations is None else [iterations],
            'pop_size': [1000, 2000, 5000] if pop_size is None else [pop_size],
            'model_iterations': [500, 1000] if model_iterations is None else [model_iterations],
            'model_pop_size': [100, 200] if model_pop_size is None else [model_pop_size],
            'nu': [1, 10] if nu is None else [nu],
        }

        self.param_grid['random_state'] = [random_state]

        self.small_name = "HEROS"
        self.color = "darkgreen"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        # HEROS accepts multiclass labels naturally through sklearn API
        self.params = {
            'iterations': trial.suggest_categorical(
                'iterations', self.param_grid['iterations']
            ),
            'pop_size': trial.suggest_categorical(
                'pop_size', self.param_grid['pop_size']
            ),
            'model_iterations': trial.suggest_categorical(
                'model_iterations', self.param_grid['model_iterations']
            ),
            'model_pop_size': trial.suggest_categorical(
                'model_pop_size', self.param_grid['model_pop_size']
            ),
            'nu': trial.suggest_categorical(
                'nu', self.param_grid['nu']
            ),
            'random_state': trial.suggest_categorical(
                'random_state', self.param_grid['random_state']
            ),
        }

        mean_cv_score = self.hyper_eval()
        return mean_cv_score
