from abc import ABC
from streamline.modeling.submodels import MulticlassClassificationModel
from streamline.modeling.parameters import get_parameters
from sklearn.linear_model import LogisticRegression as LogR


class LogisticRegression(MulticlassClassificationModel, ABC):
    model_name = "Logistic Regression"
    small_name = "LR"
    color = "dimgrey"

    def __init__(self, cv_folds=3, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None):
        super().__init__(LogR, "Logistic Regression", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = get_parameters(self.model_name)
        self.param_grid['random_state'] = [random_state, ]
        self.small_name = "LR"
        self.color = "dimgrey"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {
            'solver': trial.suggest_categorical('solver', self.param_grid['solver']),
            'C': trial.suggest_float('C', self.param_grid['C'][0], self.param_grid['C'][1], log=True),
            'class_weight': trial.suggest_categorical('class_weight', self.param_grid['class_weight']),
            'max_iter': trial.suggest_int('max_iter', self.param_grid['max_iter'][0],
                                          self.param_grid['max_iter'][1], log=True),
            'random_state': trial.suggest_categorical('random_state', self.param_grid['random_state'])}
        if self.params['solver'] == 'liblinear':
            self.params['penalty'] = trial.suggest_categorical('penalty', self.param_grid['penalty'])
            if self.params['penalty'] == 'l2':
                self.params['dual'] = trial.suggest_categorical('dual', self.param_grid['dual'])

        mean_cv_score = self.hyper_eval()
        # logging.debug("Trial Parameters" + str(self.params))
        # model = copy.deepcopy(self.model).set_params(**self.params)
        #
        # mean_cv_score = cross_val_score(model, self.x_train, self.y_train,
        #                                 scoring=self.scoring_metric,
        #                                 cv=self.cv, n_jobs=self.n_jobs).mean()
        # logging.debug("Trail Completed")
        return mean_cv_score
