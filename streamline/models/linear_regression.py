from sklearn import clone
from sklearn.model_selection import cross_val_score
from streamline.modeling.basemodel import BaseModel
from streamline.modeling.parameters import get_parameters
from sklearn.linear_model import LogisticRegression as LogR


class LogisticRegression(BaseModel):
    def __init__(self, cv_folds=5, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None):
        super().__init__(LogR, "Logistic Regression", cv_folds, scoring_metric, metric_direction, random_state, cv)
        self.param_grid = get_parameters(self.model_name)
        self.small_name = "LR"

    def objective(self, trial):
        self.params = {
            'solver': trial.suggest_categorical('solver', self.param_grid['solver']),
            'C': trial.suggest_loguniform('C', self.param_grid['C'][0], self.param_grid['C'][1]),
            'class_weight': trial.suggest_categorical('class_weight', self.param_grid['class_weight']),
            'max_iter': trial.suggest_loguniform('max_iter', self.param_grid['max_iter'][0],
                                                 self.param_grid['max_iter'][1]),
            'random_state': trial.suggest_categorical('random_state', self.param_grid['random_state'])}
        if self.params['solver'] == 'liblinear':
            self.params['penalty'] = trial.suggest_categorical('penalty', self.param_grid['penalty'])
            if self.params['penalty'] == 'l2':
                self.params['dual'] = trial.suggest_categorical('dual', self.param_grid['dual'])
        model = clone(self.model).set_params(**self.params)

        mean_cv_score = cross_val_score(model, self.x_train, self.y_train,
                                        scoring=self.scoring_metric,
                                        cv=self.cv, n_jobs=-1).mean()

        return mean_cv_score
