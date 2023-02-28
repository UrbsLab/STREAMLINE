# Adding New Modeling Algorithms

New models can easily be added to STREAMLINE by creating a custom class 
and wrapping it as warped class derived form the STREAMLINE `BaseModel` with 
specific information such as name, plot colors and hyper parameters for the sweep.

An example wrapped code is given below. This is also given as file in the 
info directory of the github [here](https://github.com/UrbsLab/STREAMLINE/blob/dev/info/elastic_net.py)


```
from abc import ABC
from streamline.modeling.basemodel import BaseModel
from sklearn.linear_model import SGDClassifier as SGD


class ElasticNetClassifier(BaseModel, ABC):
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
```

This .py file can be kept in the `streamline/models` folders and it will be automatically picked up by the STREAMLINE 
pipeline dynamically

To make your own model make an arbitrarily named Class Derived form the `BaseModel` class in STREAMLINE.

The base class should be given it's own     
    
* `model_name`
* `small_name`
* `color`
* `param_grid`

And initialized with an sklearn compatible model class such as `SGD` here.

The init and super class init parameters should be the same as above.

An objective function should also be written for optuna such that all the parameters are suggested 
through an optuna trial. All the parameters should be suggested in a proper form using the most proper 
type of the variable and the proper function in the API documentation of 
Trial as described [here](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html)

Specifically we want to correctly parameters should be categorical, integer, or 
discrete or continuous in linear or log domain and their ranges.
The parameters that go in these functions should be what is defined in the param_grid variable.
