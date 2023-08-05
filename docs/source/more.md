# Doing More with STREAMLINE
Before, or after running STREAMLINE, there are a number of things a user can do to get even more out of this framework and it's main phases.

***
## Useful Notebooks
Included in the STREAMLINE repository is the folder `UsefulNotebooks` containing a variety of Jupyter Notebooks, each designed to work with an 'experiment' folder containing the output of a STREAMLINE run. As an overview, these notebooks are designed to:
1. Regenerate key plots based on user specifications
2. Reporting model prediction probabilities for testing and replication dataset instances
3. Generate additional figures, including:
    * A feature imporance rank heatmap
    * Model vizualizations for decision tree and genetic programming models
4. Examining the impact of using decision thresholds other than the default 0.5. 

Below we detail what each of these notebooks do:
* `AccessingPickledMetricsAndPredictionProbabilities`


* *Note: Users can run these notebooks more or less 'as-is' or they may wish to modify their underlying code further.*

[Under Construction]

We have assempled a variety of 'useful' Jupyter Notebooks
designed to operate on an experiment folder allowing users to do even more
with the pipeline output. Examples include:
1. Accessing prediction probabilities.
2. Regenerating figures to user-specifications.
3. Trying out the effect of different prediction thresholds on selected
   models with an interactive slider.
4. Re-evaluating models when applying a new prediction threshold.
5. Generating an interactive model feature importance ranking visualization across
   all ML algorithms.
6. Generating an interpretable model vizualization for either decision tree or genetic programming models.


***
## Updating Modeling Algorithm Hyperparameter Options
The hard-coded range of hyperparameter options and their value options/ranges for each algorithm can be found within `streamline/modeling/parameters.py`. 
Code-savy users can adjust these value option/ranges for each ML algorithm if desired. However if you do so, and publish results of running STREAMLINE we stongly recommend indicating this or any other code changes for reproducibility. 


***
## Adding New Modeling Algorithms

New models can easily be added to STREAMLINE by creating a custom class
and wrapping it as warped class derived form the STREAMLINE `BaseModel` with
specific information such as name, plot colors and hyper parameters for the sweep.

An example wrapped code is given below. This is also given as file in the
info directory of the github [here](https://github.com/UrbsLab/STREAMLINE/blob/main/docs/source/elastic_net.py)


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
