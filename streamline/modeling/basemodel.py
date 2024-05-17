import copy
import logging
import optuna
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings

# Suppress specific warnings from sklearn, scipy, and optuna
warnings.filterwarnings(action='ignore', module='sklearn')
warnings.filterwarnings(action='ignore', module='scipy')
warnings.filterwarnings(action='ignore', module='optuna')
warnings.filterwarnings(action="ignore", category=ConvergenceWarning, module="sklearn")

class BaseModel:
    def __init__(self, model, model_name, cv_folds=3, scoring_metric='balanced_accuracy', 
                 metric_direction='maximize', random_state=None, cv=None, sampler=None, n_jobs=None):
        """
        Base Model Class for all Machine Learning Models.

        Args:
            model: The model to be used (e.g., a scikit-learn estimator).
            model_name: The name of the model.
            cv_folds: Number of cross-validation folds.
            scoring_metric: Metric used for scoring the model.
            metric_direction: Direction to optimize the scoring metric ('maximize' or 'minimize').
            random_state: Random state for reproducibility.
            cv: Custom cross-validation strategy.
            sampler: Sampler for hyperparameter optimization.
            n_jobs: Number of parallel jobs for cross-validation.
        """
        self.is_single = True  # Flag to check if the parameter grid has single values
        if model is not None:
            self.model = model()  # Initialize the model
        self.small_name = model_name.replace(" ", "_")  # Simplified model name without spaces
        self.model_name = model_name
        self.y_train = None  # Placeholder for training labels
        self.x_train = None  # Placeholder for training data
        self.param_grid = None  # Parameter grid for optimization
        self.params = None  # Optimized parameters
        self.random_state = random_state
        self.scoring_metric = scoring_metric
        self.metric_direction = metric_direction
        self.cv = cv if cv else StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        self.sampler = sampler if sampler else optuna.samplers.TPESampler(seed=self.random_state)
        self.study = None  # Optuna study for optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        """
        Objective function for Optuna optimization. This function needs to be overridden in derived classes.

        Args:
            trial: Optuna trial object.
            params: Additional parameters for the trial.
        """
        raise NotImplementedError

    @ignore_warnings(category=ConvergenceWarning)
    def optimize(self, x_train, y_train, n_trails, timeout, feature_names=None):
        """
        Optimize model hyperparameters using Optuna.

        Args:
            x_train: Training data.
            y_train: Training labels.
            n_trails: Number of Optuna trials.
            timeout: Maximum time for each Optuna trial.
            feature_names: Names of features (used in specific models).
        """
        self.x_train = x_train
        self.y_train = y_train
        
        # Check if the parameter grid contains single values
        for key, value in self.param_grid.items():
            if len(value) > 1 and key != 'expert_knowledge':
                self.is_single = False
                break

        if not self.is_single:
            # Perform hyperparameter optimization with Optuna
            self.study = optuna.create_study(direction=self.metric_direction, sampler=self.sampler)
            if self.model_name in ["Extreme Gradient Boosting", "Light Gradient Boosting"]:
                pos_inst = sum(y_train)
                neg_inst = len(y_train) - pos_inst
                class_weight = neg_inst / float(pos_inst)
                self.study.optimize(lambda trial: self.objective(trial, params={'class_weight': class_weight}),
                                    n_trials=n_trails, timeout=timeout, catch=(ValueError,))
            elif self.model_name == "Genetic Programming":
                self.study.optimize(lambda trial: self.objective(trial, params={'feature_names': feature_names}),
                                    n_trials=n_trails, timeout=timeout, catch=(ValueError,))
            else:
                self.study.optimize(lambda trial: self.objective(trial), n_trials=n_trails, timeout=timeout,
                                    catch=(ValueError,))

            # Log best trial results
            logging.info('Best trial:')
            best_trial = self.study.best_trial
            logging.info('  Value: ' + str(best_trial.value))
            logging.info('  Params: ')
            for key, value in best_trial.params.items():
                logging.info('    {}: {}'.format(key, value))
            
            if self.small_name == "ANN":
                # Special handling for ANN model parameters
                layers = []
                for j in range(best_trial.params['n_layers']):
                    layer_name = 'n_units_l' + str(j)
                    layers.append(best_trial.params[layer_name])
                    del best_trial.params[layer_name]
                best_trial.params['hidden_layer_sizes'] = tuple(layers)
                del best_trial.params['n_layers']
            
            # Set model parameters to the best trial's parameters
            self.params = best_trial.params
            self.model = copy.deepcopy(self.model).set_params(**best_trial.params)
        else:
            # If no optimization is needed, use the given parameters directly
            self.params = copy.deepcopy(self.param_grid)
            for key, value in self.param_grid.items():
                self.params[key] = value[0]
            self.model = copy.deepcopy(self.model).set_params(**self.params)

    def feature_importance(self):
        """
        Placeholder for a method to calculate feature importance. Needs to be overridden in derived classes.
        """
        raise NotImplementedError

    def hyper_eval(self):
        """
        Evaluate model performance with cross-validation.

        Returns:
            mean_cv_score: Mean cross-validation score.
        """
        logging.debug("Trial Parameters: " + str(self.params))
        logging.debug("Trial Metric: " + str(self.scoring_metric))
        try:
            model = copy.deepcopy(self.model).set_params(**self.params)
            mean_cv_score = cross_val_score(model, self.x_train, self.y_train,
                                            scoring=self.scoring_metric,
                                            cv=self.cv, n_jobs=self.n_jobs).mean()
        except Exception as e:
            logging.error("KeyError while copying model " + self.model_name)
            logging.error(str(e))
            model_class = self.model.__class__
            model = model_class(**self.params)
            mean_cv_score = cross_val_score(model, self.x_train, self.y_train,
                                            scoring=self.scoring_metric,
                                            cv=self.cv, n_jobs=self.n_jobs).mean()
        logging.debug("Trial Completed")
        logging.debug("Mean CV Score:" + str(mean_cv_score))
        return mean_cv_score

    def fit(self, x_train, y_train, n_trails, timeout, feature_names=None):
        """
        Fit the model to the training data after optimizing hyperparameters.

        Args:
            x_train: Training data.
            y_train: Training labels.
            n_trails: Number of Optuna trials.
            timeout: Maximum time for each Optuna trial.
            feature_names: Names of features (used in specific models).
        """
        self.optimize(x_train, y_train, n_trails, timeout, feature_names)
        self.model.fit(x_train, y_train)

    def predict(self, x_in):
        """
        Predict labels for the input data using the trained model.

        Args:
            x_in: Input data.

        Returns:
            y_pred: Predicted labels.
        """
        return self.model.predict(x_in)
