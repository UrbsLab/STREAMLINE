import copy
import logging
import warnings
import optuna

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings(action='ignore', module='sklearn')
warnings.filterwarnings(action='ignore', module='scipy')
warnings.filterwarnings(action='ignore', module='optuna')
warnings.filterwarnings(action="ignore", category=ConvergenceWarning, module="sklearn")


class BaseModel:
    """
    Phase-6 BaseModel with:
      • Optuna hyper-eval API (unchanged)
      • Built-in optional probability calibration (CalibratedClassifierCV)
      • Default model_evaluation() that returns exactly what ModelJob expects
        - Regression: returns a metrics dict
        - Binary/Multiclass: returns (metrics, fpr, tpr, roc_auc, prec, recall, pr_auc, ave_prec, probs)
    Subclasses must set:
      - self.model_type in {"Binary", "Multiclass", "Regression"}
      - self.param_grid (dict of lists)
      - implement objective(trial, params=None)
      - call super().__init__(model=<sk_estimator_class>, model_name=<str>, ...)
    """

    def __init__(
        self,
        model,
        model_name,
        cv_folds=3,
        scoring_metric='balanced_accuracy',
        metric_direction='maximize',
        random_state=None,
        cv=None,
        sampler=None,
        n_jobs=None,
        # NEW: calibration knobs (classification only)
        # Don't need this because we have calibration in ModelJob now
        # calibrate: bool = False,
        # calibrate_method: str = "sigmoid",   # "sigmoid" | "isotonic"
        # calibrate_cv: int = 5,
    ):
        self.is_single = True
        if model is not None:
            self.model = model()
        self.small_name = model_name.replace(" ", "_")
        self.model_name = model_name
        self.y_train = None
        self.x_train = None
        self.param_grid = None
        self.params = None
        self.random_state = random_state
        self.scoring_metric = scoring_metric
        self.metric_direction = metric_direction
        if cv is None:
            self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        else:
            self.cv = cv

        if sampler is None:
            self.sampler = optuna.samplers.TPESampler(seed=self.random_state)
        else:
            self.sampler = sampler
        self.study = None
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.n_jobs = n_jobs

        # Calibration config
        # self.calibrate = bool(calibrate)
        # self.calibrate_method = calibrate_method
        # self.calibrate_cv = calibrate_cv

        # expected from subclass: self.model_type in {"Binary","Multiclass","Regression"}

    # ----- to be implemented by subclasses -----
    def objective(self, trial, params=None):
        raise NotImplementedError

    # ----- hyper-optimization -----
    @ignore_warnings(category=ConvergenceWarning)
    def optimize(self, x_train, y_train, n_trails, timeout, feature_names=None):
        self.x_train = x_train
        self.y_train = y_train
        for key, value in self.param_grid.items():
            if len(value) > 1 and key != 'expert_knowledge':
                self.is_single = False
                break

        if not self.is_single:
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
                self.study.optimize(lambda trial: self.objective(trial),
                                    n_trials=n_trails, timeout=timeout, catch=(ValueError,))

            logging.info('Best trial:')
            best_trial = self.study.best_trial
            logging.info('  Value: ' + str(best_trial.value))
            logging.info('  Params: ')
            for key, value in best_trial.params.items():
                logging.info('    {}: {}'.format(key, value))
            if self.small_name == "ANN":
                layers = []
                for j in range(best_trial.params['n_layers']):
                    layer_name = 'n_units_l' + str(j)
                    layers.append(best_trial.params[layer_name])
                    del best_trial.params[layer_name]
                best_trial.params['hidden_layer_sizes'] = tuple(layers)
                del best_trial.params['n_layers']
            self.params = best_trial.params
            self.model = copy.deepcopy(self.model).set_params(**best_trial.params)
        else:
            self.params = copy.deepcopy(self.param_grid)
            for key, value in self.param_grid.items():
                self.params[key] = value[0]
            self.model = copy.deepcopy(self.model).set_params(**self.params)

    # ----- shared utilities -----
    def hyper_eval(self):
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
            if self.scoring_metric.startswith('mean_'):
                cv_scoring_metric = 'neg_'+self.scoring_metric
            else:
                cv_scoring_metric = self.scoring_metric
            mean_cv_score = cross_val_score(model, self.x_train, self.y_train,
                                            scoring=cv_scoring_metric,
                                            cv=self.cv, n_jobs=self.n_jobs).mean()
        logging.debug("Trail Completed")
        logging.debug("Mean CV Score:" + str(mean_cv_score))
        return mean_cv_score

    def fit(self, x_train, y_train, n_trails, timeout, feature_names=None):
        """
        Optimize → fit → (optional) calibrate for classifiers
        """
        self.optimize(x_train, y_train, n_trails, timeout, feature_names)
        self.model.fit(x_train, y_train)

        # Optional probability calibration for classification models
        # Don't need this because we have calibration in ModelJob now
        # if self.calibrate and getattr(self, "model_type", None) in {"Binary", "Multiclass"}:
        #     try:
        #         cal = CalibratedClassifierCV(
        #             estimator=self.model,
        #             method=self.calibrate_method,
        #             cv=self.calibrate_cv
        #         )
        #         cal.fit(x_train, y_train)
        #         self.model = cal
        #         logging.info(f"Calibrated {self.small_name} with {self.calibrate_method} (cv={self.calibrate_cv})")
        #     except Exception as e:
        #         logging.warning(f"Calibration failed for {self.small_name}: {e}")

    def predict(self, x_in):
        return self.model.predict(x_in)

    def predict_proba(self, x_in):
        proba = getattr(self.model, "predict_proba", None)
        if proba is None:
            return None
        return proba(x_in)