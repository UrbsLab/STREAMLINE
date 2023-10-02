import copy
import logging
import optuna
from sklearn import metrics
from sklearn.metrics import auc
from streamline.utils.evaluation import class_eval
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings(action='ignore', module='sklearn')
warnings.filterwarnings(action='ignore', module='scipy')
warnings.filterwarnings(action='ignore', module='optuna')
warnings.filterwarnings(action="ignore", category=ConvergenceWarning, module="sklearn")


class BaseModel:
    def __init__(self, model, model_name,
                 cv_folds=3, scoring_metric='balanced_accuracy', metric_direction='maximize',
                 random_state=None, cv=None, sampler=None, n_jobs=None):
        """
        Base Model Class for all ML Models

        Args:
            model:
            model_name:
            cv_folds:
            scoring_metric:
            metric_direction:
            random_state:
            cv:
            sampler:
            n_jobs:
        """
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

    def objective(self, trial, params=None):
        """
        Unimplemented objective function stub, needs to be overridden
        Args:
            trial: optuna trial object
            params: dict of optional params or None
        """
        raise NotImplementedError

    @ignore_warnings(category=ConvergenceWarning)
    def optimize(self, x_train, y_train, n_trails, timeout, feature_names=None):
        """
        Common model optimization function

        Args:
            x_train: train data
            y_train: label data
            n_trails: number of optuna trials
            timeout: maximum time for optuna trial timeout
            feature_names: header/name of features

        """
        self.x_train = x_train
        self.y_train = y_train
        for key, value in self.param_grid.items():
            if len(value) > 1 and key != 'expert_knowledge':
                self.is_single = False
                break

        if not self.is_single:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.study = optuna.create_study(direction=self.metric_direction, sampler=self.sampler)
            if self.model_name in ["Extreme Gradient Boosting", "Light Gradient Boosting"]:
                pos_inst = sum(y_train)
                neg_inst = len(y_train) - pos_inst
                class_weight = neg_inst / float(pos_inst)
                self.study.optimize(lambda trial: self.objective(trial, params={'class_weight': class_weight}),
                                    n_trials=n_trails, timeout=timeout,
                                    catch=(ValueError,))
            elif self.model_name == "Genetic Programming":
                self.study.optimize(lambda trial: self.objective(trial, params={'feature_names': feature_names}),
                                    n_trials=n_trails, timeout=timeout,
                                    catch=(ValueError,))
            else:
                self.study.optimize(lambda trial: self.objective(trial), n_trials=n_trails, timeout=timeout,
                                    catch=(ValueError,))

            logging.info('Best trial:')
            best_trial = self.study.best_trial
            logging.info('  Value: ' + str(best_trial.value))
            logging.info('  Params: ')
            for key, value in best_trial.params.items():
                logging.info('    {}: {}'.format(key, value))
            if self.small_name == "ANN":
                # Handle special parameter requirement for ANN
                layers = []
                for j in range(best_trial.params['n_layers']):
                    layer_name = 'n_units_l' + str(j)
                    layers.append(best_trial.params[layer_name])
                    del best_trial.params[layer_name]
                best_trial.params['hidden_layer_sizes'] = tuple(layers)
                del best_trial.params['n_layers']
            # Specify model with optimized hyperparameters
            # Export final model hyperparamters to csv file
            self.params = best_trial.params
            self.model = copy.deepcopy(self.model).set_params(**best_trial.params)
        else:
            self.params = copy.deepcopy(self.param_grid)
            for key, value in self.param_grid.items():
                self.params[key] = value[0]
            self.model = copy.deepcopy(self.model).set_params(**self.params)

    def feature_importance(self):
        """
        Unimplemented feature importance function stub
        """
        raise NotImplementedError

    def hyper_eval(self):
        """
        Hyper eval for objective function
        Returns: Returns hyper eval for objective function
        """
        logging.debug("Trial Parameters" + str(self.params))
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
        logging.debug("Trail Completed")
        return mean_cv_score

    def model_evaluation(self, x_test, y_test):
        """
        Runs commands to gather all evaluations for later summaries and plots.
        """
        # Prediction evaluation
        y_pred = self.model.predict(x_test)
        metric_list = class_eval(y_test, y_pred)
        # Determine probabilities of class predictions for each test instance
        # (this will be used much later in calculating an ROC curve)
        probas_ = self.model.predict_proba(x_test)
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        # Compute Precision/Recall curve and AUC
        prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
        prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
        prec_rec_auc = auc(recall, prec)
        ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])
        return metric_list, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_

    def fit(self, x_train, y_train, n_trails, timeout, feature_names=None):
        """
        Caller function to optimize
        """
        self.optimize(x_train, y_train, n_trails, timeout, feature_names)
        self.model.fit(x_train, y_train)

    def predict(self, x_in):
        """
        Function to predict with trained model
        Args:
            x_in: input data

        Returns: predictions y_pred

        """
        return self.model.predict(x_in)
