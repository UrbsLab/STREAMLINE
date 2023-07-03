from abc import ABC

import optuna
from sklearn import metrics
from sklearn.metrics import auc
from streamline.modeling.basemodel import BaseModel
from streamline.utils.evaluation import class_eval
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings(action='ignore', module='sklearn')
warnings.filterwarnings(action='ignore', module='scipy')
warnings.filterwarnings(action='ignore', module='optuna')
warnings.filterwarnings(action="ignore", category=ConvergenceWarning, module="sklearn")


class BinaryClassificationModel(BaseModel, ABC):
    def __init__(self, model, model_name, cv_folds=3,
                 scoring_metric='balanced_accuracy', metric_direction='maximize',
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
        super().__init__(model, model_name, cv_folds, scoring_metric,
                         metric_direction, random_state, cv, sampler,
                         n_jobs)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.model_type = "BinaryClassification"

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

class MulticlassClassificationModel(BaseModel, ABC):
    def __init__(self, model, model_name, cv_folds=3,
                 scoring_metric='balanced_accuracy', metric_direction='maximize',
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
        super().__init__(model, model_name, cv_folds, scoring_metric,
                         metric_direction, random_state, cv, sampler,
                         n_jobs)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.model_type = "MulticlassClassification"

    def model_evaluation(self, x_test, y_test):
        """
        Runs commands to gather all evaluations for later summaries and plots.
        """
        pass
