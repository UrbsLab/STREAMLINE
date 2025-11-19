from abc import ABC
import numpy as np
import optuna
import pandas as pd
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.metrics import (
    auc, max_error, mean_absolute_error, mean_squared_error,
    median_absolute_error, explained_variance_score
)
from sklearn.model_selection import KFold
from sklearn.exceptions import ConvergenceWarning
import warnings

# Phase-6 BaseModel (with calibration + default evaluations)
from .basemodel import BaseModel

# Optional: if you still want the classic classification metric dict
# (kept here for backward compatibility with your reporting stack)
try:
    from streamline.p6_modeling.utils.evaluation import class_eval
except Exception:
    class_eval = None  # fallback: we'll use BaseModel's defaults if missing

warnings.filterwarnings(action='ignore', module='sklearn')
warnings.filterwarnings(action='ignore', module='scipy')
warnings.filterwarnings(action='ignore', module='optuna')
warnings.filterwarnings(action="ignore", category=ConvergenceWarning, module="sklearn")


class BinaryClassificationModel(BaseModel, ABC):
    model_type = "Binary"
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
        # pass-through calibration controls to BaseModel
    ):
        super().__init__(
            model=model,
            model_name=model_name,
            cv_folds=cv_folds,
            scoring_metric=scoring_metric,
            metric_direction=metric_direction,
            random_state=random_state,
            cv=cv,
            sampler=sampler,
            n_jobs=n_jobs,
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def model_evaluation(self, x_test, y_test):
        """
        Keep the legacy binary outputs (metric_list dict from class_eval + full curves)
        when probabilities are available; otherwise fall back to BaseModel default.
        """
        # If predict_proba is missing, use BaseModel’s default evaluation
        proba_fn = getattr(self.model, "predict_proba", None)
        if proba_fn is None:
            return super().model_evaluation(x_test, y_test)

        y_pred = self.model.predict(x_test)
        probas_ = proba_fn(x_test)
        # If probas shape unexpected, defer to default
        if probas_ is None or (hasattr(probas_, "ndim") and probas_.ndim == 1):
            return super().model_evaluation(x_test, y_test)

        # Legacy metric list via class_eval if available; else compute a minimal dict
        if class_eval is not None:
            metric_list = class_eval(y_test, y_pred)
        else:
            from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score
            metric_list = {
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred)),
                "roc_auc": float(roc_auc_score(y_test, probas_[:, 1])) if probas_ is not None else None,
                "average_precision": float(average_precision_score(y_test, probas_[:, 1])) if probas_ is not None else None,
            }

        fpr, tpr, _ = metrics.roc_curve(y_test, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
        # Reverse to match prior behavior
        prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
        prec_rec_auc = auc(recall, prec)
        ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

        return metric_list, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_


class MulticlassClassificationModel(BaseModel, ABC):
    model_type = "Multiclass"
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
    ):
        super().__init__(
            model=model,
            model_name=model_name,
            cv_folds=cv_folds,
            scoring_metric=scoring_metric,
            metric_direction=metric_direction,
            random_state=random_state,
            cv=cv,
            sampler=sampler,
            n_jobs=n_jobs,
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def model_evaluation(self, x_test, y_test):
        """
        Preserve rich per-class and micro/macro ROC/PR outputs if probabilities exist.
        Otherwise fall back to BaseModel default multiclass evaluation.
        """
        proba_fn = getattr(self.model, "predict_proba", None)
        if proba_fn is None:
            return super().model_evaluation(x_test, y_test)

        y_pred = self.model.predict(x_test)
        probas_ = proba_fn(x_test)
        if probas_ is None or (hasattr(probas_, "ndim") and probas_.ndim != 2):
            return super().model_evaluation(x_test, y_test)

        # Legacy metric dict if available
        if class_eval is not None:
            metric_list = class_eval(y_test, y_pred)
        else:
            from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
            metric_list = {
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred, average="macro")),
            }

        fpr, tpr, roc_auc = dict(), dict(), dict()
        prec, recall, thresholds = dict(), dict(), dict()
        prec_rec_auc, ave_prec = dict(), dict()

        n_classes = len(np.unique(y_test))
        y_test_dummies = pd.get_dummies(y_test, drop_first=False).values

        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_test_dummies[:, i], probas_[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

            prec[i], recall[i], thresholds[i] = metrics.precision_recall_curve(y_test_dummies[:, i], probas_[:, i])
            prec[i], recall[i], thresholds[i] = prec[i][::-1], recall[i][::-1], thresholds[i][::-1]
            prec_rec_auc[i] = auc(recall[i], prec[i])
            ave_prec[i] = metrics.average_precision_score(y_test_dummies[:, i], probas_[:, i])

        # Micro
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test_dummies.ravel(), probas_.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        prec["micro"], recall["micro"], thresholds["micro"] = metrics.precision_recall_curve(
            y_test_dummies.ravel(), probas_.ravel()
        )
        prec["micro"], recall["micro"], thresholds["micro"] = (
            prec["micro"][::-1], recall["micro"][::-1], thresholds["micro"][::-1]
        )
        prec_rec_auc["micro"] = auc(recall["micro"], prec["micro"])
        ave_prec["micro"] = metrics.average_precision_score(y_test_dummies.ravel(), probas_.ravel())

        # Macro (via averaged TPR on common FPR grid)
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        mean_tpr = np.zeros_like(fpr_grid)
        for i in range(n_classes):
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = fpr_grid
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        return metric_list, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_


class RegressionModel(BaseModel, ABC):
    model_type = "Regression"
    def __init__(
        self,
        model,
        model_name,
        cv_folds=3,
        scoring_metric='explained_variance',
        metric_direction='maximize',
        random_state=None,
        cv=None,
        sampler=None,
        n_jobs=None,
    ):
        super().__init__(
            model=model,
            model_name=model_name,
            cv_folds=cv_folds,
            scoring_metric=scoring_metric,
            metric_direction=metric_direction,
            random_state=random_state,
            cv=cv,
            sampler=sampler,
            n_jobs=n_jobs,
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

    def model_evaluation(self, x_test, y_test):
        """
        Preserve the legacy list order for downstream compatibility:
        [max_error, MAE, MSE, MdAE, explained_variance, pearson_corr]
        """
        y_pred = self.predict(x_test)
        y_true = y_test

        me = max_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mdae = median_absolute_error(y_true, y_pred)
        evs = explained_variance_score(y_true, y_pred)
        p_corr = pearsonr(y_true, y_pred)[0]

        return [me, mae, mse, mdae, evs, p_corr]
