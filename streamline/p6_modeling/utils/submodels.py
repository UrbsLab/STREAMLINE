from __future__ import annotations

from abc import ABC
import warnings

import numpy as np
import optuna
import pandas as pd
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    auc,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    explained_variance_score,
    brier_score_loss,
)
from sklearn.model_selection import KFold

# Phase-6 BaseModel (with calibration + default evaluations)
from streamline.p6_modeling.utils.basemodel import BaseModel
from streamline.p6_modeling.utils.evaluation import class_eval, multiclass_brier_score

# ---------------------------------------------------------------------
# Global warning configuration
# ---------------------------------------------------------------------
warnings.filterwarnings(action="ignore", module="sklearn")
warnings.filterwarnings(action="ignore", module="scipy")
warnings.filterwarnings(action="ignore", module="optuna")
warnings.filterwarnings(
    action="ignore", category=ConvergenceWarning, module="sklearn"
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

def _get_probas_or_decision(model, x):
    """
    Try predict_proba, then decision_function.

    Returns:
        array-like or None if neither method is available.
    """
    proba_fn = getattr(model, "predict_proba", None)
    if proba_fn is not None:
        return proba_fn(x)

    decision_fn = getattr(model, "decision_function", None)
    if decision_fn is not None:
        return decision_fn(x)

    return None


# ---------------------------------------------------------------------
# BinaryClassificationModel
# ---------------------------------------------------------------------
class BinaryClassificationModel(BaseModel, ABC):
    model_type = "Binary"

    def __init__(
        self,
        model,
        model_name,
        cv_folds: int = 3,
        scoring_metric: str = "balanced_accuracy",
        metric_direction: str = "maximize",
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

    def model_evaluation(self, x_test, y_test):
        """
        Legacy-style binary evaluation.

        Returns:
            metric_list, fpr, tpr, roc_auc, prec, recall, prec_rec_auc,
            ave_prec, probas_
        """
        probas_ = _get_probas_or_decision(self.model, x_test)

        # If model has no proba/decision, delegate to BaseModel's predict
        if probas_ is None:
            return super().predict(x_test, y_test)

        y_pred = self.model.predict(x_test)

        # Expect shape (n_samples, 2) for binary; otherwise raise
        if hasattr(probas_, "ndim") and probas_.ndim == 1:
            raise ValueError(
                "Unexpected 1D probability/score array in "
                "BinaryClassificationModel.model_evaluation"
            )

        # Positive class probability assumed in column 1
        try:
            pos_proba = probas_[:, 1]
            brier = brier_score_loss(y_test, pos_proba)
        except Exception:
            pos_proba = None
            brier = float("nan")

        # Legacy metric list
        metric_list = class_eval(y_test, y_pred)
        metric_list.append(brier)  # Brier score at end

        # ROC curve
        fpr, tpr, _ = metrics.roc_curve(y_test, probas_[:, 1])
        roc_auc = auc(fpr, tpr)

        # PR curve
        prec, recall, thresholds = metrics.precision_recall_curve(
            y_test, probas_[:, 1]
        )
        # Reverse to match previous behavior
        prec = prec[::-1]
        recall = recall[::-1]
        thresholds = thresholds[::-1]
        prec_rec_auc = auc(recall, prec)
        ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

        # metric_list is a list (legacy); add Brier by convention if needed
        # (kept separate so you don't break downstream consumers)
        # You can augment metric_list or wrap this in a dict externally.

        return (
            metric_list,
            fpr,
            tpr,
            roc_auc,
            prec,
            recall,
            prec_rec_auc,
            ave_prec,
            probas_,
        )


# ---------------------------------------------------------------------
# MulticlassClassificationModel
# ---------------------------------------------------------------------
class MulticlassClassificationModel(BaseModel, ABC):
    model_type = "Multiclass"

    def __init__(
        self,
        model,
        model_name,
        cv_folds: int = 3,
        scoring_metric: str = "balanced_accuracy",
        metric_direction: str = "maximize",
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

    def model_evaluation(self, x_test, y_test):
        """
        Rich multiclass evaluation with per-class and micro/macro ROC/PR curves.

        Returns:
            metric_list, fpr, tpr, roc_auc, prec, recall,
            prec_rec_auc, ave_prec, probas_
        """
        probas_ = _get_probas_or_decision(self.model, x_test)

        if probas_ is None:
            # No probability-like scores delegate to BaseModel prediction
            return super().predict(x_test, y_test)

        if probas_ is None or (hasattr(probas_, "ndim") and probas_.ndim != 2):
            raise ValueError(
                "Unexpected probability shape in "
                "MulticlassClassificationModel.prect_probas"
            )

        y_pred = self.model.predict(x_test)

        # Old metric list
        metric_list = class_eval(y_test, y_pred)

        fpr, tpr, roc_auc = dict(), dict(), dict()
        prec, recall, thresholds = dict(), dict(), dict()
        prec_rec_auc, ave_prec = dict(), dict()

        n_classes = len(np.unique(y_test))
        y_test_dummies = pd.get_dummies(y_test, drop_first=False).values

        # Per-class ROC / PR
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(
                y_test_dummies[:, i], probas_[:, i]
            )
            roc_auc[i] = auc(fpr[i], tpr[i])

            p_i, r_i, thr_i = metrics.precision_recall_curve(
                y_test_dummies[:, i], probas_[:, i]
            )

            # Store *reversed* for backwards compatibility
            prec[i] = p_i[::-1]
            recall[i] = r_i[::-1]
            thresholds[i] = thr_i[::-1]

            # AUC and average precision computed on the "natural" curves
            prec_rec_auc[i] = auc(r_i, p_i)
            ave_prec[i] = metrics.average_precision_score(
                y_test_dummies[:, i], probas_[:, i]
            )

        # Micro ROC / PR
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(
            y_test_dummies.ravel(), probas_.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        p_micro, r_micro, thr_micro = metrics.precision_recall_curve(
            y_test_dummies.ravel(), probas_.ravel()
        )
        prec["micro"] = p_micro[::-1]
        recall["micro"] = r_micro[::-1]
        thresholds["micro"] = thr_micro[::-1]
        prec_rec_auc["micro"] = auc(r_micro, p_micro)
        ave_prec["micro"] = metrics.average_precision_score(
            y_test_dummies.ravel(), probas_.ravel()
        )

        # Macro ROC via averaged TPR on a common FPR grid
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        mean_tpr = np.zeros_like(fpr_grid)
        for i in range(n_classes):
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = fpr_grid
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Macro PR via averaged precision on a common recall grid
        recall_grid = np.linspace(0.0, 1.0, 1000)
        mean_prec = np.zeros_like(recall_grid)
        for i in range(n_classes):
            # recall[i] / prec[i] are stored reversed; flip back before interp
            r = recall[i][::-1]
            p = prec[i][::-1]
            mean_prec += np.interp(recall_grid, r, p)

        mean_prec /= n_classes
        recall["macro"] = recall_grid
        prec["macro"] = mean_prec

        # AUC of macro PR curve
        prec_rec_auc["macro"] = auc(recall["macro"], prec["macro"])

        # Macro average precision (scalar, sklearn's macro definition)
        ave_prec["macro"] = metrics.average_precision_score(
            y_test_dummies, probas_, average="macro"
        )

        # Brier score (multiclass)
        metric_list.append(multiclass_brier_score(y_test, probas_))

        return (
            metric_list,
            fpr,
            tpr,
            roc_auc,
            prec,
            recall,
            prec_rec_auc,
            ave_prec,
            probas_,
        )


# ---------------------------------------------------------------------
# RegressionModel
# ---------------------------------------------------------------------
class RegressionModel(BaseModel, ABC):
    model_type = "Regression"

    def __init__(
        self,
        model,
        model_name,
        cv_folds: int = 3,
        scoring_metric: str = "explained_variance",
        metric_direction: str = "maximize",
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
        self.cv = KFold(
            n_splits=cv_folds, shuffle=True, random_state=self.random_state
        )

    def model_evaluation(self, x_test, y_test):
        """
        Preserve the legacy list order for downstream compatibility:

        [max_error, MAE, MSE, MdAE, explained_variance, pearson_corr]
        """
        y_pred = self.predict(x_test)
        y_true = np.asarray(y_test)

        me = max_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mdae = median_absolute_error(y_true, y_pred)
        evs = explained_variance_score(y_true, y_pred)
        p_corr = pearsonr(y_true, y_pred)[0]

        return [me, mae, mse, mdae, evs, p_corr]
