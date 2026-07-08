from abc import ABC
import warnings

import numpy as np
import optuna
import pandas as pd
from typing import Any, Dict
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
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize

# Phase-6 BaseModel (with calibration)
from streamline.p6_modeling.utils.basemodel import BaseModel
from streamline.p6_modeling.utils.support import _get_probas_or_decision, multiclass_brier_score

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
        Binary evaluation returning:
          - metrics_dict: flat dict of scalar metrics
          - curves_dict: {
                "roc": {"micro": {"fpr": [...], "tpr": [...], "auc": float}},
                "prc": {"micro": {"precision": [...], "recall": [...],
                                  "pr_auc": float, "aps": float}}
            }

        This is what Phase 6 will serialize to JSON.
        """
        probas_ = _get_probas_or_decision(self.model, x_test)
        y_pred = self.model.predict(x_test)

        # --- base metrics that don't require probabilities ---
        cm = confusion_matrix(y_test, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            # defensive fallback
            tn = fp = fn = tp = 0.0

        tn = float(tn)
        fp = float(fp)
        fn = float(fn)
        tp = float(tp)

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        lr_plus = (tp / (tp + fn)) / (fp / (fp + tn)) if (tp + fn) > 0 and (fp + tn) > 0 and fp > 0 else 0.0
        lr_minus = (fn / (tp + fn)) / (tn / (fp + tn)) if (tp + fn) > 0 and (fp + tn) > 0 and tn > 0 else 0.0

        metrics_dict = {
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "specificity": float(specificity),
            "precision": float(precision_score(y_test, y_pred)),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "npv": float(npv),
            "lr_plus": float(lr_plus),
            "lr_minus": float(lr_minus),
            # These will be filled in below if probs / scores are available
            "brier_score": None,
            "roc_auc": None,
            "prc_auc": None,
            "prc_aps": None,
        }

        curves_dict = {"roc": {}, "prc": {}}

        # --- if we have probability-like scores, compute curve-based metrics ---
        if probas_ is None:
            return metrics_dict, curves_dict

        probas_ = np.asarray(probas_)
        # If 1D, treat as positive-class score; if 2D, assume column 1 is positive class
        if probas_.ndim == 1:
            pos_score = probas_
        else:
            # If shape is (n_samples, 2+) use column 1, else last column as positive class
            if probas_.shape[1] >= 2:
                pos_score = probas_[:, 1]
            else:
                pos_score = probas_[:, -1]

        # Brier score (if scores look like probabilities)
        try:
            brier = brier_score_loss(y_test, pos_score)
        except Exception:
            brier = float("nan")

        metrics_dict["brier_score"] = float(brier) if np.isfinite(brier) else None

        # ROC / PRC curves
        fpr, tpr, _ = metrics.roc_curve(y_test, pos_score)
        roc_auc_val = roc_auc_score(y_test, pos_score)

        prec, rec, _ = metrics.precision_recall_curve(y_test, pos_score)
        # AUCs on the natural orientation
        prc_auc_val = auc(rec, prec)
        aps_val = average_precision_score(y_test, pos_score)

        metrics_dict["roc_auc"] = float(roc_auc_val)
        metrics_dict["prc_auc"] = float(prc_auc_val)
        metrics_dict["prc_aps"] = float(aps_val)

        curves_dict["roc"]["micro"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(roc_auc_val),
        }
        curves_dict["prc"]["micro"] = {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "pr_auc": float(prc_auc_val),
            "aps": float(aps_val),
        }

        return metrics_dict, curves_dict


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
        Rich multiclass evaluation.

        Returns:
            metrics_dict: flat dict of scalar metrics (macro/micro variants)
            curves_dict: {
              "roc": {
                  "micro": {fpr,tpr,auc},
                  "macro": {fpr,tpr,auc}
              },
              "prc": {
                  "micro": {precision,recall,pr_auc,aps},
                  "macro": {precision,recall,pr_auc,aps}
              }
            }
        """
        probas_ = _get_probas_or_decision(self.model, x_test)
        if probas_ is None:
            # no probability-like scores; fall back to plain prediction metrics
            y_pred = self.model.predict(x_test)
            metrics_dict = {
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred, average="macro")),
                "brier_score": None,
                "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
                "f1_micro": float(f1_score(y_test, y_pred, average="micro")),
                "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
                "precision_micro": float(precision_score(y_test, y_pred, average="micro")),
                "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
                "recall_micro": float(recall_score(y_test, y_pred, average="micro")),
                "roc_auc_macro": None,
                "roc_auc_micro": None,
                "average_precision_macro": None,
                "average_precision_micro": None,
            }
            return metrics_dict, {"roc": {}, "prc": {}}

        probas_ = np.asarray(probas_)
        if probas_.ndim != 2:
            raise ValueError(
                "Unexpected probability shape in MulticlassClassificationModel.model_evaluation"
            )

        y_pred = self.model.predict(x_test)
        classes = np.asarray(getattr(self.model, "classes_", np.unique(y_test)))
        if classes.size != probas_.shape[1]:
            inferred_classes = np.unique(y_test)
            if inferred_classes.size == probas_.shape[1]:
                classes = inferred_classes
            else:
                raise ValueError(
                    "Multiclass probability columns do not align with class labels in "
                    "MulticlassClassificationModel.model_evaluation"
                )

        y_test_bin = label_binarize(np.asarray(y_test), classes=classes)
        if y_test_bin.ndim == 1:
            y_test_bin = np.column_stack([1 - y_test_bin, y_test_bin])

        # --- metrics dict (as you sketched) ---
        metrics_dict = {
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, average="macro")),
            "brier_score": float(multiclass_brier_score(y_test, probas_, classes=classes)),
            # Additional multiclass metrics to compare
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "f1_micro": float(f1_score(y_test, y_pred, average="micro")),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
            "precision_micro": float(precision_score(y_test, y_pred, average="micro")),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
            "recall_micro": float(recall_score(y_test, y_pred, average="micro")),
            "roc_auc_macro": None,
            "roc_auc_micro": None,
            "average_precision_macro": None,
            "average_precision_micro": None,
        }

        # --- per-class & micro/macro ROC / PR curves ---
        fpr: Dict[Any, np.ndarray] = {}
        tpr: Dict[Any, np.ndarray] = {}
        roc_auc: Dict[Any, float] = {}
        prec: Dict[Any, np.ndarray] = {}
        recall: Dict[Any, np.ndarray] = {}
        prec_rec_auc: Dict[Any, float] = {}
        ave_prec: Dict[Any, float] = {}

        n_classes = len(classes)

        # per-class ROC/PR
        for i in range(n_classes):
            fpr_i, tpr_i, _ = metrics.roc_curve(y_test_bin[:, i], probas_[:, i])
            fpr[i] = fpr_i
            tpr[i] = tpr_i
            roc_auc[i] = auc(fpr_i, tpr_i)

            p_i, r_i, _ = metrics.precision_recall_curve(
                y_test_bin[:, i], probas_[:, i]
            )
            prec[i] = p_i
            recall[i] = r_i
            prec_rec_auc[i] = auc(r_i, p_i)
            ave_prec[i] = metrics.average_precision_score(
                y_test_bin[:, i], probas_[:, i]
            )

        # micro ROC/PR
        fpr_micro, tpr_micro, _ = metrics.roc_curve(
            y_test_bin.ravel(), probas_.ravel()
        )
        roc_auc_micro = auc(fpr_micro, tpr_micro)

        p_micro, r_micro, _ = metrics.precision_recall_curve(
            y_test_bin.ravel(), probas_.ravel()
        )
        pr_auc_micro = auc(r_micro, p_micro)
        aps_micro = metrics.average_precision_score(
            y_test_bin, probas_, average="micro"
        )

        # macro ROC via mean TPR on common grid
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        mean_tpr = np.zeros_like(fpr_grid)
        for i in range(n_classes):
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])
        mean_tpr /= n_classes
        roc_auc_macro = auc(fpr_grid, mean_tpr)

        # macro PR via mean precision on common recall grid
        recall_grid = np.linspace(0.0, 1.0, 1000)
        mean_prec = np.zeros_like(recall_grid)
        for i in range(n_classes):
            mean_prec += np.interp(recall_grid, recall[i], prec[i])
        mean_prec /= n_classes
        pr_auc_macro = auc(recall_grid, mean_prec)
        aps_macro = metrics.average_precision_score(
            y_test_bin, probas_, average="macro"
        )

        metrics_dict["roc_auc_macro"] = float(roc_auc_macro)
        metrics_dict["roc_auc_micro"] = float(roc_auc_micro)
        metrics_dict["average_precision_macro"] = float(aps_macro)
        metrics_dict["average_precision_micro"] = float(aps_micro)

        curves_dict = {
            "roc": {
                "micro": {
                    "fpr": fpr_micro.tolist(),
                    "tpr": tpr_micro.tolist(),
                    "auc": float(roc_auc_micro),
                },
                "macro": {
                    "fpr": fpr_grid.tolist(),
                    "tpr": mean_tpr.tolist(),
                    "auc": float(roc_auc_macro),
                },
            },
            "prc": {
                "micro": {
                    "precision": p_micro.tolist(),
                    "recall": r_micro.tolist(),
                    "pr_auc": float(pr_auc_micro),
                    "aps": float(aps_micro),
                },
                "macro": {
                    "precision": mean_prec.tolist(),
                    "recall": recall_grid.tolist(),
                    "pr_auc": float(pr_auc_macro),
                    "aps": float(aps_macro),
                },
            },
        }

        return metrics_dict, curves_dict



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
        Regression evaluation returning a flat metric dict:

        {
          "max_error": ...,
          "mean_absolute_error": ...,
          "mean_squared_error": ...,
          "median_absolute_error": ...,
          "explained_variance": ...,
          "pearson_correlation": ...
        }
        """
        y_pred = self.predict(x_test)
        y_true = np.asarray(y_test)

        me = max_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mdae = median_absolute_error(y_true, y_pred)
        evs = explained_variance_score(y_true, y_pred)
        p_corr = pearsonr(y_true, y_pred)[0]

        return {
            "max_error": float(me),
            "mean_absolute_error": float(mae),
            "mean_squared_error": float(mse),
            "median_absolute_error": float(mdae),
            "explained_variance": float(evs),
            "pearson_correlation": float(p_corr),
        }
