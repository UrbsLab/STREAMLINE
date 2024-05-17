from abc import ABC
import numpy as np
import optuna
import pandas as pd
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.metrics import auc, max_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, median_absolute_error, explained_variance_score
from sklearn.model_selection import KFold
from streamline.modeling.basemodel import BaseModel
from streamline.utils.evaluation import class_eval
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress specific warnings from sklearn, scipy, and optuna
warnings.filterwarnings(action='ignore', module='sklearn')
warnings.filterwarnings(action='ignore', module='scipy')
warnings.filterwarnings(action='ignore', module='optuna')
warnings.filterwarnings(action="ignore", category=ConvergenceWarning, module="sklearn")

class BinaryClassificationModel(BaseModel, ABC):
    def __init__(self, model, model_name, cv_folds=3, scoring_metric='balanced_accuracy', 
                 metric_direction='maximize', random_state=None, cv=None, sampler=None, n_jobs=None):
        """
        Binary Classification Model Class.

        Args:
            model: The model to be used.
            model_name: The name of the model.
            cv_folds: Number of cross-validation folds.
            scoring_metric: Metric used for scoring the model.
            metric_direction: Direction to optimize the scoring metric ('maximize' or 'minimize').
            random_state: Random state for reproducibility.
            cv: Custom cross-validation strategy.
            sampler: Sampler for hyperparameter optimization.
            n_jobs: Number of parallel jobs for cross-validation.
        """
        super().__init__(model, model_name, cv_folds, scoring_metric, metric_direction, random_state, cv, sampler, n_jobs)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.model_type = "BinaryClassification"

    def model_evaluation(self, x_test, y_test):
        """
        Evaluate the binary classification model.

        Args:
            x_test: Test data.
            y_test: True labels for the test data.

        Returns:
            A tuple containing various evaluation metrics.
        """
        # Prediction evaluation
        y_pred = self.model.predict(x_test)
        metric_list = class_eval(y_test, y_pred)
        
        # Determine probabilities of class predictions for each test instance
        probas_ = self.model.predict_proba(x_test)
        
        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Compute Precision/Recall curve and AUC
        prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
        prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
        prec_rec_auc = auc(recall, prec)
        ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])
        
        return metric_list, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_

class MulticlassClassificationModel(BaseModel, ABC):
    def __init__(self, model, model_name, cv_folds=3, scoring_metric='balanced_accuracy', 
                 metric_direction='maximize', random_state=None, cv=None, sampler=None, n_jobs=None):
        """
        Multiclass Classification Model Class.

        Args:
            model: The model to be used.
            model_name: The name of the model.
            cv_folds: Number of cross-validation folds.
            scoring_metric: Metric used for scoring the model.
            metric_direction: Direction to optimize the scoring metric ('maximize' or 'minimize').
            random_state: Random state for reproducibility.
            cv: Custom cross-validation strategy.
            sampler: Sampler for hyperparameter optimization.
            n_jobs: Number of parallel jobs for cross-validation.
        """
        super().__init__(model, model_name, cv_folds, scoring_metric, metric_direction, random_state, cv, sampler, n_jobs)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.model_type = "MulticlassClassification"

    def model_evaluation(self, x_test, y_test):
        """
        Evaluate the multiclass classification model.

        Args:
            x_test: Test data.
            y_test: True labels for the test data.

        Returns:
            A tuple containing various evaluation metrics.
        """
        # Prediction evaluation
        y_pred = self.model.predict(x_test)
        metric_list = class_eval(y_test, y_pred)
        
        # Determine probabilities of class predictions for each test instance
        probas_ = self.model.predict_proba(x_test)
        
        # Initialize dictionaries to store evaluation metrics for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        prec = dict()
        recall = dict()
        thresholds = dict()
        prec_rec_auc = dict()
        ave_prec = dict()

        # Calculate dummies once
        n_classes = len(np.unique(y_test))
        y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
        
        # Compute metrics for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_test_dummies[:, i], probas_[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            prec[i], recall[i], thresholds[i] = metrics.precision_recall_curve(y_test_dummies[:, i], probas_[:, i])
            prec[i], recall[i], thresholds[i] = prec[i][::-1], recall[i][::-1], thresholds[i][::-1]
            prec_rec_auc[i] = auc(recall[i], prec[i])
            ave_prec[i] = metrics.average_precision_score(y_test_dummies[:, i], probas_[:, i])

        # Compute micro-average metrics
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test_dummies.ravel(), probas_.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        prec["micro"], recall["micro"], thresholds["micro"] = metrics.precision_recall_curve(y_test_dummies.ravel(), probas_.ravel())
        prec["micro"], recall["micro"], thresholds["micro"] = prec["micro"][::-1], recall["micro"][::-1], thresholds["micro"][::-1]
        prec_rec_auc["micro"] = auc(recall["micro"], prec["micro"])
        ave_prec["micro"] = metrics.average_precision_score(y_test_dummies.ravel(), probas_.ravel())

        # Interpolate all ROC curves at these points
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        mean_tpr = np.zeros_like(fpr_grid)
        for i in range(n_classes):
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # Linear interpolation

        # Average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = fpr_grid
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        return metric_list, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_

class RegressionModel(BaseModel, ABC):
    def __init__(self, model, model_name, cv_folds=3, scoring_metric='explained_variance', 
                 metric_direction='maximize', random_state=None, cv=None, sampler=None, n_jobs=None):
        """
        Regression Model Class.

        Args:
            model: The model to be used.
            model_name: The name of the model.
            cv_folds: Number of cross-validation folds.
            scoring_metric: Metric used for scoring the model.
            metric_direction: Direction to optimize the scoring metric ('maximize' or 'minimize').
            random_state: Random state for reproducibility.
            cv: Custom cross-validation strategy.
            sampler: Sampler for hyperparameter optimization.
            n_jobs: Number of parallel jobs for cross-validation.
        """
        super().__init__(model, model_name, cv_folds, scoring_metric, metric_direction, random_state, cv, sampler, n_jobs)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.model_type = "Regression"
        self.cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

    def model_evaluation(self, x_test, y_test):
        """
        Evaluate the regression model.

        Args:
            x_test: Test data.
            y_test: True labels for the test data.

        Returns:
            A list containing various regression evaluation metrics.
        """
        y_pred = self.predict(x_test)  # Predicted values
        y_true = y_test  # True values

        # Calculate various regression metrics
        me = max_error(y_true, y_pred)  # Max error
        mae = mean_absolute_error(y_true, y_pred)  # Mean absolute error
        mse = mean_squared_error(y_true, y_pred)  # Mean squared error
        mdae = median_absolute_error(y_true, y_pred)  # Median absolute error
        evs = explained_variance_score(y_true, y_pred)  # Explained variance score
        p_corr = pearsonr(y_true, y_pred)[0]  # Pearson correlation coefficient

        return [me, mae, mse, mdae, evs, p_corr]
