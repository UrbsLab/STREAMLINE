import logging

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix


def class_eval(y_true, y_pred):
    """
    Classification metric bundle.

    Binary case returns:
        [bac, ac, f1, re, sp, pr, tp, tn, fp, fn, npv, lrp, lrm]

    Multiclass case returns:
        [bac, ac, f1, re, pr]
    """
    # Binary classification
    if np.unique(y_true).size == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Accuracy metrics
        ac = accuracy_score(y_true, y_pred)
        bac = balanced_accuracy_score(y_true, y_pred)

        # Precision, recall, F1
        re = recall_score(y_true, y_pred, average="weighted")  # sensitivity / TPR
        pr = precision_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")

        # Specificity (TNR)
        if tn == 0 and fp == 0:
            sp = 0.0
        else:
            sp = tn / float(tn + fp)

        # Negative predictive value
        if tn == 0 and fn == 0:
            npv = 0.0
        else:
            npv = tn / float(tn + fn)

        # Likelihood ratio positive: sensitivity / (1 - specificity)
        if sp == 1:
            lrp = 0.0
        else:
            lrp = re / float(1 - sp)

        # Likelihood ratio negative: (1 - sensitivity) / specificity
        if sp == 0:
            lrm = 0.0
        else:
            lrm = (1 - re) / float(sp)

        return [bac, ac, f1, re, sp, pr, tp, tn, fp, fn, npv, lrp, lrm]

    else:
        #Multiclass case
        ac = accuracy_score(y_true, y_pred)
        bac = balanced_accuracy_score(y_true, y_pred)
        re = recall_score(y_true, y_pred, average="weighted")
        pr = precision_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")
        return [bac, ac, f1, re, pr]


def multiclass_brier_score(y_true, y_prob):
    """
    y_true: (n_samples,) integer labels
    y_prob: (n_samples, n_classes) predicted probabilities
    """
    # One-hot encode labels
    enc = OneHotEncoder(sparse_output=False)
    y_true_onehot = enc.fit_transform(y_true.reshape(-1, 1))

    # Compute multiclass Brier score
    return np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))