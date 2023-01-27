from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix


def class_eval(y_true, y_pred):
    """
    Calculates standard classification metrics including:
    True positives, false positives, true negative, false negatives, standard accuracy, balanced accuracy
    recall, precision, f1 score, negative predictive value, likelihood ratio positive, and likelihood ratio negative
    """
    # Calculate true positive, true negative, false positive, and false negative.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Calculate Accuracy metrics
    ac = accuracy_score(y_true, y_pred)
    bac = balanced_accuracy_score(y_true, y_pred)
    # Calculate Precision and Recall
    re = recall_score(y_true, y_pred)  # a.k.a. sensitivity or TPR
    pr = precision_score(y_true, y_pred)
    # Calculate F1 score
    f1 = f1_score(y_true, y_pred)
    # Calculate specificity, a.k.a. TNR
    if tn == 0 and fp == 0:
        sp = 0
    else:
        sp = tn / float(tn + fp)
    # Calculate Negative predictive value
    if tn == 0 and fn == 0:
        npv = 0
    else:
        npv = tn / float(tn + fn)
    # Calculate likelihood ratio postive
    if sp == 1:
        lrp = 0
    else:
        lrp = re / float(1 - sp)  # sensitivity / (1-specificity).... a.k.a. TPR/FPR... or TPR/(1-TNR)
    # Calculate likelihood ratio negative
    if sp == 0:
        lrm = 0
    else:
        lrm = (1 - re) / float(sp)  # (1-sensitivity) / specificity... a.k.a. FNR/TNR ... or (1-TPR)/TNR
    return [bac, ac, f1, re, sp, pr, tp, tn, fp, fn, npv, lrp, lrm]
