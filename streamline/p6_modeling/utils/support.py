import logging

import numpy as np
from sklearn.preprocessing import label_binarize



# ---------------------------------------------------------------------
# Support Functions
# ---------------------------------------------------------------------

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

def multiclass_brier_score(y_true, y_prob, classes=None):
    """
    y_true: (n_samples,) integer labels
    y_prob: (n_samples, n_classes) predicted probabilities
    https://stats.stackexchange.com/questions/403544/how-to-compute-the-brier-score-for-more-than-two-classes
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_prob.ndim != 2:
        raise ValueError("Multiclass Brier score expects a 2D probability array.")

    if classes is None:
        classes = np.unique(y_true)
    classes = np.asarray(classes)

    if classes.size != y_prob.shape[1]:
        inferred_classes = np.unique(y_true)
        if inferred_classes.size == y_prob.shape[1]:
            classes = inferred_classes
        else:
            raise ValueError(
                f"Multiclass Brier score class/probability mismatch: "
                f"{classes.size} classes vs {y_prob.shape[1]} probability columns."
            )

    y_true_onehot = label_binarize(y_true, classes=classes)
    if y_true_onehot.ndim == 1:
        y_true_onehot = np.column_stack([1 - y_true_onehot, y_true_onehot])
    y_true_onehot = y_true_onehot.astype(float, copy=False)

    # Compute multiclass Brier score
    return np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))
