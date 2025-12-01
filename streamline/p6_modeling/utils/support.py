import logging

import numpy as np
from sklearn.preprocessing import OneHotEncoder



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
