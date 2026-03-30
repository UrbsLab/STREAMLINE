from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def normalize_feature_matrix(X: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Coerce a pandas feature matrix to a NumPy-friendly numeric representation.

    This keeps the DataFrame form for column bookkeeping inside STREAMLINE while
    also producing a float64 ndarray for older third-party libraries that expect
    plain numeric NumPy inputs.
    """
    X_numeric = X.apply(pd.to_numeric, errors="coerce")
    X_array = X_numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    return X_numeric, X_array


def normalize_target_vector(y: pd.Series) -> np.ndarray:
    """
    Coerce outcome labels to a NumPy-friendly vector.

    Numeric targets stay numeric. Non-numeric classification labels are
    factorized so NumPy/scikit-compatible arrays are always produced.
    """
    if pd.api.types.is_numeric_dtype(y):
        return pd.to_numeric(y, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)

    labels, _ = pd.factorize(y)
    return labels.astype(np.int64, copy=False)

