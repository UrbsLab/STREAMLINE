import numpy as np
import pandas as pd

from streamline.p4_feature_importance.registry.mutualinformation import MutualInformation
from streamline.p4_feature_importance.utils.input_normalization import (
    normalize_feature_matrix,
    normalize_target_vector,
)


def test_p4_normalize_feature_matrix_handles_nullable_pandas_dtypes():
    X = pd.DataFrame(
        {
            "float_ext": pd.Series([1.5, pd.NA, 3.5], dtype="Float64"),
            "int_ext": pd.Series([1, 2, pd.NA], dtype="Int64"),
            "bool_ext": pd.Series([True, False, pd.NA], dtype="boolean"),
        }
    )

    X_numeric, X_array = normalize_feature_matrix(X)

    assert list(X_numeric.columns) == ["float_ext", "int_ext", "bool_ext"]
    assert X_array.dtype == np.float64
    assert X_array.shape == (3, 3)
    assert np.isnan(X_array[1, 0])
    assert np.isnan(X_array[2, 1])
    assert np.isnan(X_array[2, 2])


def test_p4_normalize_target_vector_factorizes_non_numeric_labels():
    y = pd.Series(["A", "B", "A", "C"], dtype="string")

    y_array = normalize_target_vector(y)

    assert y_array.dtype == np.int64
    assert y_array.tolist() == [0, 1, 0, 2]


def test_mutual_information_accepts_nullable_numeric_pandas_inputs():
    X = pd.DataFrame(
        {
            "signal": pd.Series([0, 0, 1, 1, 0, 1], dtype="Int64"),
            "noise": pd.Series([3, 1, 4, 1, 5, 9], dtype="Int64"),
        }
    )
    y = pd.Series([0, 0, 1, 1, 0, 1], dtype="Int64")

    model = MutualInformation(outcome_type="Binary", n_neighbors=3, random_state=0).fit(X, y)
    scores = model.get_scores()

    assert set(scores) == {"signal", "noise"}
    assert all(np.isfinite(v) for v in scores.values())
