import numpy as np

from streamline.p8_summary_statistics.utils.curve_utils import (
    interpolate_prc_curve,
    prepare_prc_curve_for_interpolation,
)


def test_prc_curve_interpolation_accepts_sklearn_descending_recall():
    recall = np.array([1.0, 0.5, 0.0])
    precision = np.array([0.2, 0.7, 1.0])

    interpolated = interpolate_prc_curve(np.array([0.0, 0.5, 1.0]), recall, precision)

    assert np.allclose(interpolated, np.array([1.0, 0.7, 0.2]))


def test_prc_curve_preparation_preserves_sklearn_threshold_order_for_duplicate_recall():
    recall = np.array([1.0, 0.5, 0.5, 0.0])
    precision = np.array([0.2, 0.4, 0.8, 1.0])

    sorted_recall, sorted_precision = prepare_prc_curve_for_interpolation(recall, precision)

    assert np.allclose(sorted_recall, np.array([0.0, 0.5, 0.5, 1.0]))
    assert np.allclose(sorted_precision, np.array([1.0, 0.8, 0.4, 0.2]))
