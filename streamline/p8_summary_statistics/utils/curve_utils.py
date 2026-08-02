from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def prepare_prc_curve_for_interpolation(
    recall: Sequence[float],
    precision: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return PRC coordinates sorted by increasing recall for interpolation."""
    recall_values = np.asarray(recall, dtype=float)
    precision_values = np.asarray(precision, dtype=float)

    if recall_values.size == 0 or precision_values.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    valid_length = min(recall_values.size, precision_values.size)
    recall_values = recall_values[:valid_length]
    precision_values = precision_values[:valid_length]

    finite_mask = np.isfinite(recall_values) & np.isfinite(precision_values)
    recall_values = recall_values[finite_mask]
    precision_values = precision_values[finite_mask]

    if recall_values.size == 0:
        return recall_values, precision_values

    order = np.argsort(recall_values, kind="mergesort")
    recall_values = recall_values[order]
    precision_values = precision_values[order]

    unique_recall = []
    unique_precision = []
    for value in np.unique(recall_values):
        matching_precision = precision_values[recall_values == value]
        unique_recall.append(float(value))
        unique_precision.append(float(np.max(matching_precision)))

    return np.asarray(unique_recall, dtype=float), np.asarray(unique_precision, dtype=float)


def interpolate_prc_curve(
    recall_grid: Sequence[float],
    recall: Sequence[float],
    precision: Sequence[float],
) -> np.ndarray:
    """Interpolate precision on an increasing recall grid."""
    recall_values, precision_values = prepare_prc_curve_for_interpolation(recall, precision)
    recall_grid_values = np.asarray(recall_grid, dtype=float)

    if recall_values.size == 0 or precision_values.size == 0:
        return np.zeros_like(recall_grid_values)

    return np.interp(
        recall_grid_values,
        recall_values,
        precision_values,
        left=precision_values[0],
        right=precision_values[-1],
    )
