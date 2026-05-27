from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


MISSING_CATEGORY_TOKEN = "__STREAMLINE_MISSING__"


def normalize_model_id(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def parse_model_id_csv(value: object, default: Optional[Iterable[str]] = None) -> set[str]:
    if value is None:
        tokens = list(default or [])
    elif isinstance(value, (list, tuple, set)):
        tokens = list(value)
    else:
        tokens = [x.strip() for x in str(value).split(",") if x.strip()]
    return {normalize_model_id(token) for token in tokens if str(token).strip()}


def cast_native_categoricals(df: pd.DataFrame, categorical_columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in categorical_columns:
        if col in out.columns:
            out[col] = out[col].where(out[col].notna(), MISSING_CATEGORY_TOKEN).astype(str)
    return out


def one_hot_align(
    train_or_input: pd.DataFrame,
    categorical_columns: Iterable[str],
    encoded_feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    categorical_columns = [c for c in categorical_columns if c in train_or_input.columns]
    encoded = pd.get_dummies(train_or_input, columns=categorical_columns)
    if encoded_feature_names is not None:
        encoded = encoded.reindex(columns=encoded_feature_names, fill_value=0)
    for col in encoded.columns:
        if encoded[col].dtype == bool:
            encoded[col] = encoded[col].astype(int)
    return encoded


class FeatureTypeModelWrapper(BaseEstimator):
    """
    Prediction-time adapter for models trained from raw categorical CV columns.

    P6 may train non-native estimators on one-hot-expanded features or native
    categorical estimators on raw categorical columns. The saved estimator needs
    to remember that preparation because later phases load only the pickle.
    """

    def __init__(
        self,
        estimator,
        *,
        mode: str,
        raw_feature_names: Iterable[str],
        categorical_columns: Iterable[str],
        encoded_feature_names: Optional[Iterable[str]] = None,
    ):
        self.estimator = estimator
        self.mode = mode
        self.raw_feature_names = list(raw_feature_names)
        self.categorical_columns = list(categorical_columns)
        self.encoded_feature_names = (
            list(encoded_feature_names) if encoded_feature_names is not None else None
        )

    def _as_raw_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.raw_feature_names if c not in X.columns]
            if missing:
                return X.copy()
            return X.loc[:, self.raw_feature_names].copy()

        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] == len(self.raw_feature_names):
            return pd.DataFrame(arr, columns=self.raw_feature_names)
        return pd.DataFrame(arr)

    def _prepare(self, X):
        if self.mode == "one_hot":
            arr = np.asarray(X)
            if (
                not isinstance(X, pd.DataFrame)
                and arr.ndim == 2
                and arr.shape[1] == len(self.raw_feature_names)
            ):
                raw = self._as_raw_frame(X)
                return one_hot_align(
                    raw,
                    self.categorical_columns,
                    self.encoded_feature_names,
                ).values
            if (
                not isinstance(X, pd.DataFrame)
                and arr.ndim == 2
                and self.encoded_feature_names is not None
                and arr.shape[1] == len(self.encoded_feature_names)
            ):
                return X
            if isinstance(X, pd.DataFrame) and self.encoded_feature_names is not None:
                if all(c in X.columns for c in self.encoded_feature_names):
                    return X.loc[:, self.encoded_feature_names].values
            raw = self._as_raw_frame(X)
            return one_hot_align(
                raw,
                self.categorical_columns,
                self.encoded_feature_names,
            ).values

        if self.mode == "native":
            raw = self._as_raw_frame(X)
            return cast_native_categoricals(raw, self.categorical_columns)

        return X

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(self._prepare(X), y, **fit_params)
        return self

    def predict(self, X):
        return self.estimator.predict(self._prepare(X))

    def transform_features(self, X):
        return self._prepare(X)

    @property
    def classes_(self):
        return getattr(self.estimator, "classes_")

    @property
    def feature_importances_(self):
        return getattr(self.estimator, "feature_importances_")

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name in {"predict_proba", "decision_function", "predict_log_proba", "score"}:
            estimator_method = getattr(self.estimator, name)

            def _wrapped(X, *args, **kwargs):
                return estimator_method(self._prepare(X), *args, **kwargs)

            return _wrapped
        return getattr(self.estimator, name)
