from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from streamline.p4_feature_importance.utils.input_normalization import (
    normalize_feature_matrix,
    normalize_target_vector,
)


class MultiSURF:
    id = "multisurf"
    model_name = "MultiSURF"
    small_name = "MS"
    path_name = "multisurf"
    uses_instance_subset = True

    def __init__(
        self,
        n_features_to_select: Optional[int] = None,
        n_neighbors: int | float = 100,
        categorical_features: Optional[List[int]] = None,
        categorical_threshold: int = 10,
        multiclass_threshold: int = 10,
        verbose: bool = False,
        n_jobs: Optional[int] = None,
        weight_final_scores: bool = False,
        rank_absolute: bool = False,
        label_type: Optional[str] = None,
        random_state: Optional[int] = None,
        use_turf: bool = False,
        turf_pct: int | float | None = None,
        turf_num_scores_to_return: Optional[int] = None,
        **kwargs,
    ):
        self.n_features_to_select = n_features_to_select
        self.n_neighbors = n_neighbors
        self.categorical_features = list(categorical_features or [])
        self.categorical_threshold = categorical_threshold
        self.multiclass_threshold = multiclass_threshold
        self.verbose = verbose
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.weight_final_scores = weight_final_scores
        self.rank_absolute = rank_absolute
        self.label_type = label_type
        self.random_state = random_state
        self.use_turf = bool(use_turf)
        self.turf_pct = turf_pct
        self.turf_num_scores_to_return = turf_num_scores_to_return
        self.kwargs = kwargs
        self.columns: List[str] = []
        self.scores: Dict[str, float] = {}
        self.implementation = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.columns = X.columns.tolist()
        X_array = self.normalize_rebate_matrix(X)
        y_array = normalize_target_vector(y)
        try:
            from skrebate import MultiSURF as RebateModel, TURF
        except (ModuleNotFoundError, ImportError) as e:
            raise Exception("MultiSURF requires 'skrebate==0.8.2'.") from e

        base = RebateModel(**self.build_rebate_params(len(self.columns)))
        if self.use_turf:
            turf_pct = 0.5 if self.turf_pct is None else self.turf_pct
            score_count = self.turf_num_scores_to_return or len(self.columns)
            self.implementation = TURF(
                base,
                pct=turf_pct,
                num_scores_to_return=int(score_count),
            ).fit(X_array, y_array)
        else:
            self.implementation = base.fit(X_array, y_array)

        self.scores = self.build_scores(getattr(self.implementation, "feature_importances_", None))
        return self

    def normalize_rebate_matrix(self, X: pd.DataFrame) -> np.ndarray:
        X_rebate = X.copy()
        for idx in self.categorical_feature_indexes():
            if idx >= len(self.columns):
                continue
            col = self.columns[idx]
            series = X_rebate[col]
            numeric = pd.to_numeric(series, errors="coerce")
            non_missing = series.notna()
            if numeric[non_missing].notna().all():
                X_rebate[col] = numeric
            else:
                codes, _ = pd.factorize(series, sort=True, use_na_sentinel=True)
                coded = codes.astype(float)
                coded[codes == -1] = np.nan
                X_rebate[col] = coded
        _, X_array = normalize_feature_matrix(X_rebate)
        return X_array

    def categorical_feature_indexes(self) -> List[int]:
        indexes = []
        for value in self.categorical_features:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if index >= 0:
                indexes.append(index)
        return indexes

    def build_rebate_params(self, feature_count: int) -> Dict[str, Any]:
        params = {
            "n_features_to_select": self.n_features_to_select or feature_count,
            "n_neighbors": self.n_neighbors,
            "categorical_features": self.categorical_feature_indexes(),
            "categorical_threshold": self.categorical_threshold,
            "multiclass_threshold": self.multiclass_threshold,
            "verbose": self.verbose,
            "n_jobs": self.n_jobs,
            "weight_final_scores": self.weight_final_scores,
            "rank_absolute": self.rank_absolute,
            "label_type": self.label_type,
        }
        params.update(self.kwargs or {})
        return params

    def build_scores(self, importances) -> Dict[str, float]:
        if importances is None:
            values = np.zeros(len(self.columns), dtype=float)
        else:
            values = np.asarray(importances, dtype=float)
            if len(values) < len(self.columns):
                values = np.pad(values, (0, len(self.columns) - len(values)), constant_values=0.0)
            elif len(values) > len(self.columns):
                values = values[: len(self.columns)]
        return {c: float(s) for c, s in zip(self.columns, values)}

    def ranked_features(self) -> List[str]:
        return sorted(self.columns, key=lambda c: self.scores.get(c, 0.0), reverse=True)

    def get_support_mask(self, *, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[bool]:
        if top_k is not None:
            keep = set(self.ranked_features()[:int(top_k)])
            return [c in keep for c in self.columns]
        if threshold is not None:
            return [self.scores.get(c, 0.0) >= float(threshold) for c in self.columns]
        return [True] * len(self.columns)

    def get_support_names(self, cols: List[str], *, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[str]:
        mask = self.get_support_mask(top_k=top_k, threshold=threshold)
        return [c for c, m in zip(self.columns, mask) if m]

    def transform(self, X: pd.DataFrame, *, top_k: Optional[int] = None, threshold: Optional[float] = None) -> pd.DataFrame:
        names = self.get_support_names(self.columns, top_k=top_k, threshold=threshold)
        return X.loc[:, names]

    def get_scores(self) -> Dict[str, float]:
        return dict(self.scores)

    def get_params(self) -> Dict[str, Any]:
        params = {
            "n_features_to_select": self.n_features_to_select,
            "n_neighbors": self.n_neighbors,
            "categorical_features": self.categorical_feature_indexes(),
            "categorical_threshold": self.categorical_threshold,
            "multiclass_threshold": self.multiclass_threshold,
            "verbose": self.verbose,
            "n_jobs": self.n_jobs,
            "weight_final_scores": self.weight_final_scores,
            "rank_absolute": self.rank_absolute,
            "label_type": self.label_type,
            "random_state": self.random_state,
            "use_turf": self.use_turf,
            "turf_pct": self.turf_pct,
            "turf_num_scores_to_return": self.turf_num_scores_to_return,
        }
        params.update(self.kwargs or {})
        return params
