# streamline/p4_feature_importance/registry/mutual_information.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from streamline.p4_feature_importance.utils.input_normalization import (
    normalize_feature_matrix,
    normalize_target_vector,
)

class MutualInformation:
    id = "mutualinformation"
    model_name = "Mutual Information"
    small_name = "MI"
    path_name = "mutualinformation"

    def __init__(self, outcome_type: str = "Binary", n_neighbors: int = 3, random_state: int | None = None, **kwargs):
        self.outcome_type = outcome_type
        self.n_neighbors = int(n_neighbors)
        self.random_state = random_state
        self.kwargs = kwargs
        self._cols: List[str] = []
        self._scores: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._cols = X.columns.tolist()
        Xn = X.select_dtypes(include=["number"])
        cols = Xn.columns.tolist()
        _, X_array = normalize_feature_matrix(Xn)
        y_array = normalize_target_vector(y)
        if self.outcome_type in ("Binary", "Multiclass"):
            scores = mutual_info_classif(
                X_array,
                y_array,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
            )
        else:
            scores = mutual_info_regression(
                X_array,
                y_array,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
            )
        self._scores = {c: float(s) for c, s in zip(cols, scores)}
        for c in self._cols:
            self._scores.setdefault(c, 0.0)
        return self

    def _ranked(self) -> List[str]:
        return sorted(self._cols, key=lambda c: self._scores.get(c,0.0), reverse=True)

    def get_support_mask(self, *, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[bool]:
        if top_k is not None:
            keep = set(self._ranked()[:int(top_k)])
            return [c in keep for c in self._cols]
        if threshold is not None:
            return [self._scores.get(c,0.0) >= float(threshold) for c in self._cols]
        return [True]*len(self._cols)

    def get_support_names(self, cols: List[str], *, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[str]:
        mask = self.get_support_mask(top_k=top_k, threshold=threshold)
        return [c for c, m in zip(self._cols, mask) if m]

    def transform(self, X: pd.DataFrame, *, top_k: Optional[int] = None, threshold: Optional[float] = None) -> pd.DataFrame:
        names = self.get_support_names(self._cols, top_k=top_k, threshold=threshold)
        return X.loc[:, names]

    def get_scores(self) -> Dict[str, float]:
        return dict(self._scores)

    def get_params(self) -> Dict[str, Any]:
        p = {"outcome_type": self.outcome_type, "n_neighbors": self.n_neighbors, "random_state": self.random_state}
        p.update(self.kwargs or {})
        return p
