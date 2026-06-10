# streamline/p4_feature_importance/registry/multisurf.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
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

    def __init__(self, n_neighbors: Optional[int] = None, random_state: int | None = None,
                 use_turf: bool = False, turf_pct: float | None = None, n_jobs: int | None = None, **kwargs):
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.use_turf = bool(use_turf)
        self.turf_pct = turf_pct
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.kwargs = kwargs
        self._cols: List[str] = []
        self._scores: Dict[str, float] = {}
        self._impl = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._cols = X.columns.tolist()
        _, X_array = normalize_feature_matrix(X)
        y_array = normalize_target_vector(y)
        try:
            from skrebate import MultiSURF as _MultiSURF
            if self.use_turf:
                from skrebate import TURF as _TURF
                base = _MultiSURF(n_jobs=self.n_jobs, **self.kwargs)
                self._impl = _TURF(base, pct=self.turf_pct).fit(X_array, y_array)
                importances = getattr(self._impl, "feature_importances_", None)
            else:
                self._impl = _MultiSURF(n_jobs=self.n_jobs, **self.kwargs).fit(X_array, y_array)
                importances = getattr(self._impl, "feature_importances_", None)
        except ModuleNotFoundError as e:
            raise Exception("MultiSURF requires the 'skrebate' package (pip install skrebate).") from e

        if importances is None:
            importances = np.zeros(len(self._cols))
        self._scores = {c: float(s) for c, s in zip(self._cols, importances)}
        return self

    def _ranked(self) -> List[str]:
        return sorted(self._cols, key=lambda c: self._scores.get(c, 0.0), reverse=True)

    def get_support_mask(self, *, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[bool]:
        if top_k is not None:
            keep = set(self._ranked()[:int(top_k)])
            return [c in keep for c in self._cols]
        if threshold is not None:
            return [self._scores.get(c, 0.0) >= float(threshold) for c in self._cols]
        return [True] * len(self._cols)

    def get_support_names(self, cols: List[str], *, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[str]:
        mask = self.get_support_mask(top_k=top_k, threshold=threshold)
        return [c for c, m in zip(self._cols, mask) if m]

    def transform(self, X: pd.DataFrame, *, top_k: Optional[int] = None, threshold: Optional[float] = None) -> pd.DataFrame:
        names = self.get_support_names(self._cols, top_k=top_k, threshold=threshold)
        return X.loc[:, names]

    def get_scores(self) -> Dict[str, float]:
        return dict(self._scores)

    def get_params(self) -> Dict[str, Any]:
        return {
            "n_neighbors": self.n_neighbors,
            "random_state": self.random_state,
            "use_turf": self.use_turf,
            "turf_pct": self.turf_pct,
            "n_jobs": self.n_jobs,
            **(self.kwargs or {})
        }
