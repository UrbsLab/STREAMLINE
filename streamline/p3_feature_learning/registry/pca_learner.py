# streamline/phases/p3_feature_learning/learners/pca.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd
from sklearn.decomposition import PCA

class PCALearner:
    id = "pca"
    def __init__(self, n_components: int | float | None = None, random_state: Optional[int] = None, **kwargs):
        self.n_components = n_components
        self.random_state = random_state
        self.kwargs = kwargs
        self._cols: Optional[pd.Index] = None
        self._impl: Optional[PCA] = None

    def fit(self, X: pd.DataFrame, y=None) -> "PCALearner":
        # numeric-only by default
        self._cols = X.select_dtypes(include=["number"]).columns
        self._impl = PCA(n_components=self.n_components, random_state=self.random_state, **self.kwargs)
        self._impl.fit(X[self._cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._impl is None or self._cols is None:
            raise RuntimeError("PCALearner must be fit before transform.")
        Xt = self._impl.transform(X[self._cols])
        # column names filled by job using get_feature_names (needs n_components)
        return pd.DataFrame(Xt, index=X.index)

    def get_feature_names(self, input_cols: List[str], namespace: str) -> List[str]:
        # Determine output dimensionality
        if self._impl is None:
            # fallback on n_components; if None, PCA would infer min(n_samples,n_features)
            # the job will re-name using transformed shape
            if isinstance(self.n_components, int) and self.n_components > 0:
                out = self.n_components
            else:
                out = len(input_cols)
        else:
            out = self._impl.n_components_
        return [f"{namespace}_PC{i+1}" for i in range(out)]

    def get_params(self) -> Dict[str, Any]:
        p = {"n_components": self.n_components, "random_state": self.random_state}
        p.update(self.kwargs or {})
        return p
