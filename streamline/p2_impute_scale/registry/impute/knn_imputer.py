from __future__ import annotations
from typing import Dict, Any
import pandas as pd
from sklearn.impute import KNNImputer
from streamline.p2_impute_scale.utils.base_impute_scale import Imputer

class KNN(Imputer):
    id = "knn"
    def __init__(self, n_neighbors: int = 5, weights: str = "uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._impl = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    def fit(self, X: pd.DataFrame, y=None) -> "KNN":
        self._impl.fit(X); return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xt = self._impl.transform(X)
        return pd.DataFrame(Xt, index=X.index, columns=X.columns)
    def get_params(self) -> Dict[str, Any]:
        return {"n_neighbors": self.n_neighbors, "weights": self.weights}
