from __future__ import annotations
from typing import Dict, Any
import pandas as pd
from sklearn.impute import IterativeImputer
from streamline.p2_impute_scale.utils.base_impute_scale import Imputer

class Iterative(Imputer):
    id = "iterative"
    def __init__(self, random_state: int = 0, max_iter: int = 30):
        self.random_state = random_state
        self.max_iter = max_iter
        self._impl = IterativeImputer(random_state=random_state, max_iter=max_iter)
    def fit(self, X: pd.DataFrame, y=None) -> "Iterative":
        self._impl.fit(X); return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xt = self._impl.transform(X)
        return pd.DataFrame(Xt, index=X.index, columns=X.columns)
    def get_params(self) -> Dict[str, Any]:
        return {"random_state": self.random_state, "max_iter": self.max_iter}
