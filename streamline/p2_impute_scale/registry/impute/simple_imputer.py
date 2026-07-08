# streamline/phases/p2_impute_scale/registry/simple.py
from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from streamline.p2_impute_scale.utils.base_impute_scale import Imputer

class Simple(Imputer):
    """
    Wrapper around sklearn's SimpleImputer.
    strategy: mean | median | most_frequent | constant
    """
    id = "simple"

    def __init__(self, strategy: str = "median", fill_value: Optional[float] = None):
        self.strategy = strategy
        self.fill_value = fill_value
        self._impl = SimpleImputer(strategy=strategy, fill_value=fill_value)

    def fit(self, X: pd.DataFrame, y=None) -> "Simple":
        self._impl.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xt = self._impl.transform(X)
        return pd.DataFrame(Xt, index=X.index, columns=X.columns)

    def get_params(self) -> Dict[str, Any]:
        return {"strategy": self.strategy, "fill_value": self.fill_value}
