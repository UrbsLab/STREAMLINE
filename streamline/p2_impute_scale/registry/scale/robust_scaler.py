from __future__ import annotations
from typing import Dict, Any
import pandas as pd
from sklearn.preprocessing import RobustScaler as _Robust

class Robust:
    id = "robust"
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.impl = _Robust(**kwargs)
        self._cols = None
    def fit(self, X: pd.DataFrame, y=None):
        self._cols = X.select_dtypes(include=["number"]).columns
        self.impl.fit(X[self._cols])
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._cols is None:
            return X
        Xt = X.copy()
        Xt.loc[:, self._cols] = self.impl.transform(Xt[self._cols])
        return Xt
    def get_params(self) -> Dict[str, Any]:
        return dict(self.kwargs)
