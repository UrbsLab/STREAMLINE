# streamline/phases/p2_impute_scale/registry/median.py
from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from streamline.p2_impute_scale.utils.base_impute_scale import Imputer

REGISTRY: Dict[str, type] = {}

class MedianMap(Imputer):
    """
    Median-map for numeric columns (manual, dataframe-friendly).
    Mirrors your reference 'median_dict' pathway for quantitative features.
    """
    id = "median_map"

    def __init__(self):
        self._medians: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "MedianMap":
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                self._medians[c] = float(X[c].median(skipna=True))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = X.copy()
        for c, v in self._medians.items():
            if c in Xc.columns:
                Xc[c] = Xc[c].fillna(v)
        return Xc

    def get_params(self) -> Dict[str, Any]:
        return {"n_medians": len(self._medians)}

REGISTRY[MedianMap.id] = MedianMap
