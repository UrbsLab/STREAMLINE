# streamline/phases/p4_feature_selection/interface.py
from __future__ import annotations
from typing import Protocol, Dict, Any, Optional, List
import pandas as pd

class Selector(Protocol):
    id: str
    def __init__(self, **params: Any): 
        pass
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Selector": 
        pass
    def transform(self, X: pd.DataFrame, *, top_k: Optional[int] = None, threshold: Optional[float] = None) -> pd.DataFrame: 
        pass
    def get_support_mask(self, *, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[bool]: 
        pass
    def get_support_names(self, cols: List[str], *, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[str]: 
        pass
    def get_scores(self) -> Dict[str, float]: 
        pass
    def get_params(self) -> Dict[str, Any]: 
        pass
