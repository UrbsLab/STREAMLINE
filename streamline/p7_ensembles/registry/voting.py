from __future__ import annotations
from typing import List, Tuple
from mlxtend.classifier import EnsembleVoteClassifier
from streamline.p6_modeling.utils.submodels import BinaryClassificationModel

class HardVoting(BinaryClassificationModel):
    id = "hard_voting"
    model_name = "Hard Ensemble Voting"
    small_name = "HEV"

    def __init__(self, base_estimators: List[Tuple[str, object]], **kw):
        # BaseModel expects a callable or None; we’ll pass a lambda that returns the estimator
        super().__init__(model=lambda: EnsembleVoteClassifier(
            clfs=[m for _, m in base_estimators],
            fit_base_estimators=False, voting='hard', use_clones=True
        ), model_name=self.model_name, **kw)
        self.param_grid = {}  # no hyperopt

    def optimize(self, *args, **kwargs):
        # No hyper-parameters; ensure self.model is a concrete estimator
        if callable(self.model):
            self.model = self.model()

class SoftVoting(BinaryClassificationModel):
    id = "soft_voting"
    model_name = "Soft Ensemble Voting"
    small_name = "SEV"

    def __init__(self, base_estimators: List[Tuple[str, object]], **kw):
        super().__init__(model=lambda: EnsembleVoteClassifier(
            clfs=[m for _, m in base_estimators],
            fit_base_estimators=False, voting='soft', use_clones=True
        ), model_name=self.model_name, **kw)
        self.param_grid = {}

    def optimize(self, *args, **kwargs):
        if callable(self.model):
            self.model = self.model()
