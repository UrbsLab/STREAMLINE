from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import optuna

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from streamline.p6_modeling.utils.submodels import BinaryClassificationModel
from streamline.p7_ensembles.utils.stacking_model import StackingClassifier

def _base_list(pairs: List[Tuple[str, object]]):
    return [est for _, est in pairs]

class _StackingBase(BinaryClassificationModel):
    """
    Reusable stacking scaffold:
    - If tune=False → no-op optimize; fixed meta model
    - If tune=True  → Optuna objective tunes meta params only, using BaseModel.hyper_eval()
    """

    def __init__(self,
                 base_estimators: List[Tuple[str, object]],
                 tune: bool = False,
                 **kw):
        self._base_estimators = base_estimators
        self._tune = bool(tune)
        super().__init__(model=lambda: self._build(meta_params=None),
                         model_name=self.model_name, **kw)
        # Set param_grid only when tuning; BaseModel uses this to decide single vs sweep
        self.param_grid = self._param_grid() if self._tune else {}

    # ---- hooks every subclass must provide ---------------------------------
    def _default_meta(self):
        raise NotImplementedError

    def _param_grid(self) -> Dict[str, Any]:
        """Return Optuna search space description (keys only)."""
        return {}

    def _trial_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """Translate _param_grid into concrete trial params."""
        return {}
    
    def fit(self, x_train, y_train, n_trails=100, timeout=450, feature_names=None):
        """Optimize → fit → (optional) calibrate for classifiers."""
        self.optimize(x_train, y_train, n_trails, timeout, feature_names)
        self.model.fit(x_train, y_train)

    # ---- Stacking builder ---------------------------------------------------
    def _build(self, meta_params: Optional[Dict[str, Any]]):
        # build meta clf
        if meta_params is None:
            meta = self._default_meta()
        else:
            meta = self._default_meta().__class__(**meta_params)

        return StackingClassifier(
            classifiers=_base_list(self._base_estimators),
            meta_classifier=meta,
            use_probas=False,
            use_clones=True,
            fit_base_estimators=False,
        )

    # ---- BaseModel integration ----------------------------------------------
    def optimize(self, x_train, y_train, n_trails, timeout, feature_names=None):
        """If tune is off → just instantiate once. Else → run normal Optuna path."""
        self.x_train = x_train
        self.y_train = y_train
        if not self._tune:
            # single-fit path
            if callable(self.model):
                self.model = self.model()
            self.params = {}
            return
        # tuning path → let BaseModel handle study lifecycle via objective()
        super().optimize(x_train, y_train, n_trails, timeout, feature_names)

    def objective(self, trial: optuna.trial.Trial, params=None):
        # produce trial params for the meta-classifier only
        mp = self._trial_params(trial)
        # set model to a fresh StackingClassifier(meta=mp)
        self.model = self._build(meta_params=mp)
        self.params = mp  # for logging/export
        return self.hyper_eval()

# ------------------ concrete stackers ---------------------------------------

class StackLR(_StackingBase):
    id = "stack_lr"
    model_name = "StackingLogReg"
    small_name = "STK_LR"

    def _default_meta(self):
        return LogisticRegression(solver="lbfgs", max_iter=1000)

    def _param_grid(self):
        # keys only; values/ranges provided in _trial_params
        return {"C": (), "penalty": (), "solver": (), "max_iter": (), "class_weight": ()}

    def _trial_params(self, trial):
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga", "newton-cg", "sag"])
        params = {
            "C": trial.suggest_float("C", 1e-4, 1e3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 2000, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "solver": solver,
        }
        # penalty depends on solver
        if solver in ("lbfgs", "newton-cg", "sag"):
            params["penalty"] = "l2"
        elif solver == "liblinear":
            params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])
        else:  # saga
            params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])
        return params

class StackDT(_StackingBase):
    id = "stack_dt"
    model_name = "Stacking DecisionTree"
    small_name = "STK_DT"

    def _default_meta(self):
        return DecisionTreeClassifier(random_state=self.random_state)

    def _param_grid(self):
        return {"criterion": (), "splitter": (), "max_depth": (), "min_samples_split": (),
                "min_samples_leaf": (), "max_features": (), "class_weight": ()}

    def _trial_params(self, trial):
        return {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
            "max_depth": trial.suggest_int("max_depth", 1, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

class StackRF(_StackingBase):
    id = "stack_rf"
    model_name = "Stacking RandomForest"
    small_name = "STK_RF"

    def _default_meta(self):
        return RandomForestClassifier(n_estimators=200, random_state=self.random_state)

    def _param_grid(self):
        return {"n_estimators": (), "criterion": (), "max_depth": (), "min_samples_split": (),
                "min_samples_leaf": (), "max_features": (), "bootstrap": (), "oob_score": (),
                "class_weight": ()}

    def _trial_params(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000, log=True),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "max_depth": trial.suggest_int("max_depth", 1, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "oob_score": trial.suggest_categorical("oob_score", [False, True]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
