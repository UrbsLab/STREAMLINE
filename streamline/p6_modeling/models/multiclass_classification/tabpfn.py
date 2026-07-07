from __future__ import annotations

import inspect
import os
from abc import ABC
from typing import Any, Dict

from streamline.p6_modeling.utils.submodels import MulticlassClassificationModel

try:
    # Local OSS package (not tabpfn-client)
    from tabpfn import TabPFNClassifier as _TabPFNClassifier
except Exception:  # pragma: no cover
    _TabPFNClassifier = None


def supported_kwargs(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter kwargs to those supported by callable_obj's signature (robust across TabPFN versions).
    """
    try:
        sig = inspect.signature(callable_obj)
    except Exception:
        return kwargs

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs

    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def tabpfn_ensemble_param(callable_obj) -> str:
    try:
        params = set(inspect.signature(callable_obj).parameters)
    except Exception:
        return "N_ensemble_configurations"
    if "n_estimators" in params:
        return "n_estimators"
    return "N_ensemble_configurations"


class TabPFNMultiClassClassifier(MulticlassClassificationModel, ABC):
    """
    TabPFN Multiclass Classifier (CPU-only).

    Notes:
      - Forces device='cpu'.
      - If allow_cpu_large_dataset=True, sets TABPFN_ALLOW_CPU_LARGE_DATASET=true (can be very slow).
    """

    model_name = "TabPFN"
    small_name = "TabPFN"
    color = "purple"

    def __init__(
        self,
        cv_folds: int = 3,
        scoring_metric: str = "balanced_accuracy",
        metric_direction: str = "maximize",
        random_state: int | None = None,
        cv=None,
        n_jobs=None,
        # TabPFN knobs
        n_ensemble_configurations: int = 32,
        fit_mode: str | None = None,
        # Env/config convenience
        model_cache_dir: str | None = None,
        allow_cpu_large_dataset: bool = False,
    ):
        if _TabPFNClassifier is None:
            raise ImportError("TabPFN is not installed. Install with: pip install tabpfn")

        if model_cache_dir:
            os.environ["TABPFN_MODEL_CACHE_DIR"] = str(model_cache_dir)

        if allow_cpu_large_dataset:
            os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "true"

        super().__init__(
            _TabPFNClassifier,
            self.model_name,
            cv_folds,
            scoring_metric,
            metric_direction,
            random_state,
            cv,
        )

        self.n_jobs = n_jobs  # unused by TabPFN; kept for interface consistency

        ensemble_param = tabpfn_ensemble_param(_TabPFNClassifier)
        self.param_grid: Dict[str, Any] = {
            ensemble_param: [8, 16, 32, 64],
            "fit_mode": ["fit_preprocessors", "fit_with_cache"],
            "device": ["cpu"],
        }

        self._base_params = {
            "device": "cpu",
            ensemble_param: int(n_ensemble_configurations),
        }
        if fit_mode is not None:
            self._base_params["fit_mode"] = fit_mode

        self._base_params = supported_kwargs(_TabPFNClassifier, self._base_params)

    def objective(self, trial, params: Dict[str, Any] | None = None):
        cand = dict(self._base_params)

        ensemble_param = tabpfn_ensemble_param(_TabPFNClassifier)
        ensemble_values = self.param_grid.get(ensemble_param) or self.param_grid.get("N_ensemble_configurations", [8])
        cand[ensemble_param] = trial.suggest_categorical(ensemble_param, ensemble_values)

        # Only keep fit_mode if your installed TabPFN supports it
        cand2 = dict(cand)
        cand2["fit_mode"] = trial.suggest_categorical("fit_mode", self.param_grid["fit_mode"])
        cand2 = supported_kwargs(_TabPFNClassifier, cand2)
        cand = cand2

        cand["device"] = "cpu"
        cand = supported_kwargs(_TabPFNClassifier, cand)

        self.params = cand
        return self.hyper_eval()
