from __future__ import annotations

import inspect
import os
from abc import ABC
from typing import Any, Dict

from streamline.p6_modeling.utils.submodels import RegressionModel

try:
    from tabpfn import TabPFNRegressor as _TabPFNRegressor
except Exception:  # pragma: no cover
    _TabPFNRegressor = None


def _supported_kwargs(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
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


class TabPFNRegressor(RegressionModel, ABC):
    """
    TabPFN Regressor (CPU-only).

    Notes:
      - Forces device='cpu'.
      - If allow_cpu_large_dataset=True, sets TABPFN_ALLOW_CPU_LARGE_DATASET=true (can be very slow).
    """

    model_name = "TabPFN (Regression)"
    small_name = "TabPFN-R"
    color = "purple"

    def __init__(
        self,
        cv_folds: int = 3,
        scoring_metric: str = "neg_mean_squared_error",
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
        if _TabPFNRegressor is None:
            raise ImportError("TabPFN is not installed. Install with: pip install tabpfn")

        if model_cache_dir:
            os.environ["TABPFN_MODEL_CACHE_DIR"] = str(model_cache_dir)

        if allow_cpu_large_dataset:
            os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "true"

        super().__init__(
            _TabPFNRegressor,
            self.model_name,
            cv_folds,
            scoring_metric,
            metric_direction,
            random_state,
            cv,
        )

        self.n_jobs = n_jobs  # unused by TabPFN; kept for interface consistency

        self.param_grid: Dict[str, Any] = {
            "N_ensemble_configurations": [8, 16, 32, 64],
            "fit_mode": ["fit_direct", "fit_with_cache"],
            "device": ["cpu"],
        }

        self._base_params = {
            "device": "cpu",
            "N_ensemble_configurations": int(n_ensemble_configurations),
        }
        if fit_mode is not None:
            self._base_params["fit_mode"] = fit_mode

        self._base_params = _supported_kwargs(_TabPFNRegressor, self._base_params)

    def objective(self, trial, params: Dict[str, Any] | None = None):
        cand = dict(self._base_params)

        cand["N_ensemble_configurations"] = trial.suggest_categorical(
            "N_ensemble_configurations", self.param_grid["N_ensemble_configurations"]
        )

        # Only keep fit_mode if your installed TabPFN supports it
        cand2 = dict(cand)
        cand2["fit_mode"] = trial.suggest_categorical("fit_mode", self.param_grid["fit_mode"])
        cand2 = _supported_kwargs(_TabPFNRegressor, cand2)
        cand = cand2

        cand["device"] = "cpu"
        cand = _supported_kwargs(_TabPFNRegressor, cand)

        self.params = cand
        return self.hyper_eval()
