from __future__ import annotations

import inspect
import os
from abc import ABC
from typing import Any, Dict

from streamline.p6_modeling.utils.submodels import BinaryClassificationModel

try:
    # Local OSS package (not the hosted tabpfn-client)
    from tabpfn import TabPFNClassifier as _TabPFNClassifier
except Exception:  # pragma: no cover
    _TabPFNClassifier = None


def _supported_kwargs(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter kwargs to those supported by callable_obj's signature (robust across TabPFN versions).
    """
    try:
        sig = inspect.signature(callable_obj)
    except Exception:
        return kwargs

    # If **kwargs is present, no need to filter.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs

    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


class TabPFNClassifier(BinaryClassificationModel, ABC):
    """
    TabPFN (CPU-only) wrapper.

    Notes:
      - Forces device='cpu' always.
      - TabPFN is designed for small/medium tabular problems; CPU can be slow.
      - If you must run CPU on larger datasets, you can set allow_cpu_large_dataset=True
        which sets TABPFN_ALLOW_CPU_LARGE_DATASET=true (very slow).
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
        # TabPFN knobs (kept intentionally small)
        n_ensemble_configurations: int = 32,
        fit_mode: str | None = None,
        # Environment/config convenience
        model_cache_dir: str | None = None,
        allow_cpu_large_dataset: bool = False,
    ):
        if _TabPFNClassifier is None:
            raise ImportError(
                "TabPFN is not installed. Install the local package with: pip install tabpfn"
            )

        # Optional caching dir (TabPFN uses env vars for cache location in newer versions)
        if model_cache_dir:
            os.environ["TABPFN_MODEL_CACHE_DIR"] = str(model_cache_dir)

        # Optional override for CPU limitation on larger datasets
        if allow_cpu_large_dataset:
            os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "true"

        super().__init__(
            _TabPFNClassifier,
            "TabPFN",
            cv_folds,
            scoring_metric,
            metric_direction,
            random_state,
            cv,
        )

        self.n_jobs = n_jobs  # unused by TabPFN; kept for interface consistency
        self.small_name = "TabPFN"
        self.color = "purple"

        # Build a conservative param grid. We'll filter by the installed TabPFN signature anyway.
        self.param_grid: Dict[str, Any] = {
            # Common across versions (documented for sklearn-style interface)
            "N_ensemble_configurations": [8, 16, 32, 64],
            # Some versions add fit_mode; we only use it if supported.
            "fit_mode": ["fit_direct", "fit_with_cache"],
            # Always CPU
            "device": ["cpu"],
        }

        self._base_params = {
            "device": "cpu",
            "N_ensemble_configurations": int(n_ensemble_configurations),
        }
        if fit_mode is not None:
            self._base_params["fit_mode"] = fit_mode

        # Ensure base params are compatible with the installed TabPFN version
        self._base_params = _supported_kwargs(_TabPFNClassifier, self._base_params)

    def objective(self, trial, params: Dict[str, Any] | None = None):
        # Start from CPU-only base params
        cand = dict(self._base_params)

        # Tune ensemble size
        cand["N_ensemble_configurations"] = trial.suggest_categorical(
            "N_ensemble_configurations", self.param_grid["N_ensemble_configurations"]
        )

        # Tune fit_mode only if supported by the installed TabPFN signature
        cand_with_fit_mode = dict(cand)
        cand_with_fit_mode["fit_mode"] = trial.suggest_categorical(
            "fit_mode", self.param_grid["fit_mode"]
        )
        cand_with_fit_mode = _supported_kwargs(_TabPFNClassifier, cand_with_fit_mode)

        # If fit_mode got filtered out, don't record it (keeps Optuna tidy)
        cand = cand_with_fit_mode

        # Always force CPU (even if someone tries to pass something in)
        cand["device"] = "cpu"
        cand = _supported_kwargs(_TabPFNClassifier, cand)

        self.params = cand
        mean_cv_score = self.hyper_eval()
        return mean_cv_score
