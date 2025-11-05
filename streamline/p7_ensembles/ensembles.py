from __future__ import annotations
import os, re, pickle, logging
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
from streamline.p7_ensembles.utils import get_ensemble_by_id
from streamline.p6_modeling.modeling import ModelJob  # your P6 ModelJob

class EnsemblePhaseJob:
    def __init__(
        self,
        dataset_dir: str,            
        output_path: str,
        experiment_name: str,
        n_splits: int,
        outcome_label: str = "Class",
        instance_label: Optional[str] = None,
        ensembles: str = "hard_voting,soft_voting,stack_lr",
        base_model_filter: Optional[str] = None,   # CSV of base small_names
        scoring_metric: str = "balanced_accuracy",
        metric_direction: str = "maximize",
        stack_tune: bool = False,        # turn on Optuna for stacking meta only
        stack_trials: int = 30,
        stack_timeout: int = 600,
        n_trials: int = 1, timeout: int = 1,       # unused (no-opt), but ModelJob interface
        calibrate: bool = False, calibrate_method: str = "sigmoid", calibrate_cv: int = 5,
        random_state: Optional[int] = 0, n_jobs: Optional[int] = None,
    ):
        self.ds_dir = Path(dataset_dir)
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.n_splits = int(n_splits)
        self.outcome_label = outcome_label
        self.instance_label = instance_label
        self.ensemble_ids = [e.strip() for e in (ensembles or "").split(",") if e.strip()]
        self.base_filter = {s.strip() for s in (base_model_filter or "").split(",") if s.strip()} or None
        
        self.stack_tune = bool(stack_tune)
        self.stack_trials = int(stack_trials)
        self.stack_timeout = int(stack_timeout)

        # BaseModel kwargs shared by all ensembles
        self.model_kwargs = dict(
            cv_folds=3, scoring_metric=scoring_metric, metric_direction=metric_direction,
            random_state=random_state, cv=None, sampler=None, n_jobs=n_jobs
        )
        # Phase-6 calibration toggles are handled inside BaseModel.fit → we pass flags via attributes
        self.calibrate = bool(calibrate)
        self.calibrate_method = calibrate_method
        self.calibrate_cv = calibrate_cv

        # for ModelJob interface (we won’t tune, but we keep signature)
        self.n_trials = n_trials
        self.timeout = timeout

    def run(self):
        for cv in range(self.n_splits):
            base_ests = self._load_base_estimators(cv)
            if not base_ests:
                logging.warning(f"[P7] No base estimators for CV {cv} in {self.ds_dir}")
                continue

            # one ModelJob per ensemble (so saving/metrics mirror Phase-6)
            for ens_id in self.ensemble_ids:
                EnsCls = get_ensemble_by_id(ens_id)
                # pass tune for stack_*; it's ignored by vote_* classes
                model_obj = EnsCls(
                    base_estimators=base_ests,
                    tune=self.stack_tune,
                    **self.model_kwargs
                )

                # wire calibration flags directly into model_obj (your BaseModel handles them in fit)
                model_obj.calibrate = self.calibrate
                model_obj.calibrate_method = self.calibrate_method
                model_obj.calibrate_cv = self.calibrate_cv
                
                # If tuning stacks, use provided trials/timeout; else keep tiny defaults
                n_trials = self.stack_trials if self.stack_tune and ens_id.startswith("stack_") else 1
                timeout  = self.stack_timeout if self.stack_tune and ens_id.startswith("stack_") else 1

                mj = ModelJob(
                    full_path=str(self.ds_dir),
                    output_path=self.output_path,
                    experiment_name=self.experiment_name,
                    cv_count=cv,
                    outcome_label=self.outcome_label,
                    instance_label=self.instance_label,
                    scoring_metric=self.model_kwargs["scoring_metric"],
                    metric_direction=self.model_kwargs["metric_direction"],
                    n_trials=n_trials, timeout=timeout,
                    training_subsample=0, uniform_fi=False, save_plot=False,
                    random_state=self.model_kwargs["random_state"]
                )
                mj.run(model_obj)

    def _load_base_estimators(self, cv: int) -> List[Tuple[str, object]]:
        pm = self.ds_dir / "models" / "pickledModels"
        if not pm.exists():
            return []
        out: List[Tuple[str, object]] = []
        for fn in os.listdir(pm):
            if not fn.endswith(".pickle"):
                continue
            m = re.match(r"(.+?)_([0-9]+)\.pickle$", fn)
            if not m:
                continue
            small, fold = m.group(1), int(m.group(2))
            if fold != cv:
                continue
            if self.base_filter and small not in self.base_filter:
                continue
            with open(pm / fn, "rb") as f:
                est = pickle.load(f)
            out.append((small, est))
        return sorted(out, key=lambda t: t[0])
