from __future__ import annotations
import os
import json
import logging
from typing import List, Optional, Dict, Any
from streamline.p6_modeling.utils.modeljob import ModelJob
from streamline.p6_modeling.utils.loader import load_model_classes, get_model_by_id

def _csv_to_list(v):
    if v is None: return None
    if isinstance(v, list): return v
    return [x.strip() for x in str(v).split(",") if x.strip()]

class ModelingPhaseJob:
    def __init__(
        self,
        *,
        dataset_dir: str,                 # <output>/<experiment>/<dataset>
        outcome_label: str = "Class",
        model_type: str = "Binary",     # "Binary" | "Multiclass" | "Regression"
        instance_label: Optional[str] = None,
        n_splits: int = 10,
        models: List[str] | str | None = None,     # CSV or list; if None -> auto discover
        model_params_json: Optional[str] = None,
        # calibration
        calibrate: bool = False,
        calibrate_method: str = "sigmoid",
        calibrate_cv: int = 5,
        # legacy knobs (forwarded to ModelJob)
        output_path: Optional[str] = None,
        experiment_name: Optional[str] = None,
        scoring_metric: str = "balanced_accuracy",
        metric_direction: str = "maximize",
        n_trials: int = 200,
        timeout: int = 900,
        training_subsample: int = 0,
        uniform_fi: bool = False,
        save_plot: bool = False,
        random_state: Optional[int] = None,
    ):
        self.dataset_dir = dataset_dir
        self.dataset_name = os.path.basename(dataset_dir.rstrip("/"))
        self.outcome_label = outcome_label
        self.model_type = model_type
        self.instance_label = instance_label
        self.n_splits = int(n_splits)
        self.models = _csv_to_list(models)

        self.calibrate = bool(calibrate)
        self.calibrate_method = calibrate_method
        self.calibrate_cv = int(calibrate_cv)

        exp_dir = os.path.dirname(dataset_dir.rstrip("/"))
        self.output_path = output_path or os.path.dirname(exp_dir)
        self.experiment_name = experiment_name or os.path.basename(exp_dir)

        self.scoring_metric = scoring_metric
        self.metric_direction = metric_direction
        self.n_trials = int(n_trials)
        self.timeout = int(timeout)
        self.training_subsample = int(training_subsample)
        self.uniform_fi = bool(uniform_fi)
        self.save_plot = bool(save_plot)
        self.random_state = random_state
        
                # --- NEW: parse model_params_json ---
        self.model_params: Dict[str, Dict[str, Any]] = {}
        if model_params_json:
            try:
                parsed = json.loads(model_params_json)
                if isinstance(parsed, dict):
                    # normalize keys to lowercase for matching
                    self.model_params = {
                        str(k).lower(): (v if isinstance(v, dict) else {})
                        for k, v in parsed.items()
                    }
                else:
                    logging.warning(
                        "[P6] model_params_json must be a JSON object mapping model ids → dicts; got %r",
                        type(parsed),
                    )
            except Exception as e:
                logging.error("[P6] Failed to parse model_params_json: %r", e)
                raise e

    def run(self):
        # Discover models if none provided
        if not self.models:
            model_classes = load_model_classes(self.model_type)
        else:
            model_classes = [get_model_by_id(self.model_type, mid) for mid in self.models]

        for ModelCls in model_classes:
            for cv_idx in range(self.n_splits):
                mj = ModelJob(
                    full_path=self.dataset_dir,
                    output_path=self.output_path,
                    experiment_name=self.experiment_name,
                    cv_count=cv_idx,
                    outcome_label=self.outcome_label,
                    instance_label=self.instance_label,
                    scoring_metric=self.scoring_metric,
                    metric_direction=self.metric_direction,
                    n_trials=self.n_trials,
                    timeout=self.timeout,
                    training_subsample=self.training_subsample,
                    uniform_fi=self.uniform_fi,
                    save_plot=self.save_plot,
                    random_state=self.random_state,
                    calibrate=self.calibrate,
                    calibrate_method=self.calibrate_method,
                    calibrate_cv=self.calibrate_cv,
                )

                model = ModelCls(
                    random_state=self.random_state,
                    n_jobs=-1,
                    scoring_metric=self.scoring_metric,
                    metric_direction=self.metric_direction,
                    # calibrate=self.calibrate,              # ← NEW
                    # calibrate_method=self.calibrate_method,
                    # calibrate_cv=self.calibrate_cv,
                )
                
                overrides = self._get_model_overrides(ModelCls)
                for attr, value in overrides.items():
                    setattr(model, attr, value)
                
                mj.run(model)
                

        # mark complete
        jc_dir = os.path.join(os.path.dirname(self.dataset_dir), "jobsCompleted")
        os.makedirs(jc_dir, exist_ok=True)
        with open(os.path.join(jc_dir, f"job_modeling_{self.dataset_name}.txt"), "w") as f:
            f.write("complete")
            
    def _get_model_overrides(self, ModelCls) -> Dict[str, Any]:
        """
        Look up overrides from model_params_json using either small_name or model_name.
        JSON keys are matched case-insensitively.
        Example JSON:
        {
            "lr": {"param_grid": {"C": [0.01, 0.1, 1.0]}},
            "Logistic Regression": {"n_trials": 100}
        }
        """
        overrides: Dict[str, Any] = {}
        small = getattr(ModelCls, "small_name", "").lower()
        mname = getattr(ModelCls, "model_name", "").lower()

        for key in (small, mname):
            if key and key in self.model_params:
                cfg = self.model_params[key]
                if isinstance(cfg, dict):
                    overrides.update(cfg)

        return overrides
