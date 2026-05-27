from __future__ import annotations
import os
import json
import logging
import pickle
from typing import List, Optional, Dict, Any
from streamline.p6_modeling.utils.modeljob import ModelJob
from streamline.p6_modeling.utils.loader import load_model_classes, get_model_by_id
from streamline.p6_modeling.utils.categorical import normalize_model_id, parse_model_id_csv

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
        bypass_one_hot_for_native_models: bool = True,
        native_categorical_models: List[str] | str | None = "CGB",
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
        self.bypass_one_hot_for_native_models = bool(bypass_one_hot_for_native_models)
        self.native_categorical_models = native_categorical_models
        self.native_categorical_model_ids = parse_model_id_csv(
            native_categorical_models,
            default=("CGB", "Category Gradient Boosting"),
        )
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
        one_hot_disabled = self._p1_one_hot_disabled()
        if one_hot_disabled and not self.bypass_one_hot_for_native_models:
            raise ValueError(
                "P1 feature metadata shows one_hot_encoding=False. Phase 6 must use "
                "native categorical models in this mode; keep "
                "--bypass_one_hot_for_native_models enabled or rerun P1 with "
                "--one_hot_encoding 1."
            )

        # Discover models if none provided
        if not self.models:
            model_classes = load_model_classes(self.model_type)
            if one_hot_disabled:
                original_count = len(model_classes)
                model_classes = [
                    cls for cls in model_classes
                    if self._model_class_allows_native_categorical(cls)
                ]
                if not model_classes:
                    raise ValueError(
                        "P1 feature metadata shows one_hot_encoding=False, but no "
                        f"{self.model_type} models match native_categorical_models="
                        f"{sorted(self.native_categorical_model_ids)}."
                    )
                logging.info(
                    "[P6] P1 one_hot_encoding=False for %s; limiting auto-discovered "
                    "models from %s to native categorical models: %s",
                    self.dataset_name,
                    original_count,
                    ", ".join(self._model_label(cls) for cls in model_classes),
                )
        else:
            model_classes = [get_model_by_id(self.model_type, mid) for mid in self.models]
            if one_hot_disabled:
                unsupported = [
                    cls for cls in model_classes
                    if not self._model_class_allows_native_categorical(cls)
                ]
                if unsupported:
                    raise ValueError(
                        "P1 feature metadata shows one_hot_encoding=False, so Phase 6 "
                        "can only run models listed in --native_categorical_models. "
                        "Unsupported requested model(s): "
                        + ", ".join(self._model_label(cls) for cls in unsupported)
                        + ". Native categorical model ids: "
                        + ", ".join(sorted(self.native_categorical_model_ids))
                    )

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
                    bypass_one_hot_for_native_models=self.bypass_one_hot_for_native_models,
                    native_categorical_models=self.native_categorical_models,
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

    def _load_feature_meta(self) -> Dict[str, Any]:
        exploratory = os.path.join(self.dataset_dir, "exploratory")
        meta_pickle = os.path.join(exploratory, "feature_meta.pickle")
        meta_json = os.path.join(exploratory, "feature_meta.json")
        try:
            if os.path.exists(meta_pickle):
                with open(meta_pickle, "rb") as f:
                    payload = pickle.load(f)
                    return payload if isinstance(payload, dict) else {}
            if os.path.exists(meta_json):
                with open(meta_json, "r") as f:
                    payload = json.load(f)
                    return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            logging.warning("[P6] Could not read feature metadata for %s: %s", self.dataset_name, exc)
        return {}

    def _p1_one_hot_disabled(self) -> bool:
        meta = self._load_feature_meta()
        if "one_hot" not in meta:
            return False
        value = meta.get("one_hot")
        if isinstance(value, bool):
            return not value
        return str(value).strip().lower() in {"0", "false", "f", "no", "n"}

    def _model_class_allows_native_categorical(self, ModelCls) -> bool:
        ids = {
            normalize_model_id(getattr(ModelCls, "small_name", "")),
            normalize_model_id(getattr(ModelCls, "model_name", "")),
        }
        return bool(ids.intersection(self.native_categorical_model_ids))

    @staticmethod
    def _model_label(ModelCls) -> str:
        small = getattr(ModelCls, "small_name", "")
        name = getattr(ModelCls, "model_name", "")
        if small and name:
            return f"{small} ({name})"
        return small or name or str(ModelCls)
            
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
