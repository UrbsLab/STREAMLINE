from __future__ import annotations
import os
import json
import logging
import pickle
import warnings
from typing import List, Optional, Dict, Any
from streamline.p6_modeling.utils.modeljob import ModelJob
from streamline.p6_modeling.utils.loader import load_default_model_classes, get_model_by_id
from streamline.p6_modeling.utils.categorical import (
    NATIVE_CATEGORICAL_MODEL_IDS_DEFAULT,
    NATIVE_CATEGORICAL_MODELS_DEFAULT,
    normalize_model_id,
    parse_model_id_csv,
)

def csv_to_list(v):
    if v is None: return None
    if isinstance(v, list): return v
    return [x.strip() for x in str(v).split(",") if x.strip()]


def parse_model_params_json(model_params_json) -> Dict[str, Dict[str, Any]]:
    model_params: Dict[str, Dict[str, Any]] = {}
    if not model_params_json:
        return model_params
    try:
        parsed = model_params_json if isinstance(model_params_json, dict) else json.loads(model_params_json)
    except Exception as exc:
        logging.error("[P6] Failed to parse model_params_json: %r", exc)
        raise

    if isinstance(parsed, dict):
        return {
            str(key).lower(): (value if isinstance(value, dict) else {})
            for key, value in parsed.items()
        }
    logging.warning(
        "[P6] model_params_json must be a JSON object mapping model ids → dicts; got %r",
        type(parsed),
    )
    return model_params


def load_feature_metadata(dataset_dir: str, dataset_name: str | None = None) -> Dict[str, Any]:
    exploratory = os.path.join(dataset_dir, "exploratory")
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
        label = dataset_name or os.path.basename(dataset_dir.rstrip("/"))
        logging.warning("[P6] Could not read feature metadata for %s: %s", label, exc)
    return {}


def p1_one_hot_is_disabled(feature_meta: Dict[str, Any]) -> bool:
    if "one_hot" not in feature_meta:
        return False
    value = feature_meta.get("one_hot")
    if isinstance(value, bool):
        return not value
    return str(value).strip().lower() in {"0", "false", "f", "no", "n"}


def model_class_label(ModelCls) -> str:
    small = getattr(ModelCls, "small_name", "")
    name = getattr(ModelCls, "model_name", "")
    if small and name:
        return f"{small} ({name})"
    return small or name or str(ModelCls)


def model_class_is_tabpfn(ModelCls) -> bool:
    ids = {
        normalize_model_id(getattr(ModelCls, "small_name", "")),
        normalize_model_id(getattr(ModelCls, "model_name", "")),
    }
    return any("tabpfn" in value for value in ids)


def model_class_supports_native_categorical(ModelCls, native_categorical_model_ids) -> bool:
    ids = {
        normalize_model_id(getattr(ModelCls, "small_name", "")),
        normalize_model_id(getattr(ModelCls, "model_name", "")),
    }
    return bool(ids.intersection(native_categorical_model_ids))


def skip_tabpfn_models_without_token(model_classes, dataset_name: str):
    if os.environ.get("TABPFN_TOKEN"):
        return model_classes

    kept = []
    skipped = []
    for ModelCls in model_classes:
        if model_class_is_tabpfn(ModelCls):
            skipped.append(model_class_label(ModelCls))
        else:
            kept.append(ModelCls)

    if skipped:
        message = (
            "WARNING: TABPFN_TOKEN is not set, so Phase 6 will skip "
            f"TabPFN model fitting for {', '.join(skipped)}. HEROS and "
            "other requested non-TabPFN models will still run. See "
            "docs/source/tabpfn_token.md."
        )
        logging.warning(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    if not kept:
        logging.warning(
            "[P6] No models remain for %s after skipping token-gated models.",
            dataset_name,
        )
    return kept


def resolve_model_classes_for_dataset(
    *,
    dataset_dir: str,
    model_type: str,
    models: List[str] | str | None,
    bypass_one_hot_for_native_models: bool,
    native_categorical_models: List[str] | str | None,
):
    dataset_name = os.path.basename(dataset_dir.rstrip("/"))
    requested_models = csv_to_list(models)
    native_model_ids = parse_model_id_csv(
        native_categorical_models,
        default=NATIVE_CATEGORICAL_MODEL_IDS_DEFAULT,
    )
    one_hot_disabled = p1_one_hot_is_disabled(load_feature_metadata(dataset_dir, dataset_name))
    if one_hot_disabled and not bypass_one_hot_for_native_models:
        raise ValueError(
            "P1 feature metadata shows one_hot_encoding=False. Phase 6 must use "
            "native categorical models in this mode; keep "
            "--bypass_one_hot_for_native_models enabled or rerun P1 with "
            "--one_hot_encoding 1."
        )

    if requested_models:
        model_classes = [get_model_by_id(model_type, model_id) for model_id in requested_models]
        if one_hot_disabled:
            unsupported = [
                cls for cls in model_classes
                if not model_class_supports_native_categorical(cls, native_model_ids)
            ]
            if unsupported:
                raise ValueError(
                    "P1 feature metadata shows one_hot_encoding=False, so Phase 6 "
                    "can only run models listed in --native_categorical_models. "
                    "Unsupported requested model(s): "
                    + ", ".join(model_class_label(cls) for cls in unsupported)
                    + ". Native categorical model ids: "
                    + ", ".join(sorted(native_model_ids))
                )
    else:
        model_classes = load_default_model_classes(model_type)
        if one_hot_disabled:
            original_count = len(model_classes)
            model_classes = [
                cls for cls in model_classes
                if model_class_supports_native_categorical(cls, native_model_ids)
            ]
            if not model_classes:
                raise ValueError(
                    "P1 feature metadata shows one_hot_encoding=False, but no "
                    f"{model_type} models match native_categorical_models="
                    f"{sorted(native_model_ids)}."
                )
            logging.info(
                "[P6] P1 one_hot_encoding=False for %s; limiting auto-discovered "
                "models from %s to native categorical models: %s",
                dataset_name,
                original_count,
                ", ".join(model_class_label(cls) for cls in model_classes),
            )

    return skip_tabpfn_models_without_token(model_classes, dataset_name)


def model_overrides_for_class(ModelCls, model_params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    small = getattr(ModelCls, "small_name", "").lower()
    model_name = getattr(ModelCls, "model_name", "").lower()

    for key in (small, model_name):
        if key and key in model_params:
            config = model_params[key]
            if isinstance(config, dict):
                overrides.update(config)

    return overrides


def create_model_instance(ModelCls, *, random_state, scoring_metric, metric_direction, model_params):
    model = ModelCls(
        random_state=random_state,
        n_jobs=None,
        scoring_metric=scoring_metric,
        metric_direction=metric_direction,
    )

    for attr, value in model_overrides_for_class(ModelCls, model_params).items():
        setattr(model, attr, value)

    return model


def model_class_matching_id(model_classes, model_id: str):
    requested = str(model_id or "")
    for ModelCls in model_classes:
        if requested in {
            getattr(ModelCls, "small_name", ""),
            getattr(ModelCls, "model_name", ""),
        }:
            return ModelCls
    return None


def model_cv_flag_path(dataset_dir: str, model_small_name: str, cv_idx: int):
    dataset_name = os.path.basename(dataset_dir.rstrip("/"))
    return os.path.join(
        os.path.dirname(dataset_dir),
        "jobsCompleted",
        f"job_model_{dataset_name}_{cv_idx}_{model_small_name}.txt",
    )


def mark_modeling_phase_complete(dataset_dir: str):
    dataset_name = os.path.basename(dataset_dir.rstrip("/"))
    jobs_completed_dir = os.path.join(os.path.dirname(dataset_dir), "jobsCompleted")
    os.makedirs(jobs_completed_dir, exist_ok=True)
    with open(os.path.join(jobs_completed_dir, f"job_modeling_{dataset_name}.txt"), "w") as f:
        f.write("complete")


def mark_modeling_phase_complete_if_all_model_cv_jobs_finished(dataset_dir: str, model_classes, n_splits: int):
    expected_flags = [
        model_cv_flag_path(dataset_dir, getattr(ModelCls, "small_name", ""), cv_idx)
        for ModelCls in model_classes
        for cv_idx in range(int(n_splits))
    ]
    if expected_flags and all(os.path.exists(path) for path in expected_flags):
        mark_modeling_phase_complete(dataset_dir)


class ModelingPhaseJob:
    def __init__(
        self,
        *,
        dataset_dir: str,                 # <output>/<experiment>/<dataset>
        outcome_label: str = "Class",
        model_type: str = "Binary",     # "Binary" | "Multiclass" | "Regression"
        instance_label: Optional[str] = None,
        n_splits: int = 10,
        models: List[str] | str | None = None,     # CSV or list; if None -> auto-discover defaults
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
        native_categorical_models: List[str] | str | None = NATIVE_CATEGORICAL_MODELS_DEFAULT,
    ):
        self.dataset_dir = dataset_dir
        self.dataset_name = os.path.basename(dataset_dir.rstrip("/"))
        self.outcome_label = outcome_label
        self.model_type = model_type
        self.instance_label = instance_label
        self.n_splits = int(n_splits)
        self.models = csv_to_list(models)

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
            default=NATIVE_CATEGORICAL_MODEL_IDS_DEFAULT,
        )
        self.resolved_model_classes = None
        self.model_params = parse_model_params_json(model_params_json)

    def run_all_model_cv_jobs(self):
        for model_job, model in self.create_model_cv_executions():
            model_job.run(model)

        self.mark_phase_complete()

    def resolve_model_classes(self):
        if self.resolved_model_classes is not None:
            return list(self.resolved_model_classes)

        self.resolved_model_classes = resolve_model_classes_for_dataset(
            dataset_dir=self.dataset_dir,
            model_type=self.model_type,
            models=self.models,
            bypass_one_hot_for_native_models=self.bypass_one_hot_for_native_models,
            native_categorical_models=self.native_categorical_models,
        )
        return list(self.resolved_model_classes)

    def model_cv_specs(self, cv_indices=None):
        specs = []
        indices = list(range(self.n_splits)) if cv_indices is None else list(cv_indices)
        for ModelCls in self.resolve_model_classes():
            for cv_idx in indices:
                specs.append((ModelCls, cv_idx))
        return specs

    def create_model_cv_executions(self, cv_indices=None):
        executions = []
        for ModelCls, cv_idx in self.model_cv_specs(cv_indices):
            executions.append((
                self.create_model_job_for_cv(cv_idx),
                create_model_instance(
                    ModelCls,
                    random_state=self.random_state,
                    scoring_metric=self.scoring_metric,
                    metric_direction=self.metric_direction,
                    model_params=self.model_params,
                ),
            ))
        return executions

    def create_model_job_for_cv(self, cv_idx: int):
        return ModelJob(
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

    def run_single_model_cv(self, ModelCls, cv_idx: int):
        model_job = self.create_model_job_for_cv(cv_idx)
        model = create_model_instance(
            ModelCls,
            random_state=self.random_state,
            scoring_metric=self.scoring_metric,
            metric_direction=self.metric_direction,
            model_params=self.model_params,
        )
        model_job.run(model)

    def mark_phase_complete(self):
        mark_modeling_phase_complete(self.dataset_dir)
