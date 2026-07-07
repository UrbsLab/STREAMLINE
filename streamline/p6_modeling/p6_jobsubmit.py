import argparse
import logging
import sys
from pathlib import Path

try:
    from .modeling import (
        create_model_instance,
        mark_modeling_phase_complete,
        mark_modeling_phase_complete_if_all_model_cv_jobs_finished,
        model_class_matching_id,
        parse_model_params_json,
        resolve_model_classes_for_dataset,
    )
    from .utils.categorical import NATIVE_CATEGORICAL_MODELS_DEFAULT
    from .utils.loader import normalize_modeling_type
    from .utils.modeljob import ModelJob
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from streamline.p6_modeling.modeling import (
        create_model_instance,
        mark_modeling_phase_complete,
        mark_modeling_phase_complete_if_all_model_cv_jobs_finished,
        model_class_matching_id,
        parse_model_params_json,
        resolve_model_classes_for_dataset,
    )
    from streamline.p6_modeling.utils.categorical import NATIVE_CATEGORICAL_MODELS_DEFAULT
    from streamline.p6_modeling.utils.loader import normalize_modeling_type
    from streamline.p6_modeling.utils.modeljob import ModelJob

def parse_bool(x):
    if x is None: return False
    return str(x).strip().lower() in ("1", "true", "t", "yes", "y")

def main():
    ap = argparse.ArgumentParser("P6 Modeling jobsubmit (single dataset)")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--outcome_label", default="Class")
    ap.add_argument("--outcome_type", default=None)
    ap.add_argument("--model_type", default=None)
    ap.add_argument("--instance_label", default=None)
    ap.add_argument("--n_splits", type=int, required=True)
    ap.add_argument("--models", default=None)
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--cv_idx", type=int, default=None)
    # JSON: per-model overrides
    ap.add_argument(
        "--model_params_json",
        default=None,
        help="JSON string mapping model ids to dicts of attribute overrides.",
    )

    # calibration
    ap.add_argument("--calibrate", default="0")
    ap.add_argument("--calibrate_method", default="sigmoid")
    ap.add_argument("--calibrate_cv", type=int, default=5)

    # ModelJob controls
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--experiment_name", required=True)
    ap.add_argument("--scoring_metric", default="balanced_accuracy")
    ap.add_argument("--metric_direction", default="maximize")
    ap.add_argument("--n_trials", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--training_subsample", type=int, default=0)
    ap.add_argument("--uniform_fi", default="0")
    ap.add_argument("--save_plot", default="0")
    ap.add_argument("--random_state", default=None)
    ap.add_argument("--bypass_one_hot_for_native_models", default="1")
    ap.add_argument("--native_categorical_models", default=NATIVE_CATEGORICAL_MODELS_DEFAULT)

    args = ap.parse_args()
    modeling_type = normalize_modeling_type(outcome_type=args.outcome_type, model_type=args.model_type)
    instance_label = args.instance_label if args.instance_label else None
    n_splits = int(args.n_splits)
    random_state = int(args.random_state) if (args.random_state not in (None, "", "None")) else None
    bypass_native = parse_bool(args.bypass_one_hot_for_native_models)
    model_params = parse_model_params_json(args.model_params_json)

    model_classes = resolve_model_classes_for_dataset(
        dataset_dir=args.dataset_dir,
        model_type=modeling_type,
        models=args.models,
        bypass_one_hot_for_native_models=bypass_native,
        native_categorical_models=args.native_categorical_models,
    )

    if (args.model_id is None) != (args.cv_idx is None):
        ap.error("--model_id and --cv_idx must be provided together for a single submitted model/CV job.")

    if args.model_id is not None:
        ModelCls = model_class_matching_id(model_classes, args.model_id)
        if ModelCls is None:
            logging.warning(
                "Phase 6 jobsubmit found no runnable model '%s' after model filtering.",
                args.model_id,
            )
            mark_modeling_phase_complete_if_all_model_cv_jobs_finished(
                args.dataset_dir,
                model_classes,
                n_splits,
            )
            return
        jobs_to_run = [(ModelCls, int(args.cv_idx))]
    else:
        jobs_to_run = [
            (ModelCls, cv_idx)
            for ModelCls in model_classes
            for cv_idx in range(n_splits)
        ]

    for ModelCls, cv_idx in jobs_to_run:
        model_job = ModelJob(
            full_path=args.dataset_dir,
            output_path=args.output_path,
            experiment_name=args.experiment_name,
            cv_count=int(cv_idx),
            outcome_label=args.outcome_label,
            instance_label=instance_label,
            scoring_metric=args.scoring_metric,
            metric_direction=args.metric_direction,
            n_trials=int(args.n_trials),
            timeout=int(args.timeout),
            training_subsample=int(args.training_subsample),
            uniform_fi=parse_bool(args.uniform_fi),
            save_plot=parse_bool(args.save_plot),
            random_state=random_state,
            bypass_one_hot_for_native_models=bypass_native,
            native_categorical_models=args.native_categorical_models,
            calibrate=parse_bool(args.calibrate),
            calibrate_method=args.calibrate_method,
            calibrate_cv=int(args.calibrate_cv),
        )
        model = create_model_instance(
            ModelCls,
            random_state=random_state,
            scoring_metric=args.scoring_metric,
            metric_direction=args.metric_direction,
            model_params=model_params,
        )
        model_job.run(model)

    if args.model_id is not None:
        mark_modeling_phase_complete_if_all_model_cv_jobs_finished(
            args.dataset_dir,
            model_classes,
            n_splits,
        )
    else:
        mark_modeling_phase_complete(args.dataset_dir)

if __name__ == "__main__":
    main()
