import argparse
from .modeling import ModelingPhaseJob  # or `.job` depending on your filename
from .utils.categorical import NATIVE_CATEGORICAL_MODELS_DEFAULT
from .utils.loader import normalize_modeling_type

def _b(x):
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

    ModelingPhaseJob(
        dataset_dir=args.dataset_dir,
        outcome_label=args.outcome_label,
        model_type=modeling_type,
        instance_label=(args.instance_label if args.instance_label else None),
        n_splits=int(args.n_splits),
        models=args.models,
        model_params_json=args.model_params_json,

        calibrate=_b(args.calibrate),
        calibrate_method=args.calibrate_method,
        calibrate_cv=int(args.calibrate_cv),

        output_path=args.output_path,
        experiment_name=args.experiment_name,
        scoring_metric=args.scoring_metric,
        metric_direction=args.metric_direction,
        n_trials=int(args.n_trials),
        timeout=int(args.timeout),
        training_subsample=int(args.training_subsample),
        uniform_fi=_b(args.uniform_fi),
        save_plot=_b(args.save_plot),
        random_state=(int(args.random_state) if (args.random_state not in (None, "", "None")) else None),
        bypass_one_hot_for_native_models=_b(args.bypass_one_hot_for_native_models),
        native_categorical_models=args.native_categorical_models,
    ).run()

if __name__ == "__main__":
    main()
