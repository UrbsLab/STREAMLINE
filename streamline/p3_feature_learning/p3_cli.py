# streamline/phases/p3_feature_learning/cli.py
import argparse, json
from streamline.p3_feature_learning.p3_runner import P3Runner
from streamline.p3_feature_learning.utils.fl_loader import list_learners
from streamline.utils.run_commands import (
    add_run_command_args,
    apply_saved_run_command,
    save_run_command_from_args,
    snapshot_args,
)

def _maybe_json(s: str):
    try: return json.loads(s or "{}")
    except Exception: return {}

def _maybe_bool(s, default=None):
    if s is None: return default
    v = str(s).lower()
    if v in ("1","true","t","yes","y"): return True
    if v in ("0","false","f","no","n"): return False
    return default

def main():
    ap = argparse.ArgumentParser("STREAMLINE Phase 3 (PCA) CLI",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--experiment_name", required=True)

    ap.add_argument("--learner_id", default=None)
    ap.add_argument("--learner_params", default=None)
    ap.add_argument("--feature_namespace", default=None)
    ap.add_argument("--keep_original_features", default=None)
    ap.add_argument("--overwrite_cv", default=None)
    ap.add_argument("--outcome_label", default=None)
    ap.add_argument("--instance_label", default=None)
    ap.add_argument("--random_state", default=None, type=int)

    ap.add_argument("--run_cluster", default="Serial")
    ap.add_argument("--queue", default="defq")
    ap.add_argument("--reserved_memory", default=4, type=int)

    ap.add_argument("--list-learners", action="store_true")
    add_run_command_args(ap)

    args = ap.parse_args()
    args = apply_saved_run_command(ap, args, "p3_feature_learning")
    run_command_args = snapshot_args(args)

    if args.list_learners:
        learners = list_learners()
        print("Available learners:")
        for k, cls in sorted(learners.items()):
            print(f"  {k:10s} -> {cls.__module__}.{cls.__name__}")
        return

    runner = P3Runner(
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        learner_id=args.learner_id,
        learner_params=_maybe_json(args.learner_params) if args.learner_params else None,
        feature_namespace=args.feature_namespace,
        keep_original_features=_maybe_bool(args.keep_original_features),
        overwrite_cv=_maybe_bool(args.overwrite_cv),
        outcome_label=args.outcome_label,
        instance_label=args.instance_label,
        random_state=args.random_state,
        run_cluster=args.run_cluster if args.run_cluster not in (None, "Serial", "False", "false") else False,
        queue=args.queue,
        reserved_memory=args.reserved_memory,
    )
    runner.run()
    save_run_command_from_args(args, "p3_feature_learning", run_command_args)

if __name__ == "__main__":
    main()


    # # Serial run (use defaults from metadata.pickle)
    # python -m streamline.p3_feature_learning.p3_cli \
    # --output_path ./test \
    # --experiment_name MyExp
