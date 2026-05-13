import argparse
from streamline.p7_ensembles.p7_runner import P7Runner
from streamline.p7_ensembles.utils.loader import list_ensembles
from streamline.utils.run_commands import (
    add_run_command_args,
    apply_saved_run_command,
    require_args,
    save_run_command_from_args,
    snapshot_args,
)

def _print_ens():
    print("\nAvailable ensembles:")
    for e in list_ensembles():
        print(f"  {e['id']:<10} - {e['name']}  [{e['module']}]")
    print("")

def main():
    ap = argparse.ArgumentParser("STREAMLINE Phase 7 (Ensembles)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--experiment_name", required=True)
    ap.add_argument("--n_splits", type=int, default=None)
    ap.add_argument("--outcome_label", default="Class")
    ap.add_argument("--outcome_type", default=None, help="Binary | Multiclass | Continuous; defaults to metadata, Continuous is rejected")
    ap.add_argument("--instance_label", default=None)
    ap.add_argument("--ensembles", default="hard_voting,soft_voting,stack_lr")
    ap.add_argument("--base_models", default=None)
    ap.add_argument("--meta_train_source", choices=["train","test"], default="train",
                help="Where to train stacking meta-classifier from base outputs (no CV, no base refit)")
    ap.add_argument("--calibrate", type=int, default=0)
    ap.add_argument("--calibrate_method", default="sigmoid")
    ap.add_argument("--calibrate_cv", type=int, default=5)
    ap.add_argument("--run_cluster", default="Serial")
    ap.add_argument("--queue", default="defq")
    ap.add_argument("--reserved_memory", type=int, default=4)
    ap.add_argument("--random_state", default=0)
    ap.add_argument("--list_ensembles", action="store_true")
    add_run_command_args(ap)
    args = ap.parse_args()
    args = apply_saved_run_command(ap, args, "p7_ensembles")
    run_command_args = snapshot_args(args)

    if args.list_ensembles:
        _print_ens()
        return

    require_args(ap, args, ["n_splits"])

    P7Runner(
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        n_splits=args.n_splits,
        outcome_label=args.outcome_label,
        outcome_type=args.outcome_type,
        instance_label=args.instance_label,
        ensembles=args.ensembles,
        base_models=args.base_models,
        meta_train_source=args.meta_train_source,
        # scoring_metric=args.scoring_metric,
        # metric_direction=args.metric_direction,
        # stack_tune=bool(args.stack_tune),
        # stack_trials=args.stack_trials,
        # stack_timeout=args.stack_timeout,
        calibrate=args.calibrate,
        calibrate_method=args.calibrate_method,
        calibrate_cv=args.calibrate_cv,
        run_cluster=args.run_cluster,
        queue=args.queue,
        reserved_memory=args.reserved_memory,
        random_state=(int(args.random_state) if str(args.random_state).lower() not in {"", "none"} else None),
    ).run()
    save_run_command_from_args(args, "p7_ensembles", run_command_args)

if __name__ == "__main__":
    
    ## List available ensembles

    # python -m streamline.p7_ensembles.p7_cli \
    # --output_path test --experiment_name data/DemoData --n_splits 5 --list_ensembles


    ## Plain (no tuning), include calibration

    # python -m streamline.p7_ensembles.p7_cli \
    # --output_path test --experiment_name data/DemoData --n_splits 5 \
    # --ensembles hard_voting,soft_voting,stack_lr \
    # --base_models LR,SVM,NB \
    # --calibrate 1 --calibrate_method sigmoid --calibrate_cv 5


    ## Tune stacking meta-LR (30 trials / 10 min)

    # python -m streamline.p7_ensembles.p7_cli \
    # --output_path test --experiment_name data/DemoData --n_splits 5 \
    # --ensembles stack_lr \
    # --base_models LR,SVM,NB \
    # --stack_tune 1 --stack_trials 30 --stack_timeout 600
    
    main()
