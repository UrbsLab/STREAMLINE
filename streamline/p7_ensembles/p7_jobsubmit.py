import argparse
from streamline.p7_ensembles.p7_runner import P7Runner
from streamline.p7_ensembles.utils.loader import list_ensembles

def _print_ensembles():
    print("\nAvailable ensembles:")
    for e in list_ensembles():
        print(f"  {e['id']:<10} - {e['name']}  [{e['module']}]")
    print("")

def main():
    ap = argparse.ArgumentParser("STREAMLINE Phase 7 (Ensembles)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--experiment_name", required=True)
    ap.add_argument("--n_splits", type=int, required=True)
    ap.add_argument("--outcome_label", default="Class")
    ap.add_argument("--instance_label", default=None)
    ap.add_argument("--ensembles", default="vote_hard,vote_soft,stack_lr")
    ap.add_argument("--base_models", default=None, help="Comma list of base model small_names filter (e.g., LR,SVM,NB)")
    ap.add_argument("--meta_train_source", choices=["train","test"], default="train")
    ap.add_argument("--calibrate", type=int, default=0)
    ap.add_argument("--calibrate_method", default="sigmoid")
    ap.add_argument("--calibrate_cv", type=int, default=5)
    ap.add_argument("--run_cluster", default="Serial", help="Serial | Local | BashSLURM | BashLSF | <dask-cluster-name>")
    ap.add_argument("--queue", default="defq")
    ap.add_argument("--reserved_memory", type=int, default=4)
    ap.add_argument("--random_state", type=int, default=0)
    ap.add_argument("--list_ensembles", action="store_true")
    args = ap.parse_args()

    if args.list_ensembles:
        _print_ensembles()
        return

    P7Runner(
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        n_splits=args.n_splits,
        outcome_label=args.outcome_label,
        instance_label=args.instance_label,
        ensembles=args.ensembles,
        base_models=args.base_models,
        meta_train_source=args.meta_train_source,
        calibrate=args.calibrate,
        calibrate_method=args.calibrate_method,
        calibrate_cv=args.calibrate_cv,
        run_cluster=args.run_cluster,
        queue=args.queue,
        reserved_memory=args.reserved_memory,
        random_state=args.random_state,
    ).run()

if __name__ == "__main__":
    main()
