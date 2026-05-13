# streamline/phases/p5_feature_selection/cli.py  (additions)
import argparse, json, os
from streamline.p5_feature_selection.p5_runner import P5Runner
from streamline.p5_feature_selection.utils.fi_resolver import _discover_algorithms
from streamline.utils.run_commands import (
    add_run_command_args,
    apply_saved_run_command,
    require_args,
    save_run_command_from_args,
    snapshot_args,
)

def main():
    ap = argparse.ArgumentParser("STREAMLINE Phase 5 (Feature Selection) CLI",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--experiment_name", required=True)

    # default now "auto"
    ap.add_argument("--algorithms", default="auto",
                    help='Comma-separated (e.g. "MI,MS") OR "auto" to discover from feature_importance/*/')
    ap.add_argument("--n_splits", default=None, type=int)
    ap.add_argument("--outcome_label", default="Class")
    ap.add_argument("--instance_label", default=None)

    ap.add_argument("--max_features_to_keep", default=2000, type=int)
    ap.add_argument("--filter_poor_features", default=1, type=int)
    ap.add_argument("--overwrite_cv", default=0, type=int)

    ap.add_argument("--selector_id", default="default")
    ap.add_argument("--selector_params", default=None)

    ap.add_argument("--export_scores", default=1, type=int)
    ap.add_argument("--top_features", default=20, type=int)
    ap.add_argument("--show_plots", default=0, type=int)

    ap.add_argument("--run_cluster", default="Serial", help='Serial | Local | BashSLURM | BashLSF | <dask-cluster-name>')
    ap.add_argument("--queue", default="defq")
    ap.add_argument("--reserved_memory", default=4, type=int)

    # Convenience: print discovered algorithms and exit
    ap.add_argument("--list-algorithms", action="store_true",
                    help="List discovered algorithms per dataset (ignores --run_cluster)")
    add_run_command_args(ap)

    args = ap.parse_args()
    args = apply_saved_run_command(ap, args, "p5_feature_selection")
    require_args(ap, args, ["n_splits"])
    run_command_args = snapshot_args(args)

    if args.list_algorithms:
        exp_root = os.path.join(args.output_path, args.experiment_name)
        for name in sorted(os.listdir(exp_root)):
            ds_dir = os.path.join(exp_root, name)
            if not os.path.isdir(os.path.join(ds_dir, "CVDatasets")):
                continue
            algs = _discover_algorithms(ds_dir, args.n_splits, strict=False)
            print(f"{name}: {', '.join(algs) if algs else '(none)'}")
        return

    P5Runner(
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        algorithms=args.algorithms,          # "auto" supported
        n_splits=args.n_splits,
        outcome_label=args.outcome_label,
        instance_label=args.instance_label,
        max_features_to_keep=args.max_features_to_keep,
        filter_poor_features=bool(args.filter_poor_features),
        overwrite_cv=bool(args.overwrite_cv),
        selector_id=args.selector_id or "default",
        selector_params=json.loads(args.selector_params) if args.selector_params else None,
        export_scores=bool(args.export_scores),
        top_features=args.top_features,
        show_plots=bool(args.show_plots),
        run_cluster=args.run_cluster or "Serial",
        queue=args.queue,
        reserved_memory=args.reserved_memory,
    ).run()
    save_run_command_from_args(args, "p5_feature_selection", run_command_args)


if __name__ == "__main__":
    
    # # Serial run (use defaults from metadata.pickle)
    # python -m streamline.p5_feature_selection.p5_cli \
    # --output_path ./test \
    # --experiment_name MyExp --show_plots 1
    
    main()
