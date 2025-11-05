import argparse
from streamline.p7_ensembles.p7_runner import EnsemblePhaseJob

def main():
    ap = argparse.ArgumentParser("P7 jobsubmit (single dataset)")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--n_splits", type=int, required=True)
    ap.add_argument("--outcome_label", default="Class")
    ap.add_argument("--instance_label", default=None)
    ap.add_argument("--ensembles", default="vote_hard,vote_soft,stack_lr")
    ap.add_argument("--base_models", default=None)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--experiment_name", required=True)
    args = ap.parse_args()

    EnsemblePhaseJob(
        dataset_dir=args.dataset_dir,
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        n_splits=args.n_splits,
        outcome_label=args.outcome_label,
        instance_label=(args.instance_label or None),
        ensembles=args.ensembles,
        base_model_filter=args.base_models,
    ).run()

if __name__ == "__main__":
    main()
