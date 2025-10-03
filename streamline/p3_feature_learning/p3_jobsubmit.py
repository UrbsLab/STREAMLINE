# streamline/phases/p3_feature_learning/p3_jobsubmit.py
import argparse, json, os, pickle
from streamline.p3_feature_learning.feature_learn import FeatureLearn

def _maybe_json(s: str):
    try: return json.loads(s or "{}")
    except Exception: return {}

def main():
    ap = argparse.ArgumentParser("P3 single-CV-pair jobsubmit")
    ap.add_argument("--cv_train_path", required=True)
    ap.add_argument("--cv_test_path", required=True)
    ap.add_argument("--experiment_path", required=True)

    ap.add_argument("--learner_id", default="pca")
    ap.add_argument("--learner_params", default="{}")
    ap.add_argument("--feature_namespace", default="FL_PCA")
    ap.add_argument("--keep_original_features", default="1")
    ap.add_argument("--overwrite_cv", default="1")
    ap.add_argument("--outcome_label", default="Class")
    ap.add_argument("--instance_label", default=None)
    ap.add_argument("--random_state", default=None)

    args = ap.parse_args()

    job = FeatureLearn(
        cv_train_path=args.cv_train_path,
        cv_test_path=args.cv_test_path,
        experiment_path=args.experiment_path,
        learner_id=args.learner_id or "pca",
        learner_params=_maybe_json(args.learner_params),
        feature_namespace=args.feature_namespace,
        keep_original_features=(str(args.keep_original_features).strip() not in ("0","false","False")),
        overwrite_cv=(str(args.overwrite_cv).strip() not in ("0","false","False")),
        outcome_label=args.outcome_label or "Class",
        instance_label=(None if args.instance_label in (None,"") else args.instance_label),
        random_state=(None if args.random_state in (None,"") else int(args.random_state)),
    )
    job.run()

if __name__ == "__main__":
    main()
