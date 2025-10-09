# streamline/phases/p4_feature_selection/p4_jobsubmit.py
import argparse, json
from streamline.p4_feature_importance.importance import FeatureImportance

def _maybe_json(s: str):
    try: return json.loads(s or "{}")
    except Exception: return {}

def _maybe_int(s): 
    return None if s in (None,"") else int(s)

def _maybe_float(s):
    return None if s in (None,"") else float(s)

def main():
    ap = argparse.ArgumentParser("P4 single-CV-pair jobsubmit")
    ap.add_argument("--cv_train_path", required=True)
    ap.add_argument("--cv_test_path", required=True)
    ap.add_argument("--experiment_path", required=True)

    ap.add_argument("--importance_id", default="mutualinformation")
    ap.add_argument("--importance_params", default="{}")
    ap.add_argument("--top_k", default=None)
    ap.add_argument("--threshold", default=None)
    ap.add_argument("--keep_original_features", default="0")
    ap.add_argument("--overwrite_cv", default="1")
    ap.add_argument("--outcome_label", default="Class")
    ap.add_argument("--outcome_type", default=None)
    ap.add_argument("--instance_label", default=None)
    ap.add_argument("--random_state", default=None)

    args = ap.parse_args()

    FeatureImportance(
        cv_train_path=args.cv_train_path,
        cv_test_path=args.cv_test_path,
        experiment_path=args.experiment_path,
        importance_id=args.importance_id or "mutualinformation",
        importance_params=_maybe_json(args.importance_params),
        top_k=_maybe_int(args.top_k),
        threshold=_maybe_float(args.threshold),
        keep_original_features=(str(args.keep_original_features).lower() not in ("0","false","no")),
        overwrite_cv=(str(args.overwrite_cv).lower() not in ("0","false","no")),
        outcome_label=args.outcome_label or "Class",
        outcome_type=args.outcome_type,
        instance_label=(None if args.instance_label in (None,"") else args.instance_label),
        random_state=_maybe_int(args.random_state),
    ).run()

if __name__ == "__main__":
    main()
