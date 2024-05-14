import os
import sys
import pickle
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.featurefns.importance import FeatureImportance


def run_cluster(argv):
    param_path = argv[1]
    with open(param_path, "rb") as input_file:
        params = pickle.load(input_file)
    globals().update(params)

    job_obj = FeatureImportance(cv_train_path, experiment_path, outcome_label,
                                instance_label, instance_subset, algorithm,
                                use_turf, turf_pct, random_state, n_jobs)
    job_obj.run()


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
