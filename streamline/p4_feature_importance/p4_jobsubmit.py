import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.featurefns.importance import FeatureImportance


def run_cluster(argv):
    cv_train_path = argv[1]
    experiment_path = argv[2]
    outcome_label = argv[3]
    instance_label = argv[4] if argv[4] != "None" else None
    instance_subset = None if argv[5] == "None" else eval(argv[5])
    algorithm = argv[6]
    use_turf = eval(argv[7])
    turf_pct = eval(argv[8])
    random_state = None if argv[9] == "None" else int(argv[9])
    n_jobs = None if argv[10] == "None" else int(argv[10])

    job_obj = FeatureImportance(cv_train_path, experiment_path, outcome_label,
                                instance_label, instance_subset, algorithm,
                                use_turf, turf_pct, random_state, n_jobs)
    job_obj.run()


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
