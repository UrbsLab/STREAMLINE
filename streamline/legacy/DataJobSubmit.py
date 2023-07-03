import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.dataprep.scale_and_impute import ScaleAndImpute


def run_cluster(argv):
    cv_train_path = argv[1]
    cv_test_path = argv[2]
    full_path = argv[3]
    scale_data = eval(argv[4])
    impute_data = eval(argv[5])
    multi_impute = eval(argv[6])
    overwrite_cv = eval(argv[7])
    outcome_label = argv[8] if argv[8] != "None" else None
    instance_label = argv[9] if argv[9] != "None" else None
    random_state = int(argv[10]) if argv[10] != "None" else None

    job_obj = ScaleAndImpute(cv_train_path, cv_test_path,
                             full_path,
                             scale_data, impute_data, multi_impute, overwrite_cv,
                             outcome_label, instance_label, random_state)
    job_obj.run()


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
