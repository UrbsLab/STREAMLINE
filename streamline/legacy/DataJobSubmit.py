import os
import sys
import pickle
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.dataprep.scale_and_impute import ScaleAndImpute


def run_cluster(argv):
    param_path = argv[1]
    with open(param_path, "rb") as input_file:
        params = pickle.load(input_file)
    globals().update(params)
    full_path = output_path + "/" + experiment_name


    job_obj = ScaleAndImpute(cv_train_path, cv_test_path,
                             full_path,
                             scale_data, impute_data, multi_impute, overwrite_cv,
                             outcome_label, instance_label, random_state)
    job_obj.run()


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
