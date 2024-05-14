import os
import sys
import pickle
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.postanalysis.gererate_report import ReportJob


def run_cluster(argv):
    param_path = argv[1]
    with open(param_path, "rb") as input_file:
        params = pickle.load(input_file)
    globals().update(params)

    job_obj = ReportJob(output_path, experiment_name, experiment_path,
                        training, train_data_path, rep_data_path)
    job_obj.run()


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
