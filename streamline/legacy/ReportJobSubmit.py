import os
import sys
import pickle
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.postanalysis.gererate_report import ReportJob


def run_cluster(argv):
    output_path = argv[1]
    experiment_name = argv[2]
    experiment_path = None
    training = eval(argv[3])
    train_data_path = None if argv[4] == "None" else argv[4]
    rep_data_path = None if argv[5] == "None" else argv[5]

    job_obj = ReportJob(output_path, experiment_name, experiment_path,
                        training, train_data_path, rep_data_path)
    job_obj.run()


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
