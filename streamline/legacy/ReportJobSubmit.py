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
    algorithms = None
    file = open(output_path + '/' + experiment_name + '/' + "algInfo.pickle", 'rb')
    alg_info = pickle.load(file)
    file.close()
    temp_algo = []
    for key in alg_info:
        if alg_info[key][0]:
            temp_algo.append(key)
    algorithms = temp_algo
    exclude = None
    training = eval(argv[6])
    train_data_path = None if argv[7] == "None" else argv[7]
    rep_data_path = None if argv[8] == "None" else argv[8]

    job_obj = ReportJob(output_path, experiment_name, experiment_path, algorithms, exclude,
                        training, train_data_path, rep_data_path)
    job_obj.run()


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
