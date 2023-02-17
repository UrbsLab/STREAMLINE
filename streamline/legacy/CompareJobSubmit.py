import os
import sys
import pickle
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.postanalysis.dataset_compare import CompareJob


def run_cluster(argv):
    output_path = argv[1]
    experiment_name = argv[2]
    experiment_path = argv[3] if argv[3] != "None" else None
    file = open(output_path + '/' + experiment_name + '/' + "algInfo.pickle", 'rb')
    alg_info = pickle.load(file)
    file.close()
    temp_algo = []
    for key in alg_info:
        if alg_info[key][0]:
            temp_algo.append(key)
    algorithms = temp_algo

    exclude = None
    class_label = argv[6]
    instance_label = argv[7] if argv[7] != "None" else None
    sig_cutoff = float(argv[8])
    show_plots = bool(argv[9])

    job_obj = CompareJob(output_path, experiment_name, experiment_path, algorithms, exclude,
                         class_label, instance_label, sig_cutoff, show_plots)
    job_obj.run()


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
