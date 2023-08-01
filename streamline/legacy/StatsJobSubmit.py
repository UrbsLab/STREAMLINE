import os
import sys
import pickle
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.postanalysis.statistics import StatsJob


def run_cluster(argv):
    full_path = argv[1]
    experiment_path = '/'.join(full_path.split('/')[:-1])
    algorithms = None
    file = open(experiment_path + '/' + "algInfo.pickle", 'rb')
    alg_info = pickle.load(file)
    file.close()
    temp_algo = []
    for key in alg_info:
        if alg_info[key][0]:
            temp_algo.append(key)
    algorithms = temp_algo

    class_label = argv[3]
    instance_label = argv[4] if argv[4] != "None" else None
    scoring_metric = argv[5]
    len_cv = int(argv[6])
    top_features = int(argv[7]) if argv[7] != "None" else None
    sig_cutoff = float(argv[8]) if argv[8] != "None" else None
    metric_weight = argv[9] if argv[9] != "None" else None
    scale_data = eval(argv[10])
    exclude_options = argv[11].split(',')
    exclude_options = [x.strip() for x in exclude_options]
    show_plots = eval(argv[15])

    job_obj = StatsJob(full_path, algorithms, class_label, instance_label, scoring_metric,
                       len_cv, top_features, sig_cutoff, metric_weight, scale_data,
                       exclude_options,
                       show_plots)
    job_obj.run()


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
