import os
import sys
import pickle
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.postanalysis.model_replicate import ReplicateJob


def run_cluster(argv):
    dataset_filename = argv[1]
    dataset_for_rep = argv[2]
    full_path = argv[3]
    class_label = argv[4]
    instance_label = argv[5] if argv[5] != "None" else None
    match_label = argv[6] if argv[6] != "None" else None
    experiment_path = '/'.join(full_path.split('/')[:-1])
    file = open(experiment_path + '/' + "algInfo.pickle", 'rb')
    alg_info = pickle.load(file)
    file.close()
    temp_algo = []
    for key in alg_info:
        if alg_info[key][0]:
            temp_algo.append(key)
    algorithms = temp_algo
    exclude = None
    len_cv = int(argv[9])
    export_feature_correlations = bool(argv[10])
    plot_roc = bool(argv[11])
    plot_prc = bool(argv[12])
    plot_metric_boxplots = bool(argv[13])
    categorical_cutoff = int(argv[14]) if argv[14] != "None" else None
    sig_cutoff = float(argv[15]) if argv[15] != "None" else None
    scale_data = bool(argv[16])
    impute_data = bool(argv[17])
    multi_impute = bool(argv[18])
    show_plots = bool(argv[19])
    scoring_metric = argv[20]

    job_obj = ReplicateJob(dataset_filename, dataset_for_rep, full_path, class_label, instance_label,
                          match_label, algorithms, exclude, len_cv, export_feature_correlations,
                          plot_roc, plot_prc, plot_metric_boxplots,
                          categorical_cutoff, sig_cutoff, scale_data, impute_data,
                          multi_impute, show_plots, scoring_metric)
    job_obj.run()


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
