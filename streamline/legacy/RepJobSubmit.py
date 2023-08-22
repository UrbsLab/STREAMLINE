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
    outcome_label = argv[4]
    outcome_type = argv[5]
    instance_label = argv[6] if argv[6] != "None" else None
    match_label = argv[7] if argv[7] != "None" else None
    experiment_path = '/'.join(full_path.split('/')[:-1])
    file = open(experiment_path + '/' + "algInfo.pickle", 'rb')
    alg_info = pickle.load(file)
    file.close()
    temp_algo = []
    for key in alg_info:
        if alg_info[key][0]:
            temp_algo.append(key)
    algorithms = temp_algo
    file = open(experiment_path + '/' + "metadata.pickle", 'rb')
    metadata = pickle.load(file)
    file.close()
    ignore_features = metadata['Ignored Features']
    exclude = None
    len_cv = int(argv[9])
    if argv != 'None':
        exclude_options = argv[10].split(',')
        exclude_options = [x.strip() for x in exclude_options]
    else:
        exclude_options = None
    categorical_cutoff = int(argv[11]) if argv[11] != "None" else None
    sig_cutoff = float(argv[12]) if argv[12] != "None" else None
    scale_data = eval(argv[13])
    impute_data = eval(argv[14])
    multi_impute = eval(argv[15])
    show_plots = eval(argv[16])
    scoring_metric = argv[17]
    random_state = eval(argv[18])

    job_obj = ReplicateJob(dataset_filename, dataset_for_rep, full_path, outcome_label, outcome_type, instance_label,
                           match_label, ignore_features, len_cv,
                           exclude_options,
                           categorical_cutoff, sig_cutoff, scale_data, impute_data,
                           multi_impute, show_plots, scoring_metric, random_state)
    job_obj.run()


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
