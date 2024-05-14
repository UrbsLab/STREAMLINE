import os
import sys
import pickle
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.postanalysis.model_replicate import ReplicateJob


def run_cluster(argv):
    param_path = argv[1]
    with open(param_path, "rb") as input_file:
        params = pickle.load(input_file)
    globals().update(params)
    # file = open(experiment_path + '/' + "algInfo.pickle", 'rb')
    # alg_info = pickle.load(file)
    # file.close()
    # temp_algo = []
    # for key in alg_info:
    #     if alg_info[key][0]:
    #         temp_algo.append(key)
    # algorithms = temp_algo
    # file = open(experiment_path + '/' + "metadata.pickle", 'rb')
    # metadata = pickle.load(file)
    # file.close()
    # ignore_features = metadata['Ignored Features']

    job_obj = ReplicateJob(dataset_filename, dataset_for_rep, full_path, outcome_label, outcome_type, instance_label,
                           match_label, ignore_features, cv_partitions,
                           exclude_plots,
                           categorical_cutoff, sig_cutoff, scale_data, impute_data,
                           multi_impute, show_plots, scoring_metric, random_state)
    job_obj.run()


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
