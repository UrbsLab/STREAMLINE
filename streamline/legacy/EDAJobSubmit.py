import os
import sys
import pickle
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.dataprep.data_process import DataProcess
from streamline.dataprep.kfold_partitioning import KFoldPartitioner
from streamline.utils.dataset import Dataset
from streamline.utils.parser_helpers import process_cli_param


def run_cluster(argv):
    param_path = argv[1]
    with open(param_path, "rb") as input_file:
        params = pickle.load(input_file)
    params = open(param_path)
    locals().update(params)


    dataset = Dataset(dataset_path, outcome_label, match_label, instance_label, outcome_type)
    eda_obj = DataProcess(dataset, output_path + '/' + experiment_name,
                          ignore_features,
                          categorical_features, quantitative_features, exclude_eda_output,
                          categorical_cutoff, sig_cutoff, featureeng_missingness,
                          cleaning_missingness, correlation_removal_threshold, partition_method, n_splits,
                          random_state)
    eda_obj.run(top_features)


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
