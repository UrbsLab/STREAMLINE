import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.dataprep.data_process import DataProcess
from streamline.dataprep.kfold_partitioning import KFoldPartitioner
from streamline.utils.dataset import Dataset


def run_cluster(argv):
    dataset_path = argv[1]
    output_path = argv[2]
    experiment_name = argv[3]
    exclude_eda_output = argv[4].split(',')
    exclude_eda_output = [x.strip() for x in exclude_eda_output]
    class_label = argv[5]
    instance_label = argv[6] if argv[6] != "None" else None
    match_label = argv[7] if argv[7] != "None" else None
    n_splits = int(argv[8])
    partition_method = argv[9]
    ignore_features = None if argv[10] == "None" else eval(argv[10])
    categorical_features = None if argv[11] == "None" else eval(argv[11])
    top_features = int(argv[12])
    categorical_cutoff = int(argv[13])
    sig_cutoff = float(argv[14])
    featureeng_missingness = float(argv[15])
    cleaning_missingness = float(argv[16])
    correlation_removal_threshold = float(argv[17])
    random_state = None if argv[18] == "None" else int(argv[18])

    dataset = Dataset(dataset_path, class_label, match_label, instance_label)
    eda_obj = DataProcess(dataset, output_path + '/' + experiment_name,
                          ignore_features,
                          categorical_features, exclude_eda_output,
                          categorical_cutoff, sig_cutoff, featureeng_missingness,
                          cleaning_missingness, correlation_removal_threshold, partition_method, n_splits,
                          random_state)
    eda_obj.run(top_features)


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
