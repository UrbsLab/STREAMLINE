import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.dataprep.data_process import DataProcess
from streamline.dataprep.kfold_partitioning import KFoldPartitioner
from streamline.utils.dataset import Dataset
from streamline.utils.parser_helpers import process_cli_param


def run_cluster(argv):
    dataset_path = argv[1]
    output_path = argv[2]
    experiment_name = argv[3]
    if argv[4] != 'None':
        exclude_eda_output = argv[4].split(',')
        exclude_eda_output = [x.strip() for x in exclude_eda_output]
    else:
        exclude_eda_output = None
    class_label = argv[5]
    instance_label = argv[6] if argv[6] != "None" else None
    match_label = argv[7] if argv[7] != "None" else None
    n_splits = int(argv[8])
    partition_method = argv[9]
    ignore_features = process_cli_param(argv[10])    
    categorical_features = process_cli_param(argv[11])  
    quantitative_features = process_cli_param(argv[12])  
    top_features = int(argv[13])
    categorical_cutoff = int(argv[14])
    sig_cutoff = float(argv[15])
    featureeng_missingness = float(argv[16])
    cleaning_missingness = float(argv[17])
    correlation_removal_threshold = float(argv[18])
    random_state = None if argv[19] == "None" else int(argv[19])

    dataset = Dataset(dataset_path, class_label, match_label, instance_label)
    eda_obj = DataProcess(dataset, output_path + '/' + experiment_name,
                          ignore_features,
                          categorical_features, quantitative_features, exclude_eda_output,
                          categorical_cutoff, sig_cutoff, featureeng_missingness,
                          cleaning_missingness, correlation_removal_threshold, partition_method, n_splits,
                          random_state)
    eda_obj.run(top_features)


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
