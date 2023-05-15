import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.dataprep.exploratory_analysis import EDAJob
from streamline.dataprep.kfold_partitioning import KFoldPartitioner
from streamline.utils.dataset import Dataset


def run_cluster(argv):
    dataset_path = argv[1]
    output_path = argv[2]
    experiment_name = argv[3]
    exploration_list = eval(argv[4])
    plot_list = eval(argv[5])
    class_label = argv[6]
    instance_label = argv[7] if argv[7] != "None" else None
    match_label = argv[8] if argv[8] != "None" else None
    n_splits = int(argv[9])
    partition_method = argv[10]
    ignore_features = None if argv[11] == "None" else eval(argv[11])
    categorical_features = None if argv[12] == "None" else eval(argv[12])
    top_features = int(argv[13])
    categorical_cutoff = int(argv[14])
    sig_cutoff = float(argv[15])
    sig_cutoff = float(argv[16])
    random_state = None if argv[17] == "None" else int(argv[17])

    dataset = Dataset(dataset_path, class_label, match_label, instance_label)
    eda_obj = EDAJob(dataset, output_path + '/' + experiment_name,
                     ignore_features,
                     categorical_features, exploration_list, plot_list,
                     categorical_cutoff, sig_cutoff,
                     random_state)
    eda_obj.run(top_features)
    kfold_obj = KFoldPartitioner(dataset,
                                 partition_method, output_path + '/' + experiment_name,
                                 n_splits, random_state)
    kfold_obj.run()


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
