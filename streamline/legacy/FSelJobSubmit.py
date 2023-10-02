import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.featurefns.selection import FeatureSelection


def run_cluster(argv):
    full_path = argv[1]
    n_datasets = int(argv[2])
    MI, MS = "MI", "MS"
    algorithms = None if argv[3] == "None" else eval(argv[3])
    # print(algorithms)
    class_label = argv[4]
    instance_label = argv[5] if argv[5] != "None" else None
    export_scores = eval(argv[6])
    top_features = int(argv[7])
    max_features_to_keep = int(argv[8])
    filter_poor_features = eval(argv[9])
    overwrite_cv = eval(argv[10])

    job_obj = FeatureSelection(full_path, n_datasets, algorithms,
                               class_label, instance_label, export_scores,
                               top_features, max_features_to_keep,
                               filter_poor_features, overwrite_cv)
    job_obj.run()


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
