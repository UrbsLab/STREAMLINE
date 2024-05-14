import os
import sys
import pickle
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.featurefns.selection import FeatureSelection


def run_cluster(argv):
    param_path = argv[1]
    with open(param_path, "rb") as input_file:
        params = pickle.load(input_file)
    globals().update(params)

    job_obj = FeatureSelection(full_path, n_datasets, algorithms,
                               outcome_label, instance_label, export_scores,
                               top_features, max_features_to_keep,
                               filter_poor_features, overwrite_cv)
    job_obj.run()


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
