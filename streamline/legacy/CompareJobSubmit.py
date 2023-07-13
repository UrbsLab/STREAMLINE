import os
import sys
import pickle
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.postanalysis.dataset_compare import CompareJob


def run_cluster(argv):
    output_path = argv[1]
    experiment_name = argv[2]
    experiment_path = argv[3] if argv[3] != "None" else None
    outcome_type = argv[4]
    outcome_label = argv[5]
    instance_label = argv[6] if argv[6] != "None" else None
    sig_cutoff = float(argv[7])
    show_plots = eval(argv[8])

    job_obj = CompareJob(output_path, experiment_name, experiment_path,
                         outcome_label, outcome_type, instance_label, sig_cutoff, show_plots)
    job_obj.run()


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
