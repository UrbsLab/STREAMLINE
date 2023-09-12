import os
import pickle
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.modeling.modeljob import ModelJob
from streamline.modeling.utils import model_str_to_obj
from streamline.modeling.utils import get_fi_for_ExSTraCS


def run_cluster(argv):
    full_path = argv[1]
    output_path = argv[2]
    experiment_name = argv[3]
    cv_count = int(argv[4])
    class_label = argv[5]
    instance_label = argv[6] if argv[6] != "None" else None
    scoring_metric = argv[7]
    metric_direction = argv[8]
    n_trials = int(argv[9])
    timeout = int(argv[10])
    training_subsample = int(argv[11])
    uniform_fi = eval(argv[12])
    save_plot = eval(argv[13])
    random_state = None if argv[14] == "None" else int(argv[14])
    algorithm = argv[15]
    n_jobs = None if argv[16] == "None" else int(argv[16])
    do_lcs_sweep = eval(argv[17])
    lcs_iterations = int(argv[18])
    lcs_n = int(argv[19])
    lcs_nu = int(argv[20])

    file = open(output_path + '/' + experiment_name + '/' + "metadata.pickle", 'rb')
    metadata = pickle.load(file)
    filter_poor_features = metadata['Filter Poor Features']
    file.close()

    dataset_directory_path = full_path.split('/')[-1]

    job_obj = ModelJob(full_path, output_path, experiment_name, cv_count, class_label,
                       instance_label, scoring_metric, metric_direction, n_trials,
                       timeout, training_subsample, uniform_fi, save_plot, random_state)

    if algorithm not in ['eLCS', 'XCS', 'ExSTraCS']:
        model = model_str_to_obj(algorithm)(cv_folds=3,
                                            scoring_metric=scoring_metric,
                                            metric_direction=metric_direction,
                                            random_state=random_state,
                                            cv=None, n_jobs=n_jobs)
    else:
        if algorithm == 'ExSTraCS':
            expert_knowledge = get_fi_for_ExSTraCS(output_path, experiment_name,
                                                   dataset_directory_path,
                                                   class_label, instance_label, cv_count,
                                                   filter_poor_features)
            if do_lcs_sweep:
                model = model_str_to_obj(algorithm)(cv_folds=3,
                                                    scoring_metric=scoring_metric,
                                                    metric_direction=metric_direction,
                                                    random_state=random_state,
                                                    cv=None, n_jobs=n_jobs,
                                                    expert_knowledge=expert_knowledge)
            else:
                model = model_str_to_obj(algorithm)(cv_folds=3,
                                                    scoring_metric=scoring_metric,
                                                    metric_direction=metric_direction,
                                                    random_state=random_state,
                                                    cv=None, n_jobs=n_jobs,
                                                    iterations=lcs_iterations,
                                                    N=lcs_n, nu=lcs_nu,
                                                    expert_knowledge=expert_knowledge)
        else:
            if do_lcs_sweep:
                model = model_str_to_obj(algorithm)(cv_folds=3,
                                                    scoring_metric=scoring_metric,
                                                    metric_direction=metric_direction,
                                                    random_state=random_state,
                                                    cv=None, n_jobs=n_jobs)
            else:
                model = model_str_to_obj(algorithm)(cv_folds=3,
                                                    scoring_metric=scoring_metric,
                                                    metric_direction=metric_direction,
                                                    random_state=random_state,
                                                    cv=None, n_jobs=n_jobs,
                                                    iterations=lcs_iterations,
                                                    N=lcs_n, nu=lcs_nu)
    job_obj.run(model)


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
