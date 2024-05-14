import os
import pickle
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.modeling.modeljob import ModelJob
from streamline.modeling.utils import get_fi_for_ExSTraCS


def run_cluster(argv):
    param_path = argv[1]
    with open(param_path, "rb") as input_file:
        params = pickle.load(input_file)
    globals().update(params)
    print(params)
    print(vars())

    # if outcome_type == "Binary":
    #     with GlobalImport() as gi:
    #         from streamline.modeling.classification_utils import model_str_to_obj
    #         gi()

    # elif outcome_type == "Continuous":
    #     if scoring_metric == 'balanced_accuracy':
    #         scoring_metric = 'explained_variance'
    #     with GlobalImport() as gi:
    #         from streamline.modeling.regression_utils import model_str_to_obj
    #         gi()
    # elif outcome_type == "Multiclass":
    #     # logging.info("Using Multiclass Classification Models")
    #     with GlobalImport() as gi:
    #         from streamline.modeling.multiclass_utils import model_str_to_obj
    #         gi()
    # else:
    #     raise Exception("Unknown Outcome Type:" + str(outcome_type))
    
    file = open(output_path + '/' + experiment_name + '/' + "metadata.pickle", 'rb')
    metadata = pickle.load(file)
    filter_poor_features = metadata['Filter Poor Features']
    outcome_type = metadata['Outcome Type']
    file.close()
    dataset_directory_path = full_path.split('/')[-1]

    if outcome_type == "Binary":
        from streamline.modeling.classification_utils import model_str_to_obj
    elif outcome_type == "Multiclass":
        from streamline.modeling.multiclass_utils import model_str_to_obj
    elif outcome_type == "Continuous":
        from streamline.modeling.regression_utils import model_str_to_obj
    else:
        raise Exception("Unknown Outcome Type:" + str(outcome_type))

    job_obj = ModelJob(full_path, output_path, experiment_name, cv_count, outcome_label,
                       instance_label, scoring_metric, metric_direction, n_trials,
                       timeout, training_subsample, uniform_fi, save_plots, random_state)


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
                                                   outcome_label, instance_label, cv_count,
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
