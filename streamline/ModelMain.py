import sys
from streamline.modeling.modeljob import ModelJob
from streamline.modeling.utils import model_str_to_obj


def run_cluster(argv):
    full_path = argv[1]
    output_path = argv[2]
    experiment_name = argv[3]
    cv_count = int(argv[4])
    class_label = argv[5]
    instance_label = argv[6] if argv[6] is not None else None
    scoring_metric = argv[7]
    metric_direction = argv[8]
    n_trials = int(argv[9])
    timeout = int(argv[10])
    uniform_fi = eval(argv[11])
    save_plot = eval(argv[12])
    random_state = None if argv[13] is None else int(argv[13])
    algorithm = argv[14]
    n_jobs = None if argv[15] is None else int(argv[15])
    do_lcs_sweep = eval(argv[16])
    lcs_iterations = int(argv[17])
    lcs_n = int(argv[18])
    lcs_nu = int(argv[19])

    job_obj = ModelJob(full_path, output_path, experiment_name, cv_count, class_label,
                       instance_label, scoring_metric, metric_direction, n_trials,
                       timeout, uniform_fi, save_plot, random_state)

    if (not do_lcs_sweep) or (algorithm not in ['eLCS', 'XCS', 'ExSTraCS']):
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
