import os
import time
import optuna
import pytest
import logging
from streamline.runners.eda_runner import EDARunner
from streamline.runners.dataprocess_runner import DataProcessRunner
from streamline.runners.feature_runner import FeatureImportanceRunner
from streamline.runners.feature_runner import FeatureSelectionRunner
from streamline.runners.model_runner import ModelExperimentRunner
from streamline.runners.stats_runner import StatsRunner
from streamline.runners.compare_runner import CompareRunner
from streamline.runners.report_runner import ReportRunner
from streamline.runners.replicate_runner import ReplicationRunner

pytest.skip("Tested Already", allow_module_level=True)

algorithms, run_parallel, output_path = ["MI", "MS"], False, "./tests/"
dataset_path, experiment_name = "./DemoData/", "demo",
model_algorithms = ["NB", "LR", "DT"]


def test_setup():
    start = time.time()
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    eda = EDARunner(dataset_path, output_path, experiment_name, exploration_list=None, plot_list=None,
                    class_label="Class", n_splits=5)
    eda.run(run_parallel=False)
    del eda

    dpr = DataProcessRunner(output_path, experiment_name)
    dpr.run(run_parallel=False)
    del dpr

    f_imp = FeatureImportanceRunner(output_path, experiment_name, algorithms=algorithms)
    f_imp.run(run_parallel=False)
    del f_imp

    f_sel = FeatureSelectionRunner(output_path, experiment_name, algorithms=algorithms)
    f_sel.run(run_parallel=False)
    del f_sel

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    runner = ModelExperimentRunner(output_path, experiment_name, model_algorithms)
    runner.run(run_parallel=True)
    del runner

    stats = StatsRunner(output_path, experiment_name, model_algorithms)
    stats.run(run_parallel=False)
    del stats

    compare = CompareRunner(output_path, experiment_name, algorithms=model_algorithms)
    compare.run(run_parallel=False)
    del compare

    report = ReportRunner(output_path, experiment_name, algorithms=model_algorithms)
    report.run(run_parallel=False)
    del report

    repl = ReplicationRunner('./DemoRepData', dataset_path + 'demodata.csv', output_path, experiment_name,
                             load_algo=True)
    repl.run(run_parallel)

    logging.warning("Ran Setup in " + str(time.time() - start))


@pytest.mark.parametrize(
    ("rep_data_path", "run_parallel"),
    [
        ("./DemoRepData/", False),
    ]
)
def test_valid_report2(rep_data_path, run_parallel):
    start = time.time()

    logging.warning("Running Replication Report Phase")

    report = ReportRunner(output_path, experiment_name, algorithms=model_algorithms,
                          training=False, rep_data_path=rep_data_path, dataset_for_rep=dataset_path + 'demodata.csv')
    report.run(run_parallel)

    if run_parallel:
        how = "parallely"
    else:
        how = "serially"
    logging.warning("Statistics Step with " + str(algorithms) +
                    ", Time running " + how + ": " + str(time.time() - start))

    # shutil.rmtree(output_path)
