import os
import time
import optuna
import pytest
import logging
from streamline.runners.dataprocess_runner import DataProcessRunner
from streamline.runners.imputation_runner import ImputationRunner
from streamline.runners.feature_runner import FeatureImportanceRunner
from streamline.runners.feature_runner import FeatureSelectionRunner
from streamline.runners.model_runner import ModelExperimentRunner
from streamline.runners.stats_runner import StatsRunner

pytest.skip("Tested Already", allow_module_level=True)

algorithms, run_parallel, output_path = ["MI", "MS"], False, "./tests/"
dataset_path, experiment_name = "./DemoData/", "demo",
model_algorithms = ["NB", "LR", "DT"]


def test_setup():
    start = time.time()
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    eda = DataProcessRunner(dataset_path, output_path, experiment_name, exploration_list=None, plot_list=None,
                            class_label="Class", n_splits=5)
    eda.run(run_parallel=False)
    del eda

    dpr = ImputationRunner(output_path, experiment_name)
    dpr.run(run_parallel=False)
    del dpr

    f_imp = FeatureImportanceRunner(output_path, experiment_name, algorithms=algorithms)
    f_imp.run(run_parallel=False)
    del f_imp

    f_sel = FeatureSelectionRunner(output_path, experiment_name, algorithms=algorithms)
    f_sel.run(run_parallel=False)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    runner = ModelExperimentRunner(output_path, experiment_name, model_algorithms)
    runner.run(run_parallel=True)

    del runner

    logging.warning("Ran Setup in " + str(time.time() - start))


@pytest.mark.parametrize(
    ("algorithms", "run_parallel"),
    [
        (model_algorithms, False),
    ]
)
def test_valid_stats(algorithms, run_parallel):
    start = time.time()

    logging.warning("Running Stats Phase")

    stats = StatsRunner(output_path, experiment_name, algorithms)
    stats.run(run_parallel=run_parallel)

    if run_parallel:
        how = "parallely"
    else:
        how = "serially"
    logging.warning("Statistics Step with " + str(algorithms) +
                    ", Time running " + how + ": " + str(time.time() - start))

    # shutil.rmtree(output_path)
