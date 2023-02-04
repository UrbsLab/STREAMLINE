import os
import time
import optuna
import pytest
import shutil
import logging
from streamline.dataprep.eda_runner import EDARunner
from streamline.dataprep.data_process import DataProcessRunner
from streamline.featurefns.feature_runner import FeatureImportanceRunner
from streamline.featurefns.feature_runner import FeatureSelectionRunner
from streamline.modeling.model_runner import ModelExperimentRunner
from streamline.postanalysis.stats_runner import StatsRunner
from streamline.postanalysis.compare_runner import CompareRunner

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
    stats.run(run_parallel=run_parallel)

    del stats

    logging.warning("Ran Setup in " + str(time.time() - start))


@pytest.mark.parametrize(
    ("algorithms", "run_parallel"),
    [
        (model_algorithms, False),
    ]
)
def test_valid_stats(algorithms, run_parallel):
    start = time.time()

    logging.warning("Running Compare Phase")

    compare = CompareRunner(output_path, experiment_name, algorithms=model_algorithms)
    compare.run(run_parallel)

    if run_parallel:
        how = "parallely"
    else:
        how = "serially"
    logging.warning("Statistics Step with " + str(algorithms) +
                    ", Time running " + how + ": " + str(time.time() - start))

    # shutil.rmtree(output_path)
