import multiprocessing
import os
import time
import optuna
import pytest
import logging
import multiprocessing
from streamline.runners.dataprocess_runner import DataProcessRunner
from streamline.runners.imputation_runner import ImputationRunner
from streamline.runners.feature_runner import FeatureImportanceRunner
from streamline.runners.feature_runner import FeatureSelectionRunner
from streamline.runners.model_runner import ModelExperimentRunner
from streamline.modeling.utils import SUPPORTED_MODELS_SMALL

pytest.skip("Tested Already", allow_module_level=True)

num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

algorithms, run_parallel, output_path = ["MI", "MS"], False, "./tests/"
dataset_path, experiment_name = "./DemoData/", "demo",


def test_setup():
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

    del f_sel


test_algorithms = list()
for algorithm in SUPPORTED_MODELS_SMALL[:2]:
    test_algorithms.append(([algorithm, ],))


@pytest.mark.parametrize(
    ("algorithms", "run_parallel"),
    [
        # (['NB'], False),
        # (["LR"], False),
        (["NB", "LR", "DT"], False),
        (["NB", "LR", "DT"], True),
        # (['CGB'], False),
        # (['LGB'], False),
        # (['XGB'], False),
        # (['GP'], False),
        # (['XCS'], True),
        # (SUPPORTED_MODELS_SMALL, True),
    ]
    # +
    # [([algo], True) for algo in SUPPORTED_MODELS_SMALL]
)
def test_valid_model_runner(algorithms, run_parallel):
    start = time.time()

    logging.warning("Running Modelling Phase")
    logging.warning("Using " + str(num_cores) + " CPUs")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    runner = ModelExperimentRunner(output_path, experiment_name, algorithms, save_plots=True)
    runner.run(run_parallel)

    if run_parallel:
        how = "parallely"
    else:
        how = "serially"
    logging.warning("Modelling Step with " + str(algorithms) +
                    ", Time running " + how + ": " + str(time.time() - start))

    # shutil.rmtree(output_path)
