import os
import sys
import time
import optuna
import logging
from run_config import *
from streamline.runners.eda_runner import EDARunner
from streamline.runners.dataprocess_runner import DataProcessRunner
from streamline.runners.feature_runner import FeatureImportanceRunner
from streamline.runners.feature_runner import FeatureSelectionRunner
from streamline.runners.model_runner import ModelExperimentRunner
from streamline.runners.stats_runner import StatsRunner
from streamline.runners.compare_runner import CompareRunner
from streamline.runners.report_runner import ReportRunner
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', None))

if num_cores:
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
else:
    file_handler = logging.FileHandler(OUTPUT_PATH + '/logs.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def run(obj, phase_str, run_parallel=False):
    start = time.time()
    obj.run(run_parallel=run_parallel)
    print("Ran " + phase_str + " Phase in " + str(time.time() - start))
    del obj


if __name__ == "__main__":

    start_g = time.time()

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    eda = EDARunner(DATASET_PATH, OUTPUT_PATH, EXPERIMENT_NAME,
                    class_label=CLASS_LABEL, instance_label=INSTANCE_LABEL, random_state=42)
    run(eda, "Exploratory")

    dpr = DataProcessRunner(OUTPUT_PATH, EXPERIMENT_NAME,
                            class_label=CLASS_LABEL, instance_label=INSTANCE_LABEL, random_state=42)
    run(dpr, "Data Process")

    f_imp = FeatureImportanceRunner(OUTPUT_PATH, EXPERIMENT_NAME, algorithms=FEATURE_ALGORITHMS,
                                    class_label=CLASS_LABEL, instance_label=INSTANCE_LABEL, random_state=42)
    run(f_imp, "Feature Imp.")

    f_sel = FeatureSelectionRunner(OUTPUT_PATH, EXPERIMENT_NAME, algorithms=FEATURE_ALGORITHMS,
                                   class_label=CLASS_LABEL, instance_label=INSTANCE_LABEL, random_state=42)
    run(f_sel, "Feature Sel.")

    model = ModelExperimentRunner(OUTPUT_PATH, EXPERIMENT_NAME, algorithms=MODEL_ALGORITHMS, exclude=["XCS", "eLCS"],
                                  class_label=CLASS_LABEL, instance_label=INSTANCE_LABEL, lcs_iterations=500000,
                                  random_state=RANDOM_STATE)
    run(model, "Modelling", RUN_PARALLEL)

    stats = StatsRunner(OUTPUT_PATH, EXPERIMENT_NAME, algorithms=MODEL_ALGORITHMS, exclude=EXCLUDE)
    run(stats, "Stats")

    compare = CompareRunner(OUTPUT_PATH, EXPERIMENT_NAME, algorithms=MODEL_ALGORITHMS, exclude=EXCLUDE)
    run(compare, "Dataset Compare")

    report = ReportRunner(OUTPUT_PATH, EXPERIMENT_NAME, algorithms=MODEL_ALGORITHMS, exclude=EXCLUDE)
    run(report, "Reporting")

    print("DONE!!!")
    print("Ran in " + str(time.time() - start_g))
