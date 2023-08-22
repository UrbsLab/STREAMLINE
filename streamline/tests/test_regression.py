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
from streamline.runners.compare_runner import CompareRunner
from streamline.runners.report_runner import ReportRunner
from streamline.runners.replicate_runner import ReplicationRunner

# pytest.skip("Tested Already", allow_module_level=True)

algorithms, run_parallel, output_path = ["MI", "MS"], True, "./tests/"
dataset_path, rep_data_path = "./data/DemoDataRegression/", "./data/DemoRepDataRegression/"
experiment_name = "regression"
outcome_label = "Cognition_Score"
instance_label = "Class"
model_algorithms = ["LR", "RF", "EN"]


def test_regression():
    start = time.time()
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    eda = DataProcessRunner(dataset_path, output_path, experiment_name,
                            outcome_label="Cognition_Score", instance_label="Class", n_splits=3, ignore_features=None)
    eda.run(run_parallel=run_parallel)
    del eda

    dpr = ImputationRunner(output_path, experiment_name,
                           outcome_label="Cognition_Score", instance_label="Class")
    dpr.run(run_parallel=run_parallel)
    del dpr

    f_imp = FeatureImportanceRunner(output_path, experiment_name,
                                    outcome_label="Cognition_Score", instance_label="Class",
                                    algorithms=algorithms)
    f_imp.run(run_parallel=run_parallel)
    del f_imp

    f_sel = FeatureSelectionRunner(output_path, experiment_name,
                                   outcome_label="Cognition_Score", instance_label="Class",
                                   algorithms=algorithms)
    f_sel.run(run_parallel=run_parallel)
    del f_sel

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    runner = ModelExperimentRunner(output_path, experiment_name, model_algorithms,
                                   outcome_label="Cognition_Score", outcome_type="Continuous", instance_label="Class")
    runner.run(run_parallel=run_parallel)
    del runner

    stats = StatsRunner(output_path, experiment_name, outcome_type="Continuous",
                        outcome_label="Cognition_Score", instance_label="Class")
    stats.run(run_parallel=run_parallel)
    del stats

    compare = CompareRunner(output_path, experiment_name, outcome_type="Continuous",
                            outcome_label="Cognition_Score", instance_label="Class")
    compare.run(run_parallel=run_parallel)
    del compare

    report = ReportRunner(output_path, experiment_name)
    report.run(run_parallel=run_parallel)
    del report

    repl = ReplicationRunner('./data/DemoRepDataRegression', dataset_path + 'simulation_data.csv',
                             output_path, experiment_name)
    repl.run(run_parallel=run_parallel)

    report = ReportRunner(output_path, experiment_name,
                          training=False, rep_data_path=rep_data_path,
                          dataset_for_rep=dataset_path + 'simulation_data.csv')
    report.run(run_parallel)

    logging.warning("Ran Pipeline in " + str(time.time() - start))
