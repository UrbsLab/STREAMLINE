import os
import time
import pytest
import shutil
import logging
from streamline.dataprep.eda_runner import EDARunner
from streamline.dataprep.data_process import DataProcessRunner
from streamline.featurefns.feature_runner import FeatureImportanceRunner


@pytest.mark.parametrize(
    ("output_path", "experiment_name", "exception"),
    [
        ("./tests/", 'demo', Exception),
        ("./tests/", ".@!#@", Exception),
    ],
)
def test_invalid_feature_sel(output_path, experiment_name, exception):
    with pytest.raises(exception):
        FeatureImportanceRunner(output_path, experiment_name)


@pytest.mark.parametrize(
    ("algorithm", "run_parallel", "use_turf", "turf_pct"),
    [
        ("MI", False, None, None),
        ("MI", True, None, None),
        ("MS", False, True, True),
        ("MS", False, True, False),
        ("MS", False, False, None),
    ],
)
def test_valid_feature_sel(algorithm, run_parallel, use_turf, turf_pct):
    dataset_path, output_path, experiment_name = "./DemoData/", "./tests4/", "demo",
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    eda = EDARunner(dataset_path, output_path, experiment_name, exploration_list=None, plot_list=None,
                    class_label="Class")
    eda.run(run_parallel=False)

    dpr = DataProcessRunner(output_path, experiment_name)
    dpr.run(run_parallel=False)

    start = time.time()

    f_imp = FeatureImportanceRunner(output_path, experiment_name, algorithm=algorithm,
                                    use_turf=use_turf, turf_pct=turf_pct)
    f_imp.run(run_parallel=run_parallel)
    if run_parallel:
        how = "parallely"
    else:
        how = "serially"
    logging.warning("Feature Importance Step with " + algorithm +
                    ", Time running " + how + ": " + str(time.time() - start))

    shutil.rmtree(output_path)
