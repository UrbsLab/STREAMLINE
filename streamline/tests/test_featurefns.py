import os
import time
import pytest
import shutil
import logging
from streamline.dataprep.eda_runner import EDARunner
from streamline.dataprep.data_process import DataProcessRunner
from streamline.featurefns.feature_runner import FeatureImportanceRunner
from streamline.featurefns.feature_runner import FeatureSelectionRunner


@pytest.mark.parametrize(
    ("output_path", "experiment_name", "exception"),
    [
        ("./tests/", 'demo', Exception),
        ("./tests/", ".@!#@", Exception),
    ],
)
def test_invalid_feature_imp(output_path, experiment_name, exception):
    with pytest.raises(exception):
        FeatureImportanceRunner(output_path, experiment_name)


@pytest.mark.parametrize(
    ("output_path", "experiment_name", "exception"),
    [
        ("./tests/", 'demo', Exception),
        ("./tests/", ".@!#@", Exception),
    ],
)
def test_invalid_feature_sel(output_path, experiment_name, exception):
    with pytest.raises(exception):
        FeatureSelectionRunner(output_path, experiment_name)


@pytest.mark.parametrize(
    ("algorithm", "run_parallel", "use_turf", "turf_pct", "output_path"),
    [
        ("MI", False, None, None, "./tests4_1/"),
        # ("MI", True, None, None, "./tests4_2/"),
        ("MS", False, True, True, "./tests4_3/"),
        # ("MS", False, True, False, "./tests4_4/"),
        # ("MS", False, False, False, "./tests4_5/"),
        # ("MS", True, True, True, "./tests4_3/"),
        # ("MS", True, True, False, "./tests4_4/"),
        # ("MS", True, False, False, "./tests4_5/"),
    ],
)
def test_valid_feature_imp(algorithm, run_parallel, use_turf, turf_pct, output_path):
    dataset_path, experiment_name = "./DemoData/", "demo",
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


@pytest.mark.parametrize(
    ("algorithms", "run_parallel", "output_path"),
    [
        (["MI", "MS"], False, "./tests5_1/"),
        # (["MI", "MS"], True, "./tests5_2/"),
    ],
)
def test_valid_feature_sel(algorithms, run_parallel, output_path):
    dataset_path, experiment_name = "./DemoData/", "demo",
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    eda = EDARunner(dataset_path, output_path, experiment_name, exploration_list=None, plot_list=None,
                    class_label="Class")
    eda.run(run_parallel=False)
    del eda

    dpr = DataProcessRunner(output_path, experiment_name)
    dpr.run(run_parallel=False)
    del dpr

    f_imp_mi = FeatureImportanceRunner(output_path, experiment_name, algorithm="MI")
    f_imp_mi.run(run_parallel=False)
    del f_imp_mi
    f_imp_ms = FeatureImportanceRunner(output_path, experiment_name, algorithm="MS")
    f_imp_ms.run(run_parallel=False)
    f_imp_ms

    start = time.time()

    logging.warning("Running Feature Selection")
    f_sel = FeatureSelectionRunner(output_path, experiment_name, algorithms, overwrite_cv=False)
    f_sel.run(run_parallel)

    if run_parallel:
        how = "parallely"
    else:
        how = "serially"
    logging.warning("Feature Selection Step with " + str(algorithms) +
                    ", Time running " + how + ": " + str(time.time() - start))

    shutil.rmtree(output_path)
