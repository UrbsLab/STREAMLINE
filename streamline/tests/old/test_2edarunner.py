import os
import time
import pytest
import shutil
import logging
from streamline.runners.dataprocess_runner import DataProcessRunner

pytest.skip("Tested Already", allow_module_level=True)


@pytest.mark.parametrize(
    ("dataset", "output_path", "experiment_name", "exception"),
    [
        ("./random_folder/", "./tests/", 'demo', Exception),
        ("./DemoData/", "./tests/", ".@!#@", Exception),
    ],
)
def test_invalid_eda(dataset, output_path, experiment_name, exception):
    with pytest.raises(exception):
        DataProcessRunner(dataset, output_path, experiment_name)


def test_valid_eda():
    if not os.path.exists('./tests1/'):
        os.mkdir('./tests1/')

    start = time.time()
    eda = DataProcessRunner("./DemoData/", "./tests1/", 'demo', exploration_list=None, plot_list=None,
                            class_label="Class", instance_label="InstanceID", ignore_features=["Alcohol"])
    eda.run(run_parallel=False)
    logging.warning("Exploratory Data Analysis, Time running serially: " + str(time.time() - start))

    shutil.rmtree('./tests1/')

    if not os.path.exists('./tests2/'):
        os.mkdir('./tests2/')

    start = time.time()
    eda = DataProcessRunner("./DemoData/", "./tests2/", 'demo', exploration_list=None, plot_list=None,
                            class_label="Class", instance_label="InstanceID", ignore_features=["Alcohol"])
    eda.run(run_parallel=True)
    logging.warning("Exploratory Data Analysis, Time running parallely: " + str(time.time() - start))
    shutil.rmtree('./tests2/')
