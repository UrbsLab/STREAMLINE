import os
import time
import pytest
import shutil
import logging
from streamline.runners.eda_runner import EDARunner
from streamline.runners.dataprocess_runner import DataProcessRunner

pytest.skip("Tested Already", allow_module_level=True)


@pytest.mark.parametrize(
    ("output_path", "experiment_name", "exception"),
    [
        ("./tests/", 'demo', Exception),
        ("./tests/", ".@!#@", Exception),
    ],
)
def test_invalid_datap(output_path, experiment_name, exception):
    with pytest.raises(exception):
        DataProcessRunner(output_path, experiment_name)


def test_valid_datap():
    if not os.path.exists('./tests3/'):
        os.mkdir('./tests3/')
    eda = EDARunner("./DemoData/", "./tests3/", 'demo', exploration_list=None, plot_list=None,
                    class_label="Class")
    eda.run(run_parallel=False)

    start = time.time()
    dpr = DataProcessRunner("./tests3/", 'demo')
    dpr.run(run_parallel=False)
    logging.warning("Data Scale and Impute, Time running serially: " + str(time.time() - start))

    shutil.rmtree('./tests3/')

    if not os.path.exists('./tests3/'):
        os.mkdir('./tests3/')
    eda = EDARunner("./DemoData/", "./tests3/", 'demo', exploration_list=None, plot_list=None,
                    class_label="Class")
    eda.run(run_parallel=True)

    start = time.time()
    dpr = DataProcessRunner("./tests3/", 'demo')
    dpr.run(run_parallel=True)
    logging.warning("Data Scale and Impute, Time running parallely: " + str(time.time() - start))

    shutil.rmtree('./tests3/')
