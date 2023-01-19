import os
import time
import pytest
import shutil
import logging
from streamline.dataprep.eda_runner import EDARunner


@pytest.mark.parametrize(
    ("dataset", "output_path", "experiment_name", "exception"),
    [
        ("./random_folder/", "./tests/", 'demo', Exception),
        ("./DemoData/", "./tests/", ".@!#@", Exception),
    ],
)
def test_invalid_eda(dataset, output_path, experiment_name, exception):
    with pytest.raises(exception):
        EDARunner(dataset, output_path, experiment_name)


def test_valid_eda():
    if not os.path.exists('./tests2/'):
        os.mkdir('./tests2/')

    start = time.time()
    eda = EDARunner("./DemoData/", "./tests2/", 'demo', exploration_list=None, plot_list=None,
                    class_label="Class")
    eda.run(run_parallel=False)
    logging.warning("Exploratory Data Analysis, Time running serially: " + str(time.time() - start))

    shutil.rmtree('./tests2/')

    start = time.time()
    eda = EDARunner("./DemoData/", "./tests2/", 'demo', exploration_list=None, plot_list=None,
                    class_label="Class")
    eda.run(run_parallel=True)
    logging.warning("Exploratory Data Analysis, Time running parallely: " + str(time.time() - start))
    shutil.rmtree('./tests2/')
