import os
import pytest
from streamline.utils.checker import FN_LIST

pytest.skip("Tested Already", allow_module_level=True)

output_path, experiment_name = './tests', 'demo'
datasets = os.listdir(output_path + "/" + experiment_name)
remove_list = ['.DS_Store', 'metadata.pickle', 'metadata.csv', 'algInfo.pickle', 'jobsCompleted',
               'dask_logs', 'logs', 'jobs',
               'DatasetComparisons', 'UsefulNotebooks',
               experiment_name + '_ML_Pipeline_Report.pdf']
for text in remove_list:
    if text in datasets:
        datasets.remove(text)


@pytest.mark.parametrize(
    ("fn", "left"),
    [
        (FN_LIST[i], 0) for i in range(len(FN_LIST))
    ]
)
def test_checker(fn, left):
    output = fn(output_path, experiment_name, datasets)
    assert (len(output) == left)
