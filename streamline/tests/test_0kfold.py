import shutil
import pytest
import pandas as pd
from streamline.utils.dataset import Dataset
from streamline.dataprep.kfold_partitioning import KFoldPartitioner

pytest.skip("Tested Already", allow_module_level=True)

dataset_valid = Dataset("./DemoData/hcc-data_example.csv", "Class")


@pytest.mark.parametrize(
    ("dataset", "partition_method", "experiment_path", "exception"),
    [
        ("", "Random", "./tests/", Exception),
        (dataset_valid, "something", "./tests/", Exception),
        (dataset_valid, "Group", "./tests/", Exception),
    ],
)
def test_invalid_kfold(dataset, partition_method, experiment_path, exception):
    with pytest.raises(exception):
        KFoldPartitioner(dataset, partition_method, experiment_path)


def test_valid_kfold():
    partition_method, experiment_path = "Stratified", "./tests/"
    kfold = KFoldPartitioner(dataset_valid, partition_method, experiment_path, n_splits=5, random_state=42)
    train_dfs, test_dfs = kfold.cv_partitioner(return_dfs=False, save_dfs=False,
                                               partition_method=partition_method)
    assert (train_dfs is None)
    assert (test_dfs is None)
    train_dfs, test_dfs = kfold.cv_partitioner(return_dfs=True, save_dfs=False,
                                               partition_method=partition_method)
    for df in train_dfs + test_dfs:
        print(len(df))
        assert (type(df) is pd.DataFrame and len(df) != 0)

    train_dfs, test_dfs = kfold.cv_partitioner(return_dfs=False, save_dfs=True,
                                               partition_method=partition_method)
    shutil.rmtree('./tests/')
