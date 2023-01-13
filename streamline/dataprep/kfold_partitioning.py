import os
import csv
from streamline.utils.job import Job
from streamline.utils.dataset import Dataset
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold


class KFoldPartitioner(Job):
    """
    Base class for KFold CrossValidation Operations on dataset
    """

    def __init__(self, dataset, partition_method, experiment_path):
        """
        Initialization for KFoldPartitioner base class

        Args:
            dataset: a streamline.utils.dataset.Dataset object or a path to dataset text file
            partition_method: KFold CV method used for partitioning, must be one of ["Random", "Stratified", "Group"]
            experiment_path: path to experiment the logging directory folder
        """
        super().__init__()
        assert (type(dataset) == Dataset)
        self.dataset = dataset
        self.dataset_path = dataset.path
        self.experiment_path = experiment_path
        self.partition_method = partition_method
        self.cv = None

    def cv_partitioner(self, n_splits=5, random_state=42, return_dfs=True):
        """

        Takes data frame (data), number of cv partitions, partition method (R, S, or M), class label,
        and the column name used for matched CV. Returns list of training and testing dataframe partitions.

        Args:
            n_splits: number of splits in k-fold cross validation
            random_state: random seed parameter for data reproducibility
            return_dfs: flag to return splits as list of dataframe, returns empty list if set to False

        Returns: train_df, test_df both list of dataframes of train and test splits

        """

        train_dfs, test_dfs = list(), list()

        # Random Partitioning Method
        if self.partition_method == 'Random':
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        # Stratified Partitioning Method
        elif self.partition_method == 'Stratified':
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        # Group Partitioning Method
        elif self.partition_method == 'Group':
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            raise Exception('Error: Requested partition method not found.')

        if return_dfs:
            if self.partition_method == "Group":
                for train_index, test_index in cv.split(self.dataset.feature_only_data(),
                                                        self.dataset.data[self.dataset.class_label],
                                                        self.dataset.data[self.dataset.match_label]):
                    train_dfs.append(self.dataset.data.iloc[train_index, :])
                    test_dfs.append(self.dataset.data.iloc[test_index, :])
            else:
                for train_index, test_index in cv.split(self.dataset.feature_only_data(),
                                                        self.dataset.data[self.dataset.class_label]):
                    train_dfs.append(self.dataset.data.iloc[train_index, :])
                    test_dfs.append(self.dataset.data.iloc[test_index, :])

        self.cv = cv

        return train_dfs, test_dfs, cv

    def save_datasets(self, experiment_path, train_dfs, test_dfs):
        """ Saves individual training and testing CV datasets as .csv files"""
        # Generate folder to contain generated CV datasets
        if not os.path.exists(experiment_path + '/' + self.dataset.name + '/CVDatasets'):
            os.mkdir(experiment_path + '/' + self.dataset.name + '/CVDatasets')

        # Export training datasets
        counter = 0
        for df in train_dfs:
            file = experiment_path + '/' + self.dataset.name + '/CVDatasets/' + self.dataset.name \
                   + '_CV_' + str(counter) + "_Train.csv"
            df.to_csv(file)

        counter = 0
        for df in train_dfs:
            file = experiment_path + '/' + self.dataset.name + '/CVDatasets/' + self.dataset.name \
                   + '_CV_' + str(counter) + "_Test.csv"
            df.to_csv(file)
