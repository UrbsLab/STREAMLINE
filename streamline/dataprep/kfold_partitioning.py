import os
import csv
from streamline.utils.job import Job
from streamline.utils.dataset import Dataset
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold


class KFoldPartitioner(Job):
    """
    Base class for KFold CrossValidation Operations on dataset, Initialization for KFoldPartitioner base class
    """

    def __init__(self, dataset, partition_method, experiment_path, n_splits=10, random_state=None):
        """

        Args:
            dataset: a streamline.utils.dataset.Dataset object or a path to dataset text file
            partition_method: KFold CV method used for partitioning, must be one of ["Random", "Stratified", "Group"]
            experiment_path: path to experiment the logging directory folder
            n_splits: number of splits in k-fold cross validation
            random_state: random seed parameter for data reproducibility
        """
        super().__init__()
        assert (type(dataset) == Dataset)
        self.dataset = dataset
        self.dataset_path = dataset.path
        self.experiment_path = experiment_path
        self.n_splits = n_splits
        self.random_state = random_state

        self.supported_ptn_methods = ["Random", "Stratified", "Group"]

        if partition_method not in self.supported_ptn_methods:
            raise Exception('Error: Unknown partition method.')
        if partition_method == "Group" and self.dataset.match_label is None:
            raise Exception("No Match Label in dataset")

        self.partition_method = partition_method
        self.train_dfs = None
        self.test_dfs = None
        self.cv = None

    def cv_partitioner(self, return_dfs=True, save_dfs=True, partition_method=None):
        """

        Takes data frame (data), number of cv partitions, partition method (R, S, or M), class label,
        and the column name used for matched CV. Returns list of training and testing dataframe partitions.

        Args:
            return_dfs: flag to return splits as list of dataframe, returns empty list if set to False
            save_dfs: save dataframes in experiment path folder
            partition_method: override default partition method

        Returns: train_df, test_df both list of dataframes of train and test splits

        """

        if partition_method:
            self.partition_method = partition_method

        train_dfs, test_dfs = list(), list()

        # Random Partitioning Method
        if self.partition_method == 'Random':
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        # Stratified Partitioning Method
        elif self.partition_method == 'Stratified':
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        # Group Partitioning Method
        elif self.partition_method == 'Group':
            cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        else:
            raise Exception('Error: Requested partition method not found.')

        self.cv = cv

        if return_dfs:
            if self.partition_method == "Group":
                if self.dataset.match_label is None:
                    raise Exception("No Match Label in dataset")
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
            self.train_dfs = train_dfs
            self.test_dfs = test_dfs

        if save_dfs:
            self.save_datasets(self.experiment_path, self.train_dfs, self.test_dfs)

        return self.train_dfs, self.test_dfs

    def save_datasets(self, experiment_path=None, train_dfs=None, test_dfs=None):
        """ Saves individual training and testing CV datasets as .csv files"""
        # Generate folder to contain generated CV datasets

        if experiment_path is None:
            experiment_path = self.experiment_path

        train_dfs, test_dfs = train_dfs, test_dfs

        if train_dfs is None and test_dfs is None:
            if self.train_dfs is None and self.test_dfs is None:
                train_dfs, test_dfs = list(), list()
                if self.partition_method == "Group":
                    for train_index, test_index in self.cv.split(self.dataset.feature_only_data(),
                                                                 self.dataset.data[self.dataset.class_label],
                                                                 self.dataset.data[self.dataset.match_label]):
                        train_dfs.append(self.dataset.data.iloc[train_index, :])
                        test_dfs.append(self.dataset.data.iloc[test_index, :])
                else:
                    for train_index, test_index in self.cv.split(self.dataset.feature_only_data(),
                                                                 self.dataset.data[self.dataset.class_label]):
                        train_dfs.append(self.dataset.data.iloc[train_index, :])
                        test_dfs.append(self.dataset.data.iloc[test_index, :])
            else:
                train_dfs, test_dfs = self.train_dfs, self.test_dfs

        if not os.path.exists(experiment_path + '/' + self.dataset.name + '/CVDatasets'):
            os.makedirs(experiment_path + '/' + self.dataset.name + '/CVDatasets')

        # Export training datasets
        counter = 0
        for df in train_dfs:
            file = experiment_path + '/' + self.dataset.name + '/CVDatasets/' + self.dataset.name \
                   + '_CV_' + str(counter) + "_Train.csv"
            df.to_csv(file, index=False)
            counter += 1

        counter = 0
        for df in test_dfs:
            file = experiment_path + '/' + self.dataset.name + '/CVDatasets/' + self.dataset.name \
                   + '_CV_' + str(counter) + "_Test.csv"
            df.to_csv(file, index=False)
            counter += 1

    def run(self):
        self.cv_partitioner()
        job_file = open(self.experiment_path + '/jobsCompleted/job_exploratory_' + self.dataset.name + '.txt', 'w')
        job_file.write('complete')
        job_file.close()
