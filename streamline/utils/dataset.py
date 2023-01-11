import pandas as pd
import logging


class Dataset:
    def __init__(self, dataset_path):
        """
        Creates dataset with path of tabular file

        Args:
            dataset_path: path of tabular file (as csv, tsv, or txt)

        """
        self.data = None
        self.path = dataset_path
        self.name = self.path.split('/')[-1].split('.')[0]
        self.format = self.path.split('/')[-1].split('.')[-1]
        self.load_data()

    def load_data(self):
        """
        Function to load data in dataset

        Returns: None
        """
        logging.log("Loading Dataset: "+str(self.name))
        if self.format == 'csv':
            self.data = pd.read_csv(self.path, na_values='NA', sep=',')
        elif self.format == 'tsv':
            self.data = pd.read_csv(self.path, na_values='NA', sep='\t')
        elif self.format == 'txt':
            self.data = pd.read_csv(self.path, na_values='NA', sep=' ')
        else:
            raise Exception("Unknown file format")

    def clean_data(self):
        pass
