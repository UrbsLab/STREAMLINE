import os
import logging

import pandas as pd

from streamline.utils.job import Job
from streamline.models.basemodel import BaseMLModel


class ModelJob(Job):
    def __init__(self, model, train_file_path, output_path, experiment_name, class_label="Class", instance_label=None,
                 metric='primary_metric', n_trails=None, timeout=None):
        self.model = model
        self.train_file_path = train_file_path
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.class_label = class_label
        self.instance_label = instance_label
        self.metric = metric
        self.full_path = train_file_path

        # Argument checks
        if not os.path.exists(self.output_path):
            raise Exception("Output path must exist (from phase 1) before phase 5 can begin")
        if not os.path.exists(self.output_path + '/' + self.experiment_name):
            raise Exception("Experiment must exist (from phase 1) before phase 5 can begin")

        self.n_trails = n_trails
        self.timeout = timeout
        self.param_grid = None

    def run(self):
        """
        Specifies hardcoded (below) range of hyperparameter options selected for each ML algorithm and
        then runs the modeling method. Set up this way so that users can easily modify
        ML hyperparameter settings when running from the Jupyter Notebook.
        """
        logging.info('Running ' + str(self.model.name) + ' on ' + str(self.train_file_path))
        # Get header names for current CV dataset for use later in GP tree visulaization
        data_name = self.full_path.split('/')[-1]
        feature_names = pd.read_csv(
            full_path + '/CVDatasets/' + data_name + '_CV_' + str(cvCount) + '_Test.csv').columns.values.tolist()
        if self.instance_label is not None:
            feature_names.remove(self.instance_label)
        feature_names.remove(self.class_label)
        # Get hyperparameter grid
        #
        # runModel(algorithm, train_file_path, test_file_path, full_path, n_trials, timeout, lcs_timeout,
        #          export_hyper_sweep_plots, instance_label, class_label, random_state, cvCount, filter_poor_features,
        #          do_lcs_sweep, nu, iterations, N, training_subsample, use_uniform_FI, primary_metric, param_grid,
        #          algAbrev)

