import os
import pickle
import re
import glob
import shutil
import multiprocessing
from streamline.utils.dataset import Dataset
from streamline.dataprep.exploratory_analysis import ExploratoryDataAnalysis
from streamline.dataprep.kfold_partitioning import KFoldPartitioner
from streamline.utils.runners import parallel_kfold_call, parallel_eda_call
from streamline.utils.runners import run_jobs


class EDARunner:
    """
    Description: Phase 1 of STREAMLINE - This 'Main' script manages Phase 1 run parameters, \
    updates the metadata file (with user specified run parameters across pipeline run) \
             and submits job to run locally (to run serially) or on a linux computing \
             cluster (parallelized).  This script runs ExploratoryAnalysisJob.py which conducts initial \
             exploratory analysis of data and cross validation (CV) partitioning. Note \
             that this entire pipeline may also be run within Jupyter Notebook (see STREAMLINE-Notebook.ipynb). \
             All 'Main' scripts in this pipeline have the potential to be extended by \
             users to submit jobs to other parallel computing frameworks (e.g. cloud computing). \

    Warnings:
        - Before running, be sure to check that all run parameters have relevant/desired values including those with\
            default values available.
        - 'Target' datasets for analysis should be in comma-separated format (.txt or .csv)
        - Missing data values should be empty or indicated with an 'NA'.
        - Dataset(s) includes a header giving column labels.
        - Data columns include features, class label, and optionally instance (i.e. row) labels, or match labels\
            (if matched cross validation will be used)
        - Binary class values are encoded as 0 (e.g. negative), and 1 (positive) with respect to true positive, \
            true negative, false positive, false negative metrics. PRC plots focus on classification of 'positives'.
        - All feature values (both categorical and quantitative) are numerically encoded. Scikit-learn does not accept \
            text-based values. However, both instance_label and match_label values may be either numeric or text.
        - One or more target datasets for analysis should be included in the same data_path folder. The path to this \
            folder is a critical pipeline run parameter. No spaces are allowed in filenames (this will lead to
          'invalid literal' by export_exploratory_analysis. If multiple datasets are being analyzed they must have the \
            same class_label, and (if present) the same instance_label and match_label.

    """

    def __init__(self, data_path, output_path, experiment_name, exploration_list=None, plot_list=None,
                 class_label="Class", instance_label=None, match_label=None, n_splits=10, partition_method="Stratified",
                 ignore_features=None, categorical_features=None,
                 categorical_cutoff=10, sig_cutoff=0.05,
                 random_state=None):
        """
        Initializer for a runner class for Exploratory Data Analysis Jobs

        Args:
            data_path: path to directory containing datasets
            output_path: path to output directory
            experiment_name: name of experiment output folder (no spaces)
            exploration_list: list of names of analysis to do while doing EDA (must be in set \
                                ["Describe", "Differentiate", "Univariate Analysis"])
            plot_list: list of analysis plots to save in experiment directory (must be in set \
                                ["Describe", "Univariate Analysis", "Feature Correlation"])
            class_label: outcome label of all datasets
            instance_label: instance label of all datasets (if present)
            match_label: only applies when M selected for partition-method; indicates column with \
                            matched instance ids
            n_splits: no of splits in cross-validation
            partition_method: method of partitioning in cross-validation must be in ["Random", "Stratified", "Group"]
            ignore_features: list of string of column names of features to ignore or \
                            path to .csv file with feature labels to be ignored in analysis
            categorical_features: list of string of column names of features to ignore or \
                            path to .csv file with feature labels specified to be treated as categorical where possible
            categorical_cutoff: number of unique values for a variable is considered to be quantitative vs categorical
            sig_cutoff: significance cutoff used throughout pipeline
            random_state: sets a specific random seed for reproducible results
        """

        self.data_path = data_path
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.class_label = class_label
        self.instance_label = instance_label
        self.match_label = match_label
        self.ignore_features = ignore_features
        self.categorical_cutoff = categorical_cutoff
        self.categorical_features = categorical_features
        self.exploration_list = exploration_list
        self.plot_list = plot_list
        self.n_splits = n_splits
        self.partition_method = partition_method

        if self.exploration_list is None or self.exploration_list == []:
            self.explorations_list = ["Describe", "Differentiate", "Univariate Analysis"]
        if self.plot_list is None or self.plot_list == []:
            self.plot_list = ["Describe", "Univariate Analysis", "Feature Correlation"]
        self.random_state = random_state
        self.sig_cutoff = sig_cutoff
        try:
            self.make_dir_tree()
        except Exception as e:
            shutil.rmtree(self.output_path)
            raise e
        self.save_metadata()

    def run(self, run_parallel=True):
        file_count, job_counter = 0, 0
        unique_datanames = []
        job_list, job_obj_list = [], []
        for dataset_path in glob.glob(self.data_path + '/*'):
            # Save unique dataset names so that analysis is run only once if there
            # is both a .txt and .csv version of dataset with same name.
            file_extension = dataset_path.split('/')[-1].split('.')[-1]
            data_name = dataset_path.split('/')[-1].split('.')[0]

            if file_extension == 'txt' or file_extension == 'csv':
                if data_name not in unique_datanames:
                    unique_datanames.append(data_name)
                    file_count += 1
                    dataset = Dataset(dataset_path, self.class_label, self.match_label, self.instance_label)
                    job_obj = ExploratoryDataAnalysis(dataset, self.output_path + self.experiment_name,
                                                      self.ignore_features,
                                                      self.categorical_features, self.exploration_list, self.plot_list,
                                                      self.categorical_cutoff, self.sig_cutoff,
                                                      self.random_state)
                    job_obj_list.append(job_obj)
                    # Cluster vs Non Cluster irrelevant as now local jobs are parallel too
                    if run_parallel:  # Run as job in parallel
                        p = multiprocessing.Process(target=parallel_eda_call, args=(job_obj, {'top_features': 20}))
                        job_list.append(p)
                    else:  # Run job locally, serially
                        job_list.append(job_obj)
                    job_counter += 1
            if file_count == 0:  # Check that there was at least 1 dataset
                raise Exception("There must be at least one .txt or .csv dataset in data_path directory")
        run_jobs(job_list)
        self.run_kfold(job_obj_list, run_parallel)

    def run_kfold(self, eda_obj_list, run_parallel=True):
        """

        Args:
            eda_obj_list:
            run_parallel:

        Returns:

        """
        file_count, job_counter = 0, 0
        job_list, job_obj_list = [], []
        for obj in eda_obj_list:
            kfold_obj = KFoldPartitioner(obj.dataset,
                                         self.partition_method, self.output_path + self.experiment_name,
                                         self.n_splits, self.random_state)
            if run_parallel:  # Run as job in parallel
                p = multiprocessing.Process(target=parallel_kfold_call, args=(kfold_obj,))
                job_list.append(p)
            else:  # Run job locally, serially
                kfold_obj.run()
            job_counter += 1
        if run_parallel:
            run_jobs(job_list)

    def check_old(self):
        """
        Instead of running job, checks whether previously run jobs were successfully completed
        """
        datasets = os.listdir(self.output_path + "/" + self.experiment_name)
        datasets.remove('logs')
        datasets.remove('jobs')
        datasets.remove('jobsCompleted')
        if 'metadata.pickle' in datasets:
            datasets.remove('metadata.pickle')
        if 'DatasetComparisons' in datasets:
            datasets.remove('DatasetComparisons')
        phase1jobs = []
        for dataset in datasets:
            phase1jobs.append('job_exploratory_' + dataset + '.txt')
        for filename in glob.glob(
                self.output_path + "/" + self.experiment_name + '/jobsCompleted/job_exploratory*'):
            ref = filename.split('/')[-1]
            phase1jobs.remove(ref)
        for job in phase1jobs:
            print(job)
        if len(phase1jobs) == 0:
            print("All Phase 1 Jobs Completed")
        else:
            print("Above Phase 1 Jobs Not Completed")

    def make_dir_tree(self):
        """
        Checks existence of data folder path.
        Checks that experiment output folder does not already exist as well as validity of experiment_name parameter.
        Then generates initial output folder hierarchy.
        """
        # Check to make sure data_path exists and experiment name is valid & unique
        if not os.path.exists(self.data_path):
            raise Exception("Provided data_path does not exist")
        if os.path.exists(self.output_path + '/' + self.experiment_name):
            raise Exception(
                "Error: A folder with the specified experiment name already exists at "
                "" + self.output_path + '/' + self.experiment_name + '. This path/folder name must be unique.')
        if not re.match(r'^[A-Za-z0-9_]+$', self.experiment_name):
            raise Exception('Experiment Name must be alphanumeric')

        # Create output folder if it doesn't already exist
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        # Create Experiment folder, with log and job folders
        os.mkdir(self.output_path + '/' + self.experiment_name)
        os.mkdir(self.output_path + '/' + self.experiment_name + '/jobsCompleted')
        os.mkdir(self.output_path + '/' + self.experiment_name + '/jobs')
        os.mkdir(self.output_path + '/' + self.experiment_name + '/logs')

    def save_metadata(self):
        metadata = dict()
        metadata['Data Path'] = self.data_path
        metadata['Output Path'] = self.output_path
        metadata['Experiment Name'] = self.experiment_name
        metadata['Class Label'] = self.class_label
        metadata['Instance Label'] = self.instance_label
        metadata['Ignored Features'] = self.ignore_features
        metadata['Specified Categorical Features'] = self.categorical_features
        metadata['CV Partitions'] = self.n_splits
        metadata['Partition Method'] = self.partition_method
        metadata['Match Label'] = self.match_label
        metadata['Categorical Cutoff'] = self.categorical_cutoff
        metadata['Statistical Significance Cutoff'] = self.sig_cutoff
        metadata['Export Feature Correlations'] = "Feature Correlations" in self.plot_list
        metadata['Export Univariate Plots'] = "Univariate Analysis" in self.plot_list
        metadata['Random Seed'] = self.random_state
        metadata['Run From Jupyter Notebook'] = False
        # Pickle the metadata for future use
        pickle_out = open(self.output_path + '/' + self.experiment_name + '/' + "metadata.pickle", 'wb')
        pickle.dump(metadata, pickle_out)
        pickle_out.close()
