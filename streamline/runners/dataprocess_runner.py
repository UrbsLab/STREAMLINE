import logging
import os
import pickle
import re
import glob
import time
import dask
from pathlib import Path
from streamline.utils.dataset import Dataset
from streamline.dataprep.data_process import DataProcess
from streamline.utils.runners import parallel_eda_call, num_cores
from joblib import Parallel, delayed
from streamline.utils.cluster import get_cluster


class DataProcessRunner:
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
            'invalid literal' by export_exploratory_analysis.) \
            If multiple datasets are being analyzed they must have the \
            same class_label, and (if present) the same instance_label and match_label.

    """

    def __init__(self, data_path, output_path, experiment_name, exclude_eda_output=None,
                 class_label="Class", instance_label=None, match_label=None, n_splits=10, partition_method="Stratified",
                 ignore_features=None, categorical_features=None, quantitative_features=None, top_features=20,
                 categorical_cutoff=10, sig_cutoff=0.05, featureeng_missingness=0.5, cleaning_missingness=0.5,
                 correlation_removal_threshold=1.0,
                 random_state=None, run_cluster=False, queue='defq', reserved_memory=4, show_plots=False):
        """
        Initializer for a runner class for Exploratory Data Analysis Jobs

        Args:
            data_path: path to directory containing datasets
            output_path: path to output directory
            experiment_name: name of experiment output folder (no spaces)
            exploration_list: list of names of analysis to do while doing EDA (must be in set \
                                ["Describe", "Univariate Analysis", "Feature Correlation"])
            plot_list: list of analysis plots to save in experiment directory (must be in set \
                                ["Describe", "Univariate Analysis", "Feature Correlation"])
            class_label: outcome label of all datasets
            instance_label: instance label of all datasets (if present)
            match_label: only applies when M selected for partition-method; indicates column with \
                            matched instance ids
            n_splits: no of splits in cross-validation (default=10)
            partition_method: method of partitioning in cross-validation must be in ["Random", "Stratified", "Group"]\
                                (default="Stratified")
            ignore_features: list of string of column names of features to ignore or \
                            path to .csv file with feature labels to be ignored in analysis (default=None)
            categorical_features: list of string of column names of features to ignore or \
                            path to .csv file with feature labels specified to be treated as categorical where possible\
                            (default=None)
            categorical_cutoff: number of unique values for a variable is considered to be quantitative vs categorical\
                            (default=10)
            sig_cutoff: significance cutoff used throughout pipeline (default=0.05)
            random_state: sets a specific random seed for reproducible results (default=None)
            run_cluster: name of cluster run setting or False (default=False)
            queue: name of queue to be used in cluster run (default="defq")
            reserved_memory: reserved memory for cluster run in GB (in default=4)
            show_plots: flag to output plots for notebooks (default=False)
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
        self.quantitative_features = quantitative_features
        self.featureeng_missingness = featureeng_missingness
        self.cleaning_missingness = cleaning_missingness
        self.correlation_removal_threshold = correlation_removal_threshold
        self.top_features = top_features
        self.exclude_eda_output = exclude_eda_output

        known_exclude_options = ['describe_csv', 'univariate_plots', 'correlation_plots']

        exploration_list = ["Describe", "Univariate Analysis", "Feature Correlation"]
        plot_list = ["Describe", "Univariate Analysis", "Feature Correlation"]

        if exclude_eda_output is not None:
            for x in exclude_eda_output:
                if x not in known_exclude_options:
                    logging.warning("Unknown EDA exclusion option " + str(x))
            if 'describe_csv' in exclude_eda_output:
                exploration_list.remove("Describe")
                plot_list.remove("Describe")
            if 'univariate_plots' in exclude_eda_output:
                plot_list.remove("Univariate Analysis")
            if 'correlation_plots' in exclude_eda_output:
                plot_list.remove("Feature Correlation")

        self.exploration_list = exploration_list
        self.plot_list = plot_list

        self.n_splits = n_splits
        self.partition_method = partition_method
        self.run_cluster = run_cluster
        self.queue = queue
        self.reserved_memory = reserved_memory
        self.show_plots = show_plots
        self.random_state = random_state
        self.sig_cutoff = sig_cutoff
        try:
            self.make_dir_tree()
        except Exception as e:
            # shutil.rmtree(self.output_path)
            raise e
        self.save_metadata()

    def run(self, run_parallel=False):
        file_count, job_counter = 0, 0
        unique_datanames = []
        job_obj_list = []
        for dataset_path in glob.glob(self.data_path + '/*'):
            dataset_path = str(Path(dataset_path).as_posix())
            # Save unique dataset names so that analysis is run only once if there
            # is both a .txt and .csv version of dataset with same name.
            file_extension = dataset_path.split('/')[-1].split('.')[-1]
            data_name = dataset_path.split('/')[-1].split('.')[0]

            if file_extension == 'txt' or file_extension == 'csv' or file_extension == 'tsv':
                if data_name not in unique_datanames:
                    unique_datanames.append(data_name)
                    file_count += 1

                    if not os.path.exists(self.output_path + '/' + self.experiment_name + '/' + data_name):
                        os.makedirs(self.output_path + '/' + self.experiment_name + '/' + data_name)

                    if self.run_cluster == "SLURMOld":
                        self.submit_slurm_cluster_job(dataset_path)
                        continue

                    if self.run_cluster == "LSFOld":
                        self.submit_lsf_cluster_job(dataset_path)
                        continue
                    dataset = Dataset(dataset_path, self.class_label, self.match_label, self.instance_label)
                    # Ryan - dataset loading has to take place on individual compute nodes
                    # (bare minimum can be running on head node for cluster parallelization)
                    job_obj = DataProcess(dataset, self.output_path + '/' + self.experiment_name,
                                          self.ignore_features,
                                          self.categorical_features, self.quantitative_features,
                                          self.exclude_eda_output,
                                          self.categorical_cutoff, self.sig_cutoff, self.featureeng_missingness,
                                          self.cleaning_missingness, self.correlation_removal_threshold,
                                          self.partition_method, self.n_splits,
                                          self.random_state, self.show_plots)
                    job_obj_list.append(job_obj)
                    # Cluster vs Non Cluster irrelevant as now local jobs are parallel too
                    if not run_parallel:  # Run as job in parallel
                        job_obj_list[-1].run(self.top_features)
                    job_counter += 1

            if file_count == 0:  # Check that there was at least 1 dataset
                raise Exception("There must be at least one .txt, .tsv, or .csv dataset in data_path directory")

        if run_parallel and run_parallel != "False" and not self.run_cluster:
            Parallel(n_jobs=num_cores)(
                delayed(
                    parallel_eda_call
                )(job_obj, {'top_features': self.top_features}) for job_obj in job_obj_list)

        if self.run_cluster and "Old" not in self.run_cluster:
            get_cluster(self.run_cluster,
                        self.output_path + '/' + self.experiment_name, self.queue, self.reserved_memory)
            dask.compute([dask.delayed(
                parallel_eda_call
            )(job_obj, {'top_features': self.top_features}) for job_obj in job_obj_list])

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
        metadata['Match Label'] = self.match_label
        metadata['Ignored Features'] = self.ignore_features
        metadata['Specified Categorical Features'] = self.categorical_features
        metadata['Specified Quantitative Features'] = self.quantitative_features
        metadata['CV Partitions'] = self.n_splits
        metadata['Partition Method'] = self.partition_method
        metadata['Categorical Cutoff'] = self.categorical_cutoff
        metadata['Statistical Significance Cutoff'] = self.sig_cutoff
        metadata['Feature Missingness Cutoff'] = self.featureeng_missingness
        metadata['Cleaning Missingness Cutoff'] = self.cleaning_missingness
        metadata['Correlation Removal Threshold'] = self.correlation_removal_threshold
        metadata['List of Exploratory Analysis Ran'] = self.exploration_list
        metadata['List of Exploratory Plots Saved'] = self.plot_list
        metadata['Random Seed'] = self.random_state
        metadata['Run From Notebook'] = self.show_plots
        # Pickle the metadata for future use
        pickle_out = open(self.output_path + '/' + self.experiment_name + '/' + "metadata.pickle", 'wb')
        pickle.dump(metadata, pickle_out)
        pickle_out.close()

    def get_cluster_params(self, dataset_path):
        exclude_param = ','.join(self.exclude_eda_output) if self.exclude_eda_output else None
        cluster_params = [dataset_path, self.output_path, self.experiment_name, exclude_param,
                          self.class_label, self.instance_label, self.match_label, self.n_splits,
                          self.partition_method, self.ignore_features, self.categorical_features,
                          self.quantitative_features, self.top_features,
                          self.categorical_cutoff, self.sig_cutoff, self.featureeng_missingness,
                          self.cleaning_missingness, self.correlation_removal_threshold, self.random_state]
        cluster_params = [str(i) if type(i) != list else '"' + str(i) + '"' for i in cluster_params]
        return cluster_params

    def submit_slurm_cluster_job(self, dataset_path):
        job_ref = str(time.time())
        job_name = self.output_path + '/' + self.experiment_name + '/jobs/P1_' + job_ref + '_run.sh'
        sh_file = open(job_name, 'w')
        sh_file.write('#!/bin/bash\n')
        sh_file.write('#SBATCH -p ' + self.queue + '\n')
        sh_file.write('#SBATCH --job-name=' + job_ref + '\n')
        sh_file.write('#SBATCH --mem=' + str(self.reserved_memory) + 'G' + '\n')
        # sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
        sh_file.write(
            '#SBATCH -o ' + self.output_path + '/' + self.experiment_name +
            '/logs/P1_' + job_ref + '.o\n')
        sh_file.write(
            '#SBATCH -e ' + self.output_path + '/' + self.experiment_name +
            '/logs/P1_' + job_ref + '.e\n')

        file_path = str(Path(__file__).parent.parent.parent) + "/streamline/legacy" + '/EDAJobSubmit.py'
        cluster_params = self.get_cluster_params(dataset_path)
        command = ' '.join(['srun', 'python', file_path] + cluster_params)
        sh_file.write(command + '\n')
        sh_file.close()
        os.system('sbatch ' + job_name)

    def submit_lsf_cluster_job(self, dataset_path):
        job_ref = str(time.time())
        job_name = self.output_path + '/' + self.experiment_name + '/jobs/P1_' + job_ref + '_run.sh'
        sh_file = open(job_name, 'w')
        sh_file.write('#!/bin/bash\n')
        sh_file.write('#BSUB -q ' + self.queue + '\n')
        sh_file.write('#BSUB -J ' + job_ref + '\n')
        sh_file.write('#BSUB -R "rusage[mem=' + str(self.reserved_memory) + 'G]"' + '\n')
        sh_file.write('#BSUB -M ' + str(self.reserved_memory) + 'GB' + '\n')
        sh_file.write(
            '#BSUB -o ' + self.output_path + '/' + self.experiment_name +
            '/logs/P1_' + job_ref + '.o\n')
        sh_file.write(
            '#BSUB -e ' + self.output_path + '/' + self.experiment_name +
            '/logs/P1_' + job_ref + '.e\n')

        file_path = str(Path(__file__).parent.parent.parent) + "/streamline/legacy" + '/EDAJobSubmit.py'
        cluster_params = self.get_cluster_params(dataset_path)
        command = ' '.join(['python', file_path] + cluster_params)
        sh_file.write(command + '\n')
        sh_file.close()
        os.system('bsub < ' + job_name)
