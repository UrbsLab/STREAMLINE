import re
import os
import shutil
import time
import glob
import pickle
import random
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import multiprocessing
import matplotlib.pyplot as plt
from streamline.utils.job import Job
from streamline.utils.dataset import Dataset
from scipy.stats import chi2_contingency, mannwhitneyu


class ExploratoryDataAnalysis(Job):
    """
    Exploratory Data Analysis Class for the EDA/Phase 1 step of STREAMLINE
    """

    def __init__(self, dataset, experiment_path, ignore_features=None,
                 categorical_features=None, explorations=None, plots=None,
                 categorical_cutoff=10, sig_cutoff=0.1,
                 random_state=None):
        """
        Initialization function for Exploratory Data Analysis Class. Parameters are defined below.

        Args:
            dataset: a streamline.utils.dataset.Dataset object or a path to dataset text file
            experiment_path: path to experiment the logging directory folder
            ignore_features: list of row names of features to ignore
            categorical_features: list of row names of categorical features
            explorations: list of names of analysis to do while doing EDA (must be in set X)
            plots: list of analysis plots to save in experiment directory (must be in set Y)
            categorical_cutoff: categorical cut off to consider a feature categorical by analysis
            sig_cutoff: significance cutoff for continuous variables
            random_state: random state to set seeds for reproducibility of algorithms
        """
        super().__init__()
        if type(dataset) != Dataset:
            raise (Exception("dataset input is not of type Dataset"))
        self.dataset = dataset
        self.dataset_path = dataset.path
        self.experiment_path = experiment_path
        self.random_state = random_state
        explorations_list = ["Describe", "Differentiate", "Univariate Analysis"]
        plot_list = ["Describe", "Univariate Analysis", "Feature Correlation"]

        # Allows user to specify features that should be ignored.
        if ignore_features is None:
            self.ignore_features = []
        elif type(ignore_features) == str:
            ignore_features = pd.read_csv(ignore_features, sep=',')
            self.ignore_features = list(ignore_features)
        elif type(ignore_features) == list:
            self.ignore_features = ignore_features
        else:
            raise Exception

        # Allows user to specify features that should be treated as categorical whenever possible,
        # rather than relying on pipelines automated strategy for distinguishing categorical vs.
        # quantitative features using the categorical_cutoff parameter.
        if categorical_features is None:
            self.categorical_features = []
        elif type(categorical_features) == str:
            categorical_features = pd.read_csv(categorical_features, sep=',')
            self.categorical_features = list(categorical_features)
        elif type(categorical_features) == list:
            self.categorical_features = categorical_features
        else:
            raise Exception

        self.categorical_cutoff = categorical_cutoff
        self.sig_cutoff = sig_cutoff

        self.explorations = explorations
        if self.explorations is None:
            self.explorations = explorations_list
        self.plots = plots
        if self.plots is None:
            self.plots = plot_list

        for x in self.explorations:
            if x not in explorations_list:
                raise Exception("Exploration " + str(x) + " is not known/implemented")
        for x in self.explorations:
            if x not in explorations_list:
                raise Exception("Plot " + str(x) + " is not known/implemented")

    def make_log_folders(self):
        """
        Makes folders for logging exploratory data analysis
        """
        if not os.path.exists(self.experiment_path + '/' + self.dataset.name):
            os.makedirs(self.experiment_path + '/' + self.dataset.name)
        if not os.path.exists(self.experiment_path + '/' + self.dataset.name + '/exploratory'):
            os.makedirs(self.experiment_path + '/' + self.dataset.name + '/exploratory')

    def run_explore(self, top_features=20):
        """
        Run Exploratory Data Analysis according to EDA object

        Args:
            top_features: no of top features to consider (default=20)

        """
        self.job_start_time = time.time()
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        # Load csv file as dataset object for exploratory analysis
        self.dataset.load_data()
        # Make analysis folder for target dataset and a folder for the respective exploratory analysis within it
        self.make_log_folders()

        self.drop_ignored_rowcols()

        # Account for possibility that only one dataset in folder has a match label.
        # Check for presence of match label (this allows multiple datasets to be analyzed
        # in the pipeline where not all of them have match labels if specified)
        if not (self.dataset.match_label is None or self.dataset.match_label in self.dataset.data.columns):
            self.dataset.match_label = None
            self.dataset.partition_method = 'S'
            logging.warning("Warning: Specified 'Match label' could not be found in dataset. "
                            "Analysis moving forward assuming there is no 'match label' column using "
                            "stratified (S) CV partitioning.")

        # Create features-only version of dataset for some operations
        x_data = self.dataset.feature_only_data()

        if len(self.categorical_features) == 0:
            self.categorical_features = self.identify_feature_types(x_data)

        self.dataset.categorical_variables = self.categorical_features

        logging.info("Running Basic Exploratory Analysis...")

        # Describe and save description if user specified
        if "Describe" in self.explorations:
            self.describe_data()
            total_missing = self.missingness_counts()
            plot = False
            if "Describe" in self.plots:
                plot = True
            self.counts_summary(total_missing, plot)

        # Export feature correlation plot if user specified
        if "Feature Correlation" in self.plots:
            logging.info("Generating Feature Correlation Heatmap...")
            self.feature_correlation_plot(x_data)

        # Conduct univariate analyses of association between individual features and class
        if "Univariate analysis" in self.explorations:
            logging.info("Running Univariate Analyses...")
            sorted_p_list = self.univariate_analysis(top_features)
            # Export univariate association plots (for significant features) if user specifies
            if "Univariate analysis" in self.plots:
                logging.info("Generating Univariate Analysis Plots...")
                self.univariate_plots(sorted_p_list)

    def drop_ignored_rowcols(self):
        """
        Basic data cleaning: Drops any instances with a missing outcome
        value as well as any features (ignore_features) specified by user
        """
        # Remove instances with missing outcome values
        self.dataset.clean_data(self.ignore_features)

    def identify_feature_types(self, x_data=None):
        """
        Automatically identify categorical vs. quantitative features/variables
        Takes a dataframe (of independent variables) with column labels and
        returns a list of column names identified as
        being categorical based on user defined cutoff (categorical_cutoff).
        """
        # Identify categorical variables in dataset
        logging.info("Identifying Feature Types...")
        # Runs unless user has specified a predefined list of variables to treat as categorical

        if x_data is None:
            x_data = self.dataset.feature_only_data()
        categorical_variables = []
        if len(self.categorical_features) == 0:
            for each in x_data:
                if x_data[each].nunique() <= self.categorical_cutoff \
                        or not pd.api.types.is_numeric_dtype(x_data[each]):
                    categorical_variables.append(each)
            self.dataset.categorical_variables = self.categorical_features
        else:
            self.dataset.categorical_variables = self.categorical_features
            categorical_variables = self.categorical_features

        # Pickle list of feature names to be treated as categorical variables
        with open(self.experiment_path + '/' + self.dataset.name +
                  '/exploratory/categorical_variables.pickle', 'wb') as outfile:
            pickle.dump(categorical_variables, outfile)

        return categorical_variables

    def describe_data(self):
        """
        Conduct and export basic dataset descriptions including basic column statistics, column variable types
        (i.e. int64 vs. float64), and unique value counts for each column
        """
        self.dataset.data.describe().to_csv(self.experiment_path + '/' + self.dataset.name +
                                            '/exploratory/' + 'DescribeDataset.csv')
        self.dataset.data.dtypes.to_csv(self.experiment_path + '/' + self.dataset.name +
                                        '/exploratory/' + 'DtypesDataset.csv',
                                        header=['DataType'], index_label='Variable')
        self.dataset.data.nunique().to_csv(self.experiment_path + '/' + self.dataset.name +
                                           '/exploratory/' + 'NumUniqueDataset.csv',
                                           header=['Count'], index_label='Variable')

    def missingness_counts(self):
        """
        Count and export missing values for all data columns.
        """
        # Assess Missingness in all data columns
        missing_count = self.dataset.data.isnull().sum()
        total_missing = self.dataset.data.isnull().sum().sum()
        missing_count.to_csv(self.experiment_path + '/' + self.dataset.name + '/exploratory/' + 'DataMissingness.csv',
                             header=['Count'], index_label='Variable')
        return total_missing

    def missing_count_plot(self, plot=False):
        """
        Plots a histogram of missingness across all data columns.
        """
        missing_count = self.dataset.data.isnull().sum()
        # Plot a histogram of the missingness observed over all columns in the dataset
        plt.hist(missing_count, bins=100)
        plt.xlabel("Missing Value Counts")
        plt.ylabel("Frequency")
        plt.title("Histogram of Missing Value Counts in Dataset")
        plt.savefig(self.experiment_path + '/' + self.dataset.name + '/exploratory/' + 'DataMissingnessHistogram.png',
                    bbox_inches='tight')
        if plot:
            plt.show()

    def counts_summary(self, total_missing=None, plot=False, show=False):
        """
        Reports various dataset counts: i.e. number of instances, total features, categorical features, quantitative
        features, and class counts. Also saves a simple bar graph of class counts if user specified.

        Args:
            total_missing: total missing values (optional, runs again if not given)
            plot: flag to output bar graph in the experiment log folder
            show: flag to output the bar graph in interactive interface

        Returns:

        """
        # Calculate, print, and export instance and feature counts
        f_count = self.dataset.data.shape[1] - 1
        if not (self.dataset.instance_label is None):
            f_count -= 1
        if not (self.dataset.match_label is None):
            f_count -= 1
        if total_missing is None:
            total_missing = self.missingness_counts()
        percent_missing = int(total_missing) / float(self.dataset.data.shape[0] * f_count)
        summary = [['instances', self.dataset.data.shape[0]],
                   ['features', f_count],
                   ['categorical_features', len(self.dataset.categorical_variables)],
                   ['quantitative_features', f_count - len(self.dataset.categorical_variables)],
                   ['missing_values', total_missing],
                   ['missing_percent', round(percent_missing, 5)]]

        summary_df = pd.DataFrame(summary, columns=['Variable', 'Count'])

        summary_df.to_csv(self.experiment_path + '/' + self.dataset.name + '/exploratory/' + 'DataCounts.csv',
                          index=False)
        # Calculate, print, and export class counts
        class_counts = self.dataset.data[self.dataset.class_label].value_counts()
        class_counts.to_csv(self.experiment_path + '/' + self.dataset.name +
                            '/exploratory/' + 'ClassCounts.csv', header=['Count'],
                            index_label='Class')

        logging.info('Data Counts: ----------------')
        logging.info('Instance Count = ' + str(self.dataset.data.shape[0]))
        logging.info('Feature Count = ' + str(f_count))
        logging.info('    Categorical  = ' + str(len(self.dataset.categorical_variables)))
        logging.info('    Quantitative = ' + str(f_count - len(self.dataset.categorical_variables)))
        logging.info('Missing Count = ' + str(total_missing))
        logging.info('    Missing Percent = ' + str(percent_missing))
        logging.info('Class Counts: ----------------')
        logging.info('Class Count Information' + str(class_counts))

        # Generate and export class count bar graph
        if plot:
            class_counts.plot(kind='bar')
            plt.ylabel('Count')
            plt.title('Class Counts')
            plt.savefig(self.experiment_path + '/' + self.dataset.name + '/exploratory/' + 'ClassCountsBarPlot.png',
                        bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close('all')

    def feature_correlation_plot(self, x_data=None, show=False):
        """
        Calculates feature correlations via pearson correlation and exports a respective heatmap visualization.
        Due to computational expense this may not be recommended for datasets with a large number of instances
        and/or features unless needed. The generated heatmap will be difficult to read with a large number
        of features in the target dataset.

        Args:
            x_data: data with only feature columns
            show: flag to show plot or not
        """
        if x_data is None:
            x_data = self.dataset.feature_only_data()
        # Calculate correlation matrix
        correlation_mat = x_data.corr(method='pearson')
        # Generate and export correlation heatmap
        plt.subplots(figsize=(40, 20))
        sns.heatmap(correlation_mat, vmax=1, square=True)
        plt.savefig(self.experiment_path + '/' + self.dataset.name + '/exploratory/' + 'FeatureCorrelations.png',
                    bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close('all')

    def univariate_analysis(self, top_features=20):
        """
        Calculates univariate association significance between each individual feature and class outcome.
        Assumes categorical outcome using Chi-square test for
        categorical features and Mann-Whitney Test for quantitative features.

        Args:
            top_features: no of top features to show/consider

        """
        try:
            # Try loop added to deal with versions specific change to using
            # mannwhitneyu in scipy and avoid STREAMLINE crash in those circumstances.
            # Create folder for univariate analysis results
            if not os.path.exists(self.experiment_path + '/' + self.dataset.name
                                  + '/exploratory/univariate_analyses'):
                os.mkdir(self.experiment_path + '/' + self.dataset.name
                         + '/exploratory/univariate_analyses')
            # Generate dictionary of p-values for each feature using appropriate test (via test_selector)
            p_value_dict = {}
            for column in self.dataset.data:
                if column != self.dataset.class_label and column != self.dataset.instance_label:
                    p_value_dict[column] = self.test_selector(column)

            sorted_p_list = sorted(p_value_dict.items(), key=lambda item: item[1])
            # Save p-values to file
            pval_df = pd.DataFrame.from_dict(p_value_dict, orient='index')
            pval_df.to_csv(
                self.experiment_path + '/' + self.dataset.name
                + '/exploratory/univariate_analyses/Univariate_Significance.csv',
                index_label='Feature', header=['p-value'])

            # Print results for top features across univariate analyses
            f_count = self.dataset.data.shape[1] - 1
            if not (self.dataset.instance_label is None):
                f_count -= 1
            if not (self.dataset.match_label is None):
                f_count -= 1

            min_num = min(top_features, f_count)
            sorted_p_list_temp = sorted_p_list[: min_num]
            logging.info('Plotting top significant ' + str(min_num) + ' features.')
            logging.info('###################################################')
            logging.info('Significant Univariate Associations:')
            for each in sorted_p_list_temp[:min_num]:
                logging.info(each[0] + ": (p-val = " + str(each[1]) + ")")

        except Exception:
            sorted_p_list = []  # won't actually be sorted
            logging.warning('WARNING: Exploratory univariate analysis failed due to scipy package '
                            'version error when running mannwhitneyu test. '
                            'To fix, we recommend updating scipy to version 1.8.0 or greater '
                            'using: pip install --upgrade scipy')
            for column in self.dataset.data:
                if column != self.dataset.class_label and column != self.dataset.instance_label:
                    sorted_p_list.append([column, 'None'])

        return sorted_p_list

    def univariate_plots(self, sorted_p_list=None, top_features=20):
        """
        Checks whether p-value of each feature is less than or equal to significance cutoff.
        If so, calls graph_selector to generate an appropriate plot.

        Args:
            sorted_p_list: sorted list of p-values
            top_features: no of top features to consider (default=20)

        """

        if sorted_p_list is None:
            sorted_p_list = self.univariate_analysis(top_features)

        for i in sorted_p_list:  # each feature in sorted p-value dictionary
            if i[1] == 'None':
                pass
            else:
                for j in self.dataset.data:  # each feature
                    if j == i[0] and i[1] <= self.sig_cutoff:  # ONLY EXPORTS SIGNIFICANT FEATURES
                        self.graph_selector(j)

    def graph_selector(self, feature_name):
        """
        Assuming a categorical class outcome, a
        barplot is generated given a categorical feature, and a boxplot is generated given a quantitative feature.

        Args:
            feature_name: feature name of the column the function is doing operation on

        """
        # Feature and Outcome are discrete/categorical/binary
        if feature_name in self.dataset.categorical_variables:
            # Generate contingency table count bar plot.
            # Calculate Contingency Table - Counts
            table = pd.crosstab(self.dataset.data[feature_name], self.dataset.data[self.dataset.class_label])
            geom_bar_data = pd.DataFrame(table)
            geom_bar_data.plot(kind='bar')
            plt.ylabel('Count')
        else:
            # Feature is continuous and Outcome is discrete/categorical/binary
            # Generate boxplot
            self.dataset.data.boxplot(column=feature_name, by=self.dataset.class_label)
            plt.ylabel(feature_name)
            plt.title('')

        # Deal with the dataset specific characters causing problems in this dataset.
        new_feature_name = feature_name.replace(" ", "")
        new_feature_name = new_feature_name.replace("*", "")
        new_feature_name = new_feature_name.replace("/", "")
        plt.savefig(self.experiment_path + '/' + self.dataset.name
                    + '/exploratory/univariate_analyses/' + 'Barplot_' +
                    str(new_feature_name) + ".png", bbox_inches="tight", format='png')
        plt.close('all')

    def test_selector(self, feature_name):
        """
        Selects and applies appropriate univariate association test for a given feature. Returns resulting p-value

        Args:
            feature_name: name of feature column operation is running on
        """
        class_label = self.dataset.class_label
        # Feature and Outcome are discrete/categorical/binary
        if feature_name in self.dataset.categorical_variables:
            # Calculate Contingency Table - Counts
            table_temp = pd.crosstab(self.dataset.data[feature_name], self.dataset.data[class_label])
            # Univariate association test (Chi Square Test of Independence - Non-parametric)
            c, p, dof, expected = chi2_contingency(table_temp)
            p_val = p
        # Feature is continuous and Outcome is discrete/categorical/binary
        else:
            # Univariate association test (Mann-Whitney Test - Non-parametric)
            try:  # works in scipy 1.5.0
                c, p = mannwhitneyu(
                    x=self.dataset.data[feature_name].loc[self.dataset.data[class_label] == 0],
                    y=self.dataset.data[feature_name].loc[self.dataset.data[class_label] == 1])
            except Exception:  # for scipy 1.8.0
                c, p = mannwhitneyu(
                    x=self.dataset.data[feature_name].loc[self.dataset.data[class_label] == 0],
                    y=self.dataset.data[feature_name].loc[self.dataset.data[class_label] == 1], nan_policy='omit')
            p_val = p
        return p_val

    def save_runtime(self):
        """
        Export runtime for this phase of the pipeline on current target dataset
        """
        runtime = str(time.time() - self.job_start_time)
        logging.log(0, "PHASE 1 Completed: Runtime=" + str(runtime))
        if not os.path.exists(self.experiment_path + '/' + self.dataset.name + '/runtime'):
            os.mkdir(self.experiment_path + '/' + self.dataset.name + '/runtime')
        runtime_file = open(self.experiment_path + '/' + self.dataset.name + '/runtime/runtime_exploratory.txt', 'w')
        runtime_file.write(runtime)
        runtime_file.close()

    def run(self, top_features=20):
        """
        Wrapper function to run_explore

        Args:
            top_features: no of top features to consider (default=20)

        """
        self.run_explore(top_features)


def parallel_eda_call(eda_job, params):
    print("here", eda_job)
    if params and 'top_features' in params:
        eda_job.run(params['top_features'])
    else:
        eda_job.run()


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
                 class_label="Class", instance_label=None, match_label=None,
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
        if self.exploration_list is None:
            self.explorations_list = ["Describe", "Differentiate", "Univariate Analysis"]
        if self.plot_list is None:
            self.plot_list = ["Describe", "Univariate Analysis", "Feature Correlation"]
        self.random_state = random_state
        self.sig_cutoff = sig_cutoff
        try:
            self.make_dir_tree()
        except Exception as e:
            shutil.rmtree(self.output_path)
            raise e

    def run(self, run_parallel=True):
        file_count, job_counter = 0, 0
        unique_datanames = []
        job_list = []
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

                    # Cluster vs Non Cluster irrelevant as now local jobs are parallel too
                    if run_parallel:  # Run as job in parallel
                        p = multiprocessing.Process(target=parallel_eda_call, args=(job_obj, {}))
                        job_list.append(p)
                    else:  # Run job locally, serially
                        job_obj.run()

                    job_counter += 1
            if file_count == 0:  # Check that there was at least 1 dataset
                raise Exception("There must be at least one .txt or .csv dataset in data_path directory")
        print(file_count, " Files Processed")
        self.run_jobs(job_list)

    @staticmethod
    def run_jobs(job_list):
        for j in job_list:
            j.start()
        for j in job_list:
            j.join()

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
