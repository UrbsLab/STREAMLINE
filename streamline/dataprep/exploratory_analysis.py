import os
import time
import pickle
import random
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamline.utils.job import Job
from streamline.utils.dataset import Dataset


class ExploratoryDataAnalysis(Job):
    """
    Exploratory Data Analysis Class for the EDA/Phase 1 step of STREAMLINE
    """

    def __init__(self, dataset, experiment_path, ignore_features=None,
                 categorical_features=None, explorations=None, plots=None,
                 categorical_cutoff=None, sig_cutoff=None,
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
            sig_cutoff:
            random_state: random state to set seeds for reproducibility of algorithms
        """
        super().__init__()
        if type(dataset) == str:
            self.dataset = Dataset(dataset)
        else:
            assert (type(dataset) == Dataset)
            self.dataset = dataset
        self.dataset_path = dataset.path
        self.experiment_path = experiment_path
        self.random_state = random_state
        explorations_list = ["Describe", "Differentiate", "Univariate Analysis"]
        plot_list = ["Univariate Analysis", "Feature Correlation"]

        # Allows user to specify features that should be ignored.
        if type(ignore_features) == type(None):
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
        if type(categorical_features) == type(None):
            self.categorical_features = []
        elif type(categorical_features) == str:
            categorical_features = pd.read_csv(categorical_features, sep=',')
            self.categorical_features = list(categorical_features)
        elif type(categorical_features) == list:
            self.categorical_features = categorical_features
        else:
            raise Exception

        self.categorical_cutoff = categorical_cutoff

        self.explorations = explorations
        if self.explorations is None:
            self.explorations = explorations_list
        self.plot = plots
        if self.plot is None:
            plots = plot_list

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
            os.mkdir(self.experiment_path + '/' + self.dataset.name)
        if not os.path.exists(self.experiment_path + '/' + self.dataset.name + '/exploratory'):
            os.mkdir(self.experiment_path + '/' + self.dataset.name + '/exploratory')

    def run_explore(self, match_label=None, class_label=None, instance_label=None, top_features=20):
        """
        Run Exploratory Data Analysis according to EDA object

        Args:
            match_label: ?
            class_label: ?
            instance_label: ?
            top_features: no of top features to consider (default=20)

        """
        job_start_time = time.time()
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        # Load csv file as dataset object for exploratory analysis
        self.dataset.load_data()
        # Make analysis folder for target dataset and a folder for the respective exploratory analysis within it
        self.make_log_folders()

        self.drop_ignored_rowcols(class_label, self.ignore_features)

        # Account for possibility that only one dataset in folder has a match label.
        # Check for presence of match label (this allows multiple datasets to be analyzed
        # in the pipeline where not all of them have match labels if specified)
        if not (match_label is None or match_label in self.dataset.data.columns):
            match_label = None
            partition_method = 'S'
            logging.warning("Warning: Specified 'Match label' could not be found in dataset. "
                            "Analysis moving forward assuming there is no 'match label' column using "
                            "stratified (S) CV partitioning.")

        # Create features-only version of dataset for some operations
        x_data = self.dataset.feature_only_data(match_label, class_label, instance_label)

        if len(self.categorical_features) == 0:
            self.categorical_features = self.identify_feature_types(x_data)

        self.data.catagorical_variables = self.categorical_features

        logging.log(0, "Running Basic Exploratory Analysis...")

        # Describe and save description if user specified
        if "Describe" in self.explorations:
            self.describe_data()
            total_missing = self.missingness_counts()
            plot = False
            if "Describe" in self.plots:
                plot = True
            self.counts_summary(match_label, class_label, instance_label, plot)

        # Export feature correlation plot if user specified
        if "Feature Correlation" in self.plots:
            logging.log(0, "Generating Feature Correlation Heatmap...")
            self.feature_correlation_plot(x_data)

        # Conduct univariate analyses of association between individual features and class
        if "Univariate analysis" in self.explorations:
            logging.log(0, "Running Univariate Analyses...")
            # sorted_p_list = univariateAnalysis(data, experiment_path, dataset_name, class_label, instance_label,
            #                                    match_label,
            #                                    categorical_variables, jupyterRun, topFeatures)

        # Export univariate association plots (for significant features) if user specifies
        if ("Univariate analysis" in self.explorations) and ("Univariate analysis" in self.plots):
            logging.log(0, "Generating Univariate Analysis Plots...")
            # univariatePlots(data, sorted_p_list, class_label, categorical_variables, experiment_path, dataset_name,
            #                 sig_cutoff)

    def drop_ignored_rowcols(self, class_label, ignore_features):
        """
        Basic data cleaning: Drops any instances with a missing outcome
        value as well as any features (ignore_features) specified by user
        """
        # Remove instances with missing outcome values
        self.dataset.clean_data(class_label, ignore_features)

    def identify_feature_types(self, x_data):
        """
        Automatically identify categorical vs. quantitative features/variables
        Takes a dataframe (of independent variables) with column labels and
        returns a list of column names identified as
        being categorical based on user defined cutoff (categorical_cutoff).
        """
        # Identify categorical variables in dataset
        logging.log(0, "Identifying Feature Types...")
        # Runs unless user has specified a predefined list of variables to treat as categorical
        if len(self.categorical_features) == 0:
            categorical_variables = []
            for each in x_data:
                if x_data[each].nunique() <= self.categorical_cutoff \
                        or not pd.api.types.is_numeric_dtype(x_data[each]):
                    categorical_variables.append(each)
        else:
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
        Count and export missing values for all data columns. Also plots a histogram of missingness across all data columns.
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

    def counts_summary(self, match_label, class_label, instance_label, plot=False, show=False):
        """
        Reports various dataset counts: i.e. number of instances, total features, categorical features, quantitative features, and class counts.
        Also saves a simple bar graph of class counts.

        Args:
            class_label:
            instance_label:
            match_label:
            plot:
            show:

        Returns:

        """
        # Calculate, print, and export instance and feature counts
        fCount = self.dataset.data.shape[1] - 1
        if instance_label is not None:
            fCount -= 1
        if match_label is not None:
            fCount -= 1
        total_missing = self.missingness_counts()
        percent_missing = int(total_missing) / float(self.dataset.data.shape[0] * fCount)
        summary = [['instances', self.dataset.data.shape[0]],
                   ['features', fCount],
                   ['categorical_features', len(self.dataset.categorical_variables)],
                   ['quantitative_features', fCount - len(self.dataset.categorical_variables)],
                   ['missing_values', total_missing],
                   ['missing_percent', round(percent_missing, 5)]]

        summary_df = pd.DataFrame(summary, columns=['Variable', 'Count'])

        summary_df.to_csv(self.experiment_path + '/' + self.dataset.name + '/exploratory/' + 'DataCounts.csv',
                          index=False)
        # Calculate, print, and export class counts
        class_counts = self.dataset.data[class_label].value_counts()
        class_counts.to_csv(self.experiment_path + '/' + self.dataset.name +
                            '/exploratory/' + 'ClassCounts.csv', header=['Count'],
                            index_label='Class')

        #     print('Data Counts: ----------------')
        #     print('Instance Count = ' + str(data.shape[0]))
        #     print('Feature Count = ' + str(fCount))
        #     print('    Categorical  = ' + str(len(categorical_variables)))
        #     print('    Quantitative = ' + str(fCount - len(categorical_variables)))
        #     print('Missing Count = ' + str(totalMissing))
        #     print('    Missing Percent = ' + str(percentMissing))
        #     print('Class Counts: ----------------')
        #     print(class_counts)

        # Generate and export class count bar graph
        if plot:
            class_counts.plot(kind='bar')
            plt.ylabel('Count')
            plt.title('Class Counts')
            plt.savefig(self.experiment_path + '/' + self.dataset_name + '/exploratory/' + 'ClassCountsBarPlot.png',
                        bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close('all')

    def feature_correlation_plot(self, x_data, show=False):
        """
        Calculates feature correlations via pearson correlation and exports a respective heatmap visualization.
        Due to computational expense this may not be recommended for datasets with a large number of instances
        and/or features unless needed. The generated heatmap will be difficult to read with a large number
        of features in the target dataset.

        Args:
            x_data: data with only feature columns
            show: flag to show plot or not
        """
        # Calculate correlation matrix
        correlation_mat = x_data.corr(method='pearson')
        # Generate and export correlation heatmap
        f, ax = plt.subplots(figsize=(40, 20))
        sns.heatmap(correlation_mat, vmax=1, square=True)
        plt.savefig(self.experiment_path + '/' + self.dataset.name + '/exploratory/' + 'FeatureCorrelations.png',
                    bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close('all')

    # def univariateAnalysis(data, experiment_path, dataset_name, class_label, instance_label, match_label,
    #                        categorical_variables, jupyterRun, topFeatures):
    #     """ Calculates univariate association significance between each individual feature and class outcome. Assumes categorical outcome using Chi-square test for
    #         categorical features and Mann-Whitney Test for quantitative features. """
    #     try:  # Try loop added to deal with versions specific change to using mannwhitneyu in scipy and avoid STREAMLINE crash in those circumstances.
    #         # Create folder for univariate analysis results
    #         if not os.path.exists(experiment_path + '/' + dataset_name + '/exploratory/univariate_analyses'):
    #             os.mkdir(experiment_path + '/' + dataset_name + '/exploratory/univariate_analyses')
    #         # Generate dictionary of p-values for each feature using appropriate test (via test_selector)
    #         p_value_dict = {}
    #         for column in data:
    #             if column != class_label and column != instance_label:
    #                 p_value_dict[column] = test_selector(column, class_label, data, categorical_variables)
    #         sorted_p_list = sorted(p_value_dict.items(), key=lambda item: item[1])
    #         # Save p-values to file
    #         pval_df = pd.DataFrame.from_dict(p_value_dict, orient='index')
    #         pval_df.to_csv(
    #             experiment_path + '/' + dataset_name + '/exploratory/univariate_analyses/Univariate_Significance.csv',
    #             index_label='Feature', header=['p-value'])
    #         # Print results for top features across univariate analyses
    #         if eval(jupyterRun):
    #             fCount = data.shape[1] - 1
    #             if not instance_label == 'None':
    #                 fCount -= 1
    #             if not match_label == 'None':
    #                 fCount -= 1
    #             min_num = min(topFeatures, fCount)
    #             sorted_p_list_temp = sorted_p_list[: min_num]
    #             print('Plotting top significant ' + str(min_num) + ' features.')
    #             print('###################################################')
    #             print('Significant Univariate Associations:')
    #             for each in sorted_p_list_temp[:min_num]:
    #                 print(each[0] + ": (p-val = " + str(each[1]) + ")")
    #     except:
    #         sorted_p_list = []  # won't actually be sorted
    #         print(
    #             'WARNING: Exploratory univariate analysis failed due to scipy package version error when running mannwhitneyu test. To fix, we recommend updating scipy to version 1.8.0 or greater using: pip install --upgrade scipy')
    #         for column in data:
    #             if column != class_label and column != instance_label:
    #                 sorted_p_list.append([column, 'None'])
    #     return sorted_p_list
    #
    #
    # def univariatePlots(data, sorted_p_list, class_label, categorical_variables, experiment_path, dataset_name, sig_cutoff):
    #     """ Checks whether p-value of each feature is less than or equal to significance cutoff. If so, calls graph_selector to generate an appropriate plot."""
    #     for i in sorted_p_list:  # each feature in sorted p-value dictionary
    #         if i[1] == 'None':
    #             pass
    #         else:
    #             for j in data:  # each feature
    #                 if j == i[0] and i[1] <= sig_cutoff:  # ONLY EXPORTS SIGNIFICANT FEATURES
    #                     graph_selector(j, class_label, data, categorical_variables, experiment_path, dataset_name)
    #
    #
    # def graph_selector(featureName, class_label, data, categorical_variables, experiment_path, dataset_name):
    #     """ Assuming a categorical class outcome, a barplot is generated given a categorical feature, and a boxplot is generated given a quantitative feature. """
    #     if featureName in categorical_variables:  # Feature and Outcome are discrete/categorical/binary
    #         # Generate contingency table count bar plot. ------------------------------------------------------------------------
    #         # Calculate Contingency Table - Counts
    #         table = pd.crosstab(data[featureName], data[class_label])
    #         geom_bar_data = pd.DataFrame(table)
    #         mygraph = geom_bar_data.plot(kind='bar')
    #         plt.ylabel('Count')
    #         new_feature_name = featureName.replace(" ",
    #                                                "")  # Deal with the dataset specific characters causing problems in this dataset.
    #         new_feature_name = new_feature_name.replace("*",
    #                                                     "")  # Deal with the dataset specific characters causing problems in this dataset.
    #         new_feature_name = new_feature_name.replace("/",
    #                                                     "")  # Deal with the dataset specific characters causing problems in this dataset.
    #         plt.savefig(experiment_path + '/' + dataset_name + '/exploratory/univariate_analyses/' + 'Barplot_' + str(
    #             new_feature_name) + ".png", bbox_inches="tight", format='png')
    #         plt.close('all')
    #     else:  # Feature is continuous and Outcome is discrete/categorical/binary
    #         # Generate boxplot-----------------------------------------------------------------------------------------------------
    #         mygraph = data.boxplot(column=featureName, by=class_label)
    #         plt.ylabel(featureName)
    #         plt.title('')
    #         new_feature_name = featureName.replace(" ",
    #                                                "")  # Deal with the dataset specific characters causing problems in this dataset.
    #         new_feature_name = new_feature_name.replace("*",
    #                                                     "")  # Deal with the dataset specific characters causing problems in this dataset.
    #         new_feature_name = new_feature_name.replace("/",
    #                                                     "")  # Deal with the dataset specific characters causing problems in this dataset.
    #         plt.savefig(experiment_path + '/' + dataset_name + '/exploratory/univariate_analyses/' + 'Boxplot_' + str(
    #             new_feature_name) + ".png", bbox_inches="tight", format='png')
    #         plt.close('all')
    #

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
