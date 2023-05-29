import csv
import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


class Dataset:
    def __init__(self, dataset_path, class_label, match_label=None, instance_label=None):
        """
        Creates dataset with path of tabular file

        Args:
            dataset_path: path of tabular file (as csv, tsv, or txt)
            class_label: column label for the outcome to be predicted in the dataset
            match_label: column to identify unique groups of instances in the dataset \
            that have been 'matched' as part of preparing the dataset with cases and controls \
            that have been matched for some co-variates \
            Match label is really only used in the cross validation partitioning \
            It keeps any set of instances with the same match label value in the same partition.
            instance_label: Instance label is mostly used by the rule based learner in modeling, \
            we use it to trace back heterogeneous subgroups to the instances in the original dataset

        """
        self.data = None
        self.path = dataset_path
        self.name = self.path.split('/')[-1].split('.')[0]
        self.format = self.path.split('/')[-1].split('.')[-1]
        self.class_label = class_label
        self.match_label = match_label
        self.instance_label = instance_label
        self.categorical_variables = None
        self.load_data()

    def load_data(self):
        """
        Function to load data in dataset
        """
        logging.info("Loading Dataset: " + str(self.name))
        if self.format == 'csv':
            self.data = pd.read_csv(self.path, na_values='NA', sep=',')
        elif self.format == 'tsv':
            self.data = pd.read_csv(self.path, na_values='NA', sep='\t')
        elif self.format == 'txt':
            self.data = pd.read_csv(self.path, na_values='NA', delim_whitespace=True)
        else:
            raise Exception("Unknown file format")

        if not (self.class_label in self.data.columns):
            raise Exception("Class label not found in file")
        if self.match_label and not (self.match_label in self.data.columns):
            raise Exception("Match label not found in file")
        if self.instance_label and not (self.instance_label in self.data.columns):
            raise Exception("Instance label not found in file")

    def feature_only_data(self):
        """
        Create features-only version of dataset for some operations
        Returns: dataframe x_data with only features

        """

        if self.instance_label is None and self.match_label is None:
            x_data = self.data.drop([self.class_label], axis=1)  # exclude class column
        elif self.instance_label is not None and self.match_label is None:
            x_data = self.data.drop([self.class_label, self.instance_label], axis=1)  # exclude class column
        elif self.instance_label is None and self.match_label is not None:
            x_data = self.data.drop([self.class_label, self.match_label], axis=1)  # exclude class column
        else:
            x_data = self.data.drop([self.class_label, self.instance_label, self.match_label],
                                    axis=1)  # exclude class column
        return x_data

    def non_feature_data(self):
        """
        Create non features version of dataset for some operations
        Returns: dataframe y_data with only non features

        """
        if self.instance_label is None and self.match_label is None:
            y_data = self.data[[self.class_label]]
        elif self.instance_label is not None and self.match_label is None:
            y_data = self.data[[self.class_label, self.instance_label]]
        elif self.instance_label is None and self.match_label is not None:
            y_data = self.data[[self.class_label, self.match_label]]
        else:
            y_data = self.data[[self.class_label, self.instance_label, self.match_label]]
        return y_data

    def get_outcome(self):
        """
        Function to get outcome value form data
        Returns: outcome column

        """
        return self.data[self.class_label]

    def clean_data(self, ignore_features):
        """
        Basic data cleaning: Drops any instances with a missing outcome
        value as well as any features (ignore_features) specified by user
        """
        # Remove instances with missing outcome values
        self.data = self.data.dropna(axis=0, how='any', subset=[self.class_label])
        self.data = self.data.reset_index(drop=True)
        self.data[self.class_label] = self.data[self.class_label].astype(dtype='int8')
        # Remove columns to be ignored in analysis
        if ignore_features:
            self.data = self.data.drop(ignore_features, axis=1, errors='ignore')

    def get_headers(self):
        """
        Return feature names of the datasets

        Returns: list of feature names

        """
        headers = list(self.data.columns.values)
        headers.remove(self.class_label)
        if not (self.match_label is None):
            headers.remove(self.match_label)
        if not (self.instance_label is None):
            headers.remove(self.instance_label)
        return headers

    def set_headers(self, experiment_path, phase='exploratory'):
        """
        Exports dataset header labels for use as a reference later in the pipeline.

        Returns: list of headers labels
        """
        # Get Original Headers
        if not os.path.exists(experiment_path + '/' + self.name + '/' + phase):
            os.makedirs(experiment_path + '/' + self.name + '/' + phase)
        headers = self.data.columns.values.tolist()
        with open(experiment_path + '/' + self.name + '/' + phase + '/OriginalFeatureNames.csv', mode='w',
                  newline="") as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(headers)
        return headers

    def initial_eda(self, experiment_path, plot=False, initial='initial/'):
        self.eda(experiment_path, plot=plot, initial=initial)

    def eda(self, experiment_path, plot=False, initial=''):
        if not os.path.exists(experiment_path + '/' + self.name + '/exploratory/' + initial):
            os.makedirs(experiment_path + '/' + self.name + '/exploratory/' + initial)
        self.describe_data(experiment_path, initial=initial)
        total_missing = self.missingness_counts(experiment_path, initial=initial)
        self.missing_count_plot(experiment_path, plot=plot, initial=initial)
        self.counts_summary(experiment_path, total_missing, show_plots=plot, initial=initial)
        self.feature_correlation(experiment_path, None, show_plots=plot, initial=initial)

    def describe_data(self, experiment_path, initial=''):
        """
        Conduct and export basic dataset descriptions including basic column statistics, column variable types
        (i.e. int64 vs. float64), and unique value counts for each column
        """
        self.data.describe().to_csv(experiment_path + '/' + self.name +
                                    '/exploratory/' + initial + 'DescribeDataset.csv')
        self.data.dtypes.to_csv(experiment_path + '/' + self.name +
                                '/exploratory/' + initial + 'DtypesDataset.csv',
                                header=['DataType'], index_label='Variable')
        self.data.nunique().to_csv(experiment_path + '/' + self.name +
                                   '/exploratory/' + initial + 'NumUniqueDataset.csv',
                                   header=['Count'], index_label='Variable')

    def missingness_counts(self, experiment_path, initial='', save=True):
        """
        Count and export missing values for all data columns.
        """
        # Assess Missingness in all data columns
        missing_count = self.data.isnull().sum()
        total_missing = self.data.isnull().sum().sum()
        if save:
            missing_count.to_csv(experiment_path + '/' + self.name + '/exploratory/' + initial + 'DataMissingness.csv',
                                 header=['Count'], index_label='Variable')
        return total_missing

    def missing_count_plot(self, experiment_path, plot=False, initial=''):
        """
        Plots a histogram of missingness across all data columns.
        """
        missing_count = self.data.isnull().sum()
        # Plot a histogram of the missingness observed over all columns in the dataset
        plt.hist(missing_count, bins=100)
        plt.xlabel("Missing Value Counts")
        plt.ylabel("Frequency")
        plt.title("Histogram of Missing Value Counts in Dataset")
        plt.savefig(experiment_path + '/' + self.name + '/exploratory/' + initial + 'DataMissingnessHistogram.png',
                    bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close('all')

    def counts_summary(self, experiment_path, total_missing=None, plot=True, show_plots=False, initial=''):
        """
        Reports various dataset counts: i.e. number of instances, total features, categorical features, quantitative
        features, and class counts. Also saves a simple bar graph of class counts if user specified.

        Args:
            experiment_path:
            total_missing: total missing values (optional, runs again if not given)
            plot: flag to output bar graph in the experiment log folder
            show_plots: flag to show plots
            initial: flag for initial eda

        Returns:

        """
        # Calculate, print, and export instance and feature counts
        f_count = self.data.shape[1] - 1
        if not (self.instance_label is None):
            f_count -= 1
        if not (self.match_label is None):
            f_count -= 1
        if total_missing is None:
            total_missing = self.missingness_counts(experiment_path)
        percent_missing = int(total_missing) / float(self.data.shape[0] * f_count)
        n_categorical_variables = len(self.categorical_variables)
        summary = [['instances', self.data.shape[0]],
                   ['features', f_count],
                   ['categorical_features', n_categorical_variables],
                   ['quantitative_features', f_count - n_categorical_variables],
                   ['missing_values', total_missing],
                   ['missing_percent', round(percent_missing, 5)]]

        summary_df = pd.DataFrame(summary, columns=['Variable', 'Count'])

        summary_df.to_csv(experiment_path + '/' + self.name + '/exploratory/' + initial + 'DataCounts.csv',
                          index=False)
        # Calculate, print, and export class counts
        class_counts = self.data[self.class_label].value_counts()
        class_counts.to_csv(experiment_path + '/' + self.name +
                            '/exploratory/' + initial + 'ClassCounts.csv', header=['Count'],
                            index_label='Class')

        logging.info('Initial Data Counts: ----------------')
        logging.info('Instance Count = ' + str(self.data.shape[0]))
        logging.info('Feature Count = ' + str(f_count))
        logging.info('    Categorical  = ' + str(n_categorical_variables))
        logging.info('    Quantitative = ' + str(f_count - n_categorical_variables))
        logging.info('Missing Count = ' + str(total_missing))
        logging.info('    Missing Percent = ' + str(percent_missing))
        logging.info('Class Counts: ----------------')
        logging.info('Class Count Information')
        df_value_counts = pd.DataFrame(class_counts)
        df_value_counts = df_value_counts.reset_index()
        df_value_counts.columns = ['Class', 'Instances']
        logging.info("\n" + df_value_counts.to_string())

        # Generate and export class count bar graph
        if plot:
            class_counts.plot(kind='bar')
            plt.ylabel('Count')
            plt.title('Class Counts')
            plt.savefig(experiment_path + '/' + self.name + '/exploratory/' + initial + 'ClassCountsBarPlot.png',
                        bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close('all')

    def feature_correlation(self, experiment_path, x_data=None, plot=True, show_plots=False, initial=''):
        """
        Calculates feature correlations via pearson correlation and exports a respective heatmap visualization.
        Due to computational expense this may not be recommended for datasets with a large number of instances
        and/or features unless needed. The generated heatmap will be difficult to read with a large number
        of features in the target dataset.

        Args:

            experiment_path:
            x_data: data with only feature columns
            plot:
            show_plots:
            initial:
        """
        if x_data is None:
            x_data = self.feature_only_data()
        # Calculate correlation matrix
        correlation_mat = x_data.corr(method='pearson', numeric_only=True)
        # corr_matrix_abs = correlation_mat.abs()

        correlation_mat.to_csv(experiment_path + '/' + self.name
                               + '/exploratory/' + initial + 'FeatureCorrelations.csv')

        if plot:
            # Create a mask for the upper triangle of the correlation matrix
            mask = np.zeros_like(correlation_mat, dtype=bool)
            mask[np.triu_indices_from(mask)] = True

            # Calculate the number of features in the dataset
            num_features = len(x_data.columns)

            sns.set_style("white")
            # Set the fig-size parameter based on the number of features
            if num_features > 70:  #
                fig_size = (70 // 2, 70 // 2)
                # Create a heatmap using Seaborn
                plt.subplots(figsize=fig_size)
                sns.heatmap(correlation_mat, xticklabels=False, yticklabels=False, mask=mask, vmax=1, vmin=-1,
                            square=True, cmap='RdBu', cbar_kws={"shrink": .75})
            else:
                fig_size = (num_features // 2, num_features // 2)
                # Create a heatmap using Seaborn
                plt.subplots(figsize=fig_size)
                sns.heatmap(correlation_mat, mask=mask, vmax=1, vmin=-1, square=True, cmap='RdBu',
                            cbar_kws={"shrink": .75})

            plt.savefig(experiment_path + '/' + self.name + '/exploratory/' + initial + 'FeatureCorrelations.png',
                        bbox_inches='tight')
            if show_plots:
                plt.show()
                plt.close('all')
            else:
                plt.close('all')
            plt.close('all')
            sns.set_theme()
