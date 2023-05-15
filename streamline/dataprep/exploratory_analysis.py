import os
import time
import pickle
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from streamline.utils.job import Job
from streamline.utils.dataset import Dataset
from scipy.stats import chi2_contingency, mannwhitneyu
import seaborn as sns

sns.set_theme()


class EDAJob(Job):
    """
    Exploratory Data Analysis Class for the EDA/Phase 1 step of STREAMLINE
    """

    def __init__(self, dataset, experiment_path, ignore_features=None,
                 categorical_features=None, explorations=None, plots=None,
                 categorical_cutoff=10, sig_cutoff=0.05, missingness_percentage=0.5,
                 random_state=None, show_plots=False):
        """
        Initialization function for Exploratory Data Analysis Class. Parameters are defined below.

        Args:
            dataset: a streamline.utils.dataset.Dataset object or a path to dataset text file
            experiment_path: path to experiment the logging directory folder
            ignore_features: list of row names of features to ignore
            categorical_features: list of row names of categorical features
            explorations: list of names of analysis to do while doing EDA (must be in set X)
            plots: list of analysis plots to save in experiment directory (must be in set Y)
            categorical_cutoff: categorical cut off to consider a feature categorical by analysis, default=10
            sig_cutoff: significance cutoff for continuous variables, default=0.05
            random_state: random state to set seeds for reproducibility of algorithms
        """
        super().__init__()
        if type(dataset) != Dataset:
            raise (Exception("dataset input is not of type Dataset"))
        self.dataset = dataset
        self.dataset_path = dataset.path
        self.experiment_path = experiment_path
        self.random_state = random_state
        explorations_list = ["Describe", "Univariate Analysis", "Feature Correlation"]
        plot_list = ["Describe", "Univariate Analysis", "Feature Correlation"]

        for item in plot_list:
            if item not in explorations_list:
                logging.warning("Notice: Need to run analysis before plotting a result,"
                                + item + " plot will be skipped")

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

        self.engineered_features = list()
        self.categorical_cutoff = categorical_cutoff
        self.missingness_percentage = missingness_percentage
        self.sig_cutoff = sig_cutoff
        self.show_plots = show_plots

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

    def missingness_feature_engineering(self):
        """
        Feature Engineering - Missingness as a feature: (right after an initial EDA, make new missingness feature
        engineering phase)  Add new user run parameter defining the minimum missingness of a variable at which
        streamline will automatically engineer a new feature (i.e. 0 not missing vs. 1 missing) This parameter
        would have value of 0-1 and default of 0.5 meaning any feature with a missingness of >50% will have a
        corresponding missinness feature added.  This new feature would have the inserted label of
        “Miss_”+originalFeatureName. (update documentation to reflect new run parameter) - also keep a list of
        feature names for which a missingness feature was constructed. In the ‘apply’ phase, use this feature list
        to build similar new missingness features added to the replication dataset. Store the number of features
        added here, for later. I think these new binary features can be included as categorical for the rest of the
        pipeline (but make sure that one hot encoding running later only splits one feature into multiple ones,
        for features with more than two possible feature values -i.e. Keep binary features as categorical.
        """

        logging.info("Running Feature Engineering")

        # Calculating missingness
        missingness = self.dataset.data.isnull().sum() / len(self.dataset.data)

        # Finding features with missingness greater than missingness_percentage
        high_missingness_features = missingness[missingness > self.missingness_percentage]
        high_missingness_features = list(high_missingness_features.index)
        self.engineered_features = high_missingness_features

        # For each Feature with high missingness creating a categorical feature.
        for feat in high_missingness_features:
            self.dataset.data['miss_' + feat] = self.dataset.data[feat].isnull().astype(int)

    def categorical_feature_encoding(self):
        """
        Stub Function for categorical feature encoding
        """
        non_binary_categorical = list()
        for feat in self.categorical_features:
            if self.dataset.data[feat].nunique() > 2:
                non_binary_categorical.append(feat)
        if len(non_binary_categorical) > 0:
            one_hot_df = pd.get_dummies(self.dataset.data[non_binary_categorical])
            self.dataset.data.drop(non_binary_categorical, inplace=True)
            self.dataset.data = pd.concat([self.dataset.data, one_hot_df], axis=1)

    def data_manipulation(self):
        """
        Wrapper function for all feature engineering data manipulation

        """
        # Dropping rows with missing target variable.
        self.drop_ignored_rowcols()

        self.missingness_feature_engineering()

        x_data = self.dataset.feature_only_data()
        self.identify_feature_types(x_data)

        # self.categorical_feature_encoding()

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
        self.dataset.set_headers(self.experiment_path)

        self.identify_feature_types(x_data)

        return x_data

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

        # ---- INITIAL EDA ----

        # Run initial EDA from the Dataset Class giving in the current experiment folder.
        self.dataset.initial_eda(self.experiment_path)

        # ------------------ START OF DATA MANIPULATION ------------------

        x_data = self.data_manipulation()

        # ------------------ END OF DATA MANIPULATION ------------------

        logging.info("Running Basic Exploratory Analysis...")

        # Describe and save description if user specified
        if "Describe" in self.explorations:
            self.describe_data()
            total_missing = self.missingness_counts()
            plot = False
            if "Describe" in self.plots:
                plot = True
                self.missing_count_plot()
            self.counts_summary(total_missing, plot)

        # Export feature correlation plot if user specified
        if "Feature Correlation" in self.explorations:
            logging.info("Generating Feature Correlation Heatmap...")
            if "Feature Correlation" in self.plots:
                plot = True
                self.feature_correlation(x_data, plot)

        del x_data

        # Conduct uni-variate analyses of association between individual features and class
        if "Univariate Analysis" in self.explorations:
            logging.info("Running Univariate Analyses...")
            sorted_p_list = self.univariate_analysis(top_features)
            # Export uni-variate association plots (for significant features) if user specifies
            if "Univariate Analysis" in self.plots:
                logging.info("Generating Univariate Analysis Plots...")
                self.univariate_plots(sorted_p_list)
        self.save_runtime()

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
        else:
            categorical_variables = self.categorical_features + ['miss_' + feat for feat in self.engineered_features]

        categorical_features = list()
        for item in list(self.dataset.data.columns):
            if item in categorical_variables:
                categorical_features.append(item)
        self.categorical_features = categorical_features
        self.dataset.categorical_variables = self.categorical_features

        # Pickle list of feature names to be treated as categorical variables
        with open(self.experiment_path + '/' + self.dataset.name +
                  '/exploratory/categorical_variables.pickle', 'wb') as outfile:
            pickle.dump(self.categorical_features, outfile)

        with open(self.experiment_path + '/' + self.dataset.name +
                  '/exploratory/engineered_varaibles.pickle', 'wb') as outfile:
            pickle.dump(self.engineered_features, outfile)

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

    def counts_summary(self, total_missing=None, plot=False):
        """
        Reports various dataset counts: i.e. number of instances, total features, categorical features, quantitative
        features, and class counts. Also saves a simple bar graph of class counts if user specified.

        Args:
            total_missing: total missing values (optional, runs again if not given)
            plot: flag to output bar graph in the experiment log folder

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
            if self.show_plots:
                plt.show()
            else:
                plt.close('all')
                # plt.cla() # not required

    def feature_correlation(self, x_data=None, plot=True):
        """
        Calculates feature correlations via pearson correlation and exports a respective heatmap visualization.
        Due to computational expense this may not be recommended for datasets with a large number of instances
        and/or features unless needed. The generated heatmap will be difficult to read with a large number
        of features in the target dataset.

        Args:
            plot:
            x_data: data with only feature columns
        """
        if x_data is None:
            x_data = self.dataset.feature_only_data()
        # Calculate correlation matrix
        correlation_mat = x_data.corr(method='pearson')
        # corr_matrix_abs = correlation_mat.abs()

        correlation_mat.to_csv(self.experiment_path + '/' + self.dataset.name
                               + '/exploratory/' + 'FeatureCorrelations.csv')

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
                fig, ax = plt.subplots(figsize=fig_size)
                heatmap = sns.heatmap(correlation_mat, xticklabels=False, yticklabels=False, mask=mask, vmax=1, vmin=-1,
                                      square=True, cmap='RdBu')
            else:
                fig_size = (num_features // 2, num_features // 2)
                # Create a heatmap using Seaborn
                fig, ax = plt.subplots(figsize=fig_size)
                heatmap = sns.heatmap(correlation_mat, mask=mask, vmax=1, vmin=-1, square=True, cmap='RdBu')

            plt.savefig(self.experiment_path + '/' + self.dataset.name + '/exploratory/' + 'FeatureCorrelations.png',
                        bbox_inches='tight')
            if self.show_plots:
                plt.show()
            else:
                plt.close('all')
            sns.set_theme()

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

            dict_items = list(p_value_dict.items())
            sorted_p_list = sorted(dict_items, key=lambda item: float(item[1][0]))
            sorted_p_list = [(item[0], float(item[1][0])) for item in sorted_p_list]
            # Save p-values to file
            pval_df = pd.DataFrame.from_dict(p_value_dict, orient='index')
            pval_df.to_csv(
                self.experiment_path + '/' + self.dataset.name
                + '/exploratory/univariate_analyses/Univariate_Significance.csv',
                index_label='Feature', header=['p-value', 'Test-statistic', 'Test-name'], na_rep='NaN')

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
        if not os.path.exists(self.experiment_path + '/' + self.dataset.name
                              + '/exploratory/univariate_analyses/'):
            os.makedirs(self.experiment_path + '/' + self.dataset.name
                        + '/exploratory/univariate_analyses/')

        new_feature_name = feature_name.replace(" ", "")
        new_feature_name = new_feature_name.replace("*", "")
        new_feature_name = new_feature_name.replace("/", "")
        plt.savefig(self.experiment_path + '/' + self.dataset.name
                    + '/exploratory/univariate_analyses/' + 'Barplot_' +
                    str(new_feature_name) + ".png", bbox_inches="tight", format='png')
        plt.close('all')
        # plt.cla() # not required

    def test_selector(self, feature_name):
        """
        Selects and applies appropriate univariate association test for a given feature. Returns resulting p-value

        Args:
            feature_name: name of feature column operation is running on
        """
        test_name, test_stat = None, None
        class_label = self.dataset.class_label
        # Feature and Outcome are discrete/categorical/binary
        if feature_name in self.dataset.categorical_variables:
            # Calculate Contingency Table - Counts
            table_temp = pd.crosstab(self.dataset.data[feature_name], self.dataset.data[class_label])
            # Univariate association test (Chi Square Test of Independence - Non-parametric)
            c, p, dof, expected = chi2_contingency(table_temp)
            p_val = p
            test_stat = c
            test_name = "Chi Square Test"
        # Feature is continuous and Outcome is discrete/categorical/binary
        else:
            # Univariate association test (Mann-Whitney Test - Non-parametric)
            try:  # works in scipy 1.5.0
                c, p = mannwhitneyu(
                    x=self.dataset.data[feature_name].loc[self.dataset.data[class_label] == 0],
                    y=self.dataset.data[feature_name].loc[self.dataset.data[class_label] == 1], nan_policy='omit')
            except Exception as e:  # for scipy 1.8.0
                logging.error(e)
                raise Exception("Exception in scipy, must have scipy version>=1.8.0")
            p_val = p
            test_stat = c
            test_name = "Mann-Whitney U Test"
        return p_val, test_stat, test_name

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

    def start(self, top_features=20):
        self.run_explore(top_features)

    def join(self):
        pass
