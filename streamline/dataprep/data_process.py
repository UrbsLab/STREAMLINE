import csv
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
from streamline.dataprep.kfold_partitioning import KFoldPartitioner
from scipy.stats import chi2_contingency, mannwhitneyu
import seaborn as sns

sns.set_theme()


class DataProcess(Job):
    """
    Exploratory Data Analysis Class for the EDA/Phase 1 step of STREAMLINE
    """

    def __init__(self, dataset, experiment_path, ignore_features=None,
                 categorical_features=None, quantitative_features=None, exclude_eda_output=None,
                 categorical_cutoff=10, sig_cutoff=0.05, featureeng_missingness=0.5,
                 cleaning_missingness=0.5, correlation_removal_threshold=1.0,
                 partition_method="Stratified", n_splits=10,
                 random_state=None, show_plots=False):
        """
        Initialization function for Exploratory Data Analysis Class. Parameters are defined below.

        Args:
            dataset: a streamline.utils.dataset.Dataset object or a path to dataset text file
            experiment_path: path to experiment the logging directory folder
            ignore_features: list of string of column names of features to ignore or \
                            path to .csv file with feature labels to be ignored in analysis (default=None)
            categorical_features: list of string of column names of features to ignore or \
                            path to .csv file with feature labels specified to be treated as categorical where possible\
                            (default=None)
            categorical_cutoff: number of unique values for a variable is considered to be quantitative vs categorical\
                            (default=10)
            exclude_eda_output: list of names of analysis to do while doing EDA (must be in set X)
            categorical_cutoff: categorical cut off to consider a feature categorical by analysis, default=10
            sig_cutoff: significance cutoff for continuous variables, default=0.05
            featureeng_missingness: the proportion of missing values within a feature (above which) a new
                            binary categorical feature is generated that indicates if the
                            value for an instance was missing or not
            cleaning_missingness: the proportion of missing values, within a feature or instance, (at which) the
                            given feature or instance will be automatically cleaned (i.e. removed)
                            from the processed ‘target dataset’
            correlation_removal_threshold: the (pearson) feature correlation at which one out of a pair of
                            features is randomly removed from the processed ‘target dataset’
            random_state: random state to set seeds for reproducibility of algorithms
        """
        super().__init__()
        if type(dataset) != Dataset:
            raise (Exception("dataset input is not of type Dataset"))
        self.dataset = dataset
        self.dataset_path = dataset.path
        self.experiment_path = experiment_path
        self.random_state = random_state

        known_exclude_options = ['describe_csv', 'univariate_plots', 'correlation_plots']

        explorations_list = ["Describe", "Univariate Analysis", "Feature Correlation"]
        plot_list = ["Describe", "Univariate Analysis", "Feature Correlation"]

        if exclude_eda_output is not None:
            for x in exclude_eda_output:
                if x not in known_exclude_options:
                    logging.warning("Unknown EDA exclusion option " + str(x))
            if 'describe_csv' in exclude_eda_output:
                explorations_list.remove("Describe")
                plot_list.remove("Describe")
            if 'univariate_plots' in exclude_eda_output:
                plot_list.remove("Univariate Analysis")
            if 'correlation_plots' in exclude_eda_output:
                plot_list.remove("Feature Correlation")

        for item in plot_list:
            if item not in explorations_list:
                logging.warning("Notice: Need to run analysis before plotting a result,"
                                + item + " plot will be skipped")

        # Set up ignore_features: Allows user to specify features that should be ignored.
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
            self.specified_categorical = None  # List of feature names specified by user to be treated as categorical
        elif type(categorical_features) == str and not categorical_features == '':
            categorical_features = pd.read_csv(categorical_features, sep=',')
            self.specified_categorical = list(categorical_features)
        elif type(categorical_features) == list:
            self.specified_categorical = list(categorical_features)
        elif categorical_features == '':
            self.specified_categorical = None
        else:
            raise Exception
        if quantitative_features is None:
            self.specified_quantitative = None  # List of feature names specified by user to be treated as quantitative
        elif type(quantitative_features) == str and not quantitative_features == '':
            quantitative_features = pd.read_csv(quantitative_features, sep=',')
            self.specified_quantitative = list(quantitative_features)
        elif type(quantitative_features) == list:
            self.specified_quantitative = list(quantitative_features)
        elif quantitative_features == '':
            self.specified_quantitative = None
        else:
            raise Exception

        self.quantitative_features = []  # List of feature names in dataset to be treated as quantitative
        self.categorical_features = []  # List of feature names in dataset to be treated as categorical

        self.engineered_features = list()
        self.one_hot_features = list()
        self.categorical_cutoff = categorical_cutoff
        self.featureeng_missingness = featureeng_missingness
        self.cleaning_missingness = cleaning_missingness
        self.correlation_removal_threshold = correlation_removal_threshold
        self.sig_cutoff = sig_cutoff
        self.show_plots = show_plots

        self.explorations = explorations_list
        self.plots = plot_list

        self.cv_partitioner = None
        self.partition_method = partition_method
        self.n_splits = n_splits

    def run(self, top_features=20):
        """
        Wrapper function to run_explore and KFoldPartitioner

        Args:
            top_features: no of top features to consider (default=20)

        """
        self.job_start_time = time.time()

        # Conduct Exploratory Analysis, Data Cleaning, and Feature Engineering
        self.run_process(top_features)

        # Conduct k-fold partitioning and generate CV datasets
        self.cv_partitioner = KFoldPartitioner(self.dataset, self.partition_method,
                                               self.experiment_path, self.n_splits, self.random_state)
        self.cv_partitioner.run()
        self.save_runtime()

    def run_process(self, top_features=20):
        """
        Run Exploratory Data Process accordingly on the EDA Object

        Args:
            top_features: no of top features to consider (default=20)
        """
        # Random seed for reproducibility
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        # Make analysis folder for target dataset and a folder for the respective exploratory analysis within it
        self.make_log_folders()

        # Account for possibility that only one dataset in folder has a match label.
        # Check for presence of match label (this allows multiple datasets to be analyzed
        # in the pipeline where not all of them have match labels if specified)
        if (self.dataset.match_label is None) or (self.dataset.match_label not in self.dataset.data.columns):
            self.dataset.match_label = None
            self.partition_method = 'Stratified'
            logging.warning("Warning: Specified 'Match label' could not be found in dataset. "
                            "Analysis moving forward assuming there is no 'match label' column using "
                            "stratified (S) CV partitioning.")

        # Pass user defined lists of categorical and quantitative features to dataset object
        # self.dataset.categorical_variables = self.categorical_features
        # self.dataset.quantitative_variables = self.quantitative_features

        # Identify and save feature types (i.e. categorical vs. quantitative)
        self.identify_feature_types()  # Completed

        # Run initial EDA from the Dataset Class
        logging.info("Running Initial EDA:")
        self.dataset.initial_eda(self.experiment_path)

        # Running all data manipulation steps: cleaning and feature engineering
        self.data_manipulation()

        # Running EDA after all data manipulation
        self.second_eda(top_features)

    def make_log_folders(self):
        """
        Makes folders for logging exploratory data analysis
        """
        if not os.path.exists(self.experiment_path + '/' + self.dataset.name):
            os.makedirs(self.experiment_path + '/' + self.dataset.name)
        if not os.path.exists(self.experiment_path + '/' + self.dataset.name + '/exploratory'):
            os.makedirs(self.experiment_path + '/' + self.dataset.name + '/exploratory')
        if not os.path.exists(self.experiment_path + '/' + self.dataset.name + '/exploratory/initial'):
            os.makedirs(self.experiment_path + '/' + self.dataset.name + '/exploratory/initial')

    def identify_feature_types(self, x_data=None):
        """
        Automatically identify categorical vs. quantitative features/variables
        Takes a dataframe (of independent variables) with column labels and
        returns a list of column names identified as
        being categorical based on user defined cutoff (categorical_cutoff).
        """
        # Validate and Identify categorical variables in dataset
        logging.info("Validating and Identifying Feature Types...")

        # Strip whitespace off user-specified feature names for consistency with dataset loading
        if self.specified_categorical is not None:
            self.specified_categorical = [s.strip() for s in self.specified_categorical]
        if self.specified_quantitative is not None:
            self.specified_quantitative = [s.strip() for s in self.specified_quantitative]
        logging.debug("spec cat: " + str(self.specified_categorical))  # Testing
        logging.debug("spec quant: " + str(self.specified_quantitative))  # Testing
        # Quality control of user-specified feature lists: duplicates check and warnings
        if self.specified_quantitative is not None and self.specified_categorical is not None:
            duplicates = list(set(self.specified_categorical) & set(self.specified_quantitative))
            if len(duplicates) > 0:
                raise Exception(
                    "Following feature(s) assigned by user as both categorical and quantitative:" + str(duplicates))
            logging.warning(
                "User specified both categorical vs quantitative features; any unspecified binary features will be "
                "treated as categorical, and any remaining features will have their feature types automatically "
                "assigned based on categorical_cutoff parameter")
        if self.specified_quantitative is None and self.specified_categorical is None:
            logging.warning(
                "User did not specify categorical vs quantitative features; feature types will be automatically "
                "assigned based on categorical_cutoff parameter")

        # Get feature data
        if x_data is None:
            x_data = self.dataset.feature_only_data()

        # Quality control of user-specified feature lists: remove specified features not in target dataset
        headers = list(x_data.columns)  # Get feature names included in target dataset
        logging.debug("data features: " + str(headers))  # TESTING
        cat_not_in_data = []
        quant_not_in_data = []
        if self.specified_categorical is not None:
            cat_not_in_data = list(set(self.specified_categorical) - set(headers))
            for feat in cat_not_in_data:
                self.specified_categorical.remove(feat)
        if self.specified_quantitative is not None:
            quant_not_in_data = list(set(self.specified_quantitative) - set(headers))
            for feat in quant_not_in_data:
                self.specified_quantitative.remove(feat)
        # Since some datasets might be very large, report this warning as a summary
        if len(cat_not_in_data) > 0:
            logging.warning(
                "Following features specified as categorical were not in target dataset: " + str(cat_not_in_data))
        if len(quant_not_in_data) > 0:
            logging.warning(
                "Following features specified as quantitative were not in target dataset: " + str(quant_not_in_data))
        logging.debug("cleaned spec cat: " + str(self.specified_categorical))  # Testing
        logging.debug("cleaned spec quant: " + str(self.specified_quantitative))  # Testing

        # Assign all binary features categorical list
        quant_to_cat = []
        unassigned_to_cat = []

        binary_categoricals_dict = dict()

        for each in x_data:
            unique_vals = list(x_data[each].unique())
            unique_vals = [x for x in unique_vals if not pd.isnull(x)]
            if len(unique_vals) == 2:
                if str(x_data[each].dtype) != 'object':
                    binary_categoricals_dict[each] = list(unique_vals)
                self.categorical_features.append(each)
                if self.specified_quantitative is not None and each in self.specified_quantitative:
                    quant_to_cat.append(each)
                    self.specified_quantitative.remove(each)  # update user specified list
                if self.specified_categorical is not None and each not in self.specified_categorical:
                    unassigned_to_cat.append(each)
                if self.specified_categorical is not None and each in self.specified_categorical:
                    self.specified_categorical.remove(each)  # update user specified list

        logging.debug("binary cat: " + str(self.categorical_features))  # TESTING

        with open(self.experiment_path + '/' + self.dataset.name +
                  '/exploratory/binary_categorical_dict.pickle', 'wb') as outfile:
            pickle.dump(binary_categoricals_dict, outfile)

        # Since some datasets might be very large, report this warning as a summary
        if len(quant_to_cat) > 0:
            logging.warning(
                "Following binary feature(s) specified as quantitative, "
                "but will be treated it as categorical: " + str(quant_to_cat))
        if len(unassigned_to_cat) > 0:
            logging.warning(
                "Following binary feature(s) were not in the categorical list, "
                "but will be treated as categorical: " + str(unassigned_to_cat))

        # Assign remaining user specified features as categorical or quantitative
        if self.specified_categorical is not None and self.specified_quantitative is None:
            logging.warning(
                "No quantitative features specified; non-binary features not specified as categorical will be treated "
                "as quantitative unless they are binary")
            self.categorical_features = self.categorical_features + self.specified_categorical
            self.quantitative_features = list(set(self.dataset.get_headers()) - set(
                self.categorical_features))  # All other features assigned as quantitative

        if self.specified_quantitative is not None and self.specified_categorical is None:
            logging.warning(
                "No categorical features specified; features not specified as quantitative will be treated as "
                "categorical")
            self.quantitative_features = self.specified_quantitative
            self.categorical_features = list(set(self.dataset.get_headers()) - set(self.quantitative_features))

        if self.specified_quantitative is not None and self.specified_categorical is not None:  # both lists specified
            self.quantitative_features = self.specified_quantitative
            self.categorical_features = self.categorical_features + self.specified_categorical
        logging.debug("assigned cat: " + str(self.categorical_features))  # TESTING
        logging.debug("assigned quant: " + str(self.quantitative_features))  # TESTING

        # Any remaining unassigned features will be assigned to categorical or quantitative lists based on user
        # specified categorical cutoff
        for each in x_data:
            if each not in self.categorical_features and each not in self.quantitative_features:
                if x_data[each].nunique() <= self.categorical_cutoff or not pd.api.types.is_numeric_dtype(x_data[each]):
                    self.categorical_features.append(each)
                else:
                    self.quantitative_features.append(each)
        logging.debug("final cat: " + str(self.categorical_features))  # TESTING
        logging.debug("final quant: " + str(self.quantitative_features))  # TESTING

        # Assign feature type lists to dataset object
        self.dataset.categorical_variables = self.categorical_features
        self.dataset.quantitative_variables = self.quantitative_features

        # Pickle feature type lists  #Ryan - where/how do these get used?
        with open(self.experiment_path + '/' + self.dataset.name +
                  '/exploratory/initial/initial_categorical_features.pickle', 'wb') as outfile:
            pickle.dump(self.categorical_features, outfile)
        with open(self.experiment_path + '/' + self.dataset.name +
                  '/exploratory/initial/initial_quantitative_features.pickle', 'wb') as outfile:
            pickle.dump(self.quantitative_features, outfile)

        with open(self.experiment_path + '/' + self.dataset.name +
                  '/exploratory/initial/initial_categorical_features.csv', 'w') as outfile:
            writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(self.categorical_features)
        with open(self.experiment_path + '/' + self.dataset.name +
                  '/exploratory/initial/initial_quantitative_features.csv', 'w') as outfile:
            writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(self.quantitative_features)

        return self.categorical_features, self.quantitative_features

    def data_manipulation(self):
        """
        Wrapper function for all data cleaning and feature engineering data manipulation
        """
        # Create features-only version of original dataset as .csv
        self.dataset.set_original_headers(self.experiment_path)  # Already Completed

        # Dataframe to record feature statistics
        transition_df = pd.DataFrame(columns=['Instances', 'Total Features',
                                              'Categorical Features',
                                              'Quantitative Features', 'Missing Values',
                                              'Missing Percent', 'Class 0', 'Class 1'])

        transition_df.loc["Original"] = self.counts_summary(save=False)

        # ordinal encode the labels
        self.label_encoder()

        # Dropping rows with missing target variable and users specified features to ignore
        self.drop_ignored_rowcols()  # Completed
        transition_df.loc["C1"] = self.counts_summary(save=False)

        # Generating categorical features for features with missingness greater that featureeng_missingness percentage
        self.feature_engineering()  # Completed
        transition_df.loc["E1"] = self.counts_summary(save=False)

        # Remove features with missingness greater than cleaning_missingness percentage
        self.drop_invariant()  # Completed
        self.feature_removal()  # Completed
        transition_df.loc["C2"] = self.counts_summary(save=False)

        # Remove instances with more features missing greater than cleaning_missingness percentage
        self.instance_removal()  # Completed
        transition_df.loc["C3"] = self.counts_summary(save=False)

        # Generated onehot categorical feature encoding
        self.categorical_feature_encoding_pandas()
        transition_df.loc["E2"] = self.counts_summary(save=False)

        # Drop highly correlated features with correlation greater that max_correlation
        self.drop_highly_correlated_features()  # Completed
        transition_df.loc["C4"] = self.counts_summary(save=False)

        # Create features-only version of processed dataset and save as .csv
        self.dataset.set_processed_headers(self.experiment_path)  # Already Completed

        # Save Transition Summary of the data manipulation process

        transition_df.to_csv(self.experiment_path + '/' + self.dataset.name + '/exploratory/'
                             + 'DataProcessSummary.csv', index=True)

        # Pickle list of feature names to be treated as categorical variables
        with open(self.experiment_path + '/' + self.dataset.name +
                  '/exploratory/categorical_features.pickle', 'wb') as outfile:
            pickle.dump(self.categorical_features, outfile)

        # Pickle list of processed feature names
        with open(self.experiment_path + '/' + self.dataset.name +
                  '/exploratory/post_processed_features.pickle', 'wb') as outfile:
            pickle.dump(list(self.dataset.data.columns), outfile)
        # with open(self.experiment_path + '/' + self.dataset.name +
        #          '/exploratory/ProcessedFeatureNames.csv', 'w') as outfile:
        #    writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #    writer.writerow(list(self.dataset.data.columns))

    def counts_summary(self, total_missing=None, plot=False, save=True, replicate=False):
        """
        Reports various dataset counts: i.e. number of instances, total features, categorical features, quantitative
        features, and class counts. Also saves a simple bar graph of class counts if user specified.

        Args:
            save:
            total_missing: total missing values (optional, runs again if not given)
            plot: flag to output bar graph in the experiment log folder
            replicate:
        Returns:

        """
        # Calculate, print, and export instance and feature counts
        f_count = self.dataset.data.shape[1] - 1
        if not (self.dataset.instance_label is None):
            f_count -= 1
        if not (self.dataset.match_label is None):
            f_count -= 1
        if total_missing is None:
            total_missing = self.dataset.missingness_counts(self.experiment_path, save=False)
        percent_missing = int(total_missing) / float(self.dataset.data.shape[0] * f_count)
        # n_categorical_variables = len(list(self.categorical_features)) \
        #                           + len(list(self.engineered_features)) + len(list(self.one_hot_features))
        summary = [['instances', self.dataset.data.shape[0]],
                   ['features', f_count],
                   ['categorical_features', len(self.categorical_features)],
                   ['quantitative_features', len(self.quantitative_features)],
                   ['missing_values', total_missing],
                   ['missing_percent', round(percent_missing, 5)]]

        summary_df = pd.DataFrame(summary, columns=['Variable', 'Count'])
        class_counts = self.dataset.data[self.dataset.class_label].value_counts()

        if save:
            summary_df.to_csv(self.experiment_path + '/' + self.dataset.name + '/exploratory/' + 'DataCounts.csv',
                              index=False)
            # Calculate, print, and export class counts
            class_counts.to_csv(self.experiment_path + '/' + self.dataset.name +
                                '/exploratory/' + 'ClassCounts.csv', header=['Count'],
                                index_label='Class')

            logging.info('Processed Data Counts: ----------------')
            logging.info('Instance Count = ' + str(self.dataset.data.shape[0]))
            logging.info('Feature Count = ' + str(f_count))
            logging.info('    Categorical  = ' + str(len(self.categorical_features)))
            logging.info('    Quantitative = ' + str(len(self.quantitative_features)))
            logging.info('Missing Count = ' + str(total_missing))
            logging.info('    Missing Percent = ' + str(percent_missing))
            logging.info('Class Counts: ----------------')
            logging.info('Class Count Information')
            df_value_counts = pd.DataFrame(class_counts)
            df_value_counts = df_value_counts.reset_index()
            df_value_counts.columns = ['Class', 'Instances']
            logging.info("\n" + df_value_counts.to_string())

            if not replicate:
                logging.info("Categorical Features: " + str(self.categorical_features))
                logging.info("\t Engineered Features: " + str(self.engineered_features))
                logging.info("\t One Hot Features: " + str(self.one_hot_features))
                logging.info("Quantitative Features: " + str(self.quantitative_features))
                logging.info("Final List of Features:")
                logging.info(list(self.dataset.get_headers()))
            else:
                logging.info("Final List of Features:")
                logging.info(list(self.dataset.get_headers()))

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
        return list(summary_df['Count']) + [class_counts[0], class_counts[1]]

    def label_encoder(self):
        """
        Numerical Data Encoder:
        for any features in the data (other than the instanceID, but including the class column) if the
        feature (which should also be considered to be categorical - so check that feature is in the list of features
        being treated as categorical, and if not add it to that list) has any non-numerical values, numerically encode
        these values based on alphabetical order of the feature values.
        As we do this we create a new output .csv file (called Numerical_Encoding_Map.csv),
        where each row provides the feature that was numerically encoded,
        and the subsequent columns provide a mapping of the original values to new numerical values.
        """

        string_type_columns = list()
        dtypes_dict = self.dataset.data.dtypes.to_dict()
        for feat, typ in dtypes_dict.items():
            if self.dataset.instance_label and feat == self.dataset.instance_label:
                continue
            if str(typ) == 'object':
                string_type_columns.append(feat)

        ord_label = pd.DataFrame(columns=['Category', 'Encoding'])
        if len(string_type_columns) > 0:
            logging.info("Ordinal encoding the following features:")
            for feat in string_type_columns:
                if feat in self.quantitative_features \
                        and not (feat == self.dataset.class_label or
                                 (self.dataset.match_label and feat == self.dataset.match_label)):
                    raise Exception("Text values specified as quantitative, any text value features that need to be "
                                    "treated as quantitative need to be numerically encoded by the user before "
                                    "running STREAMLINE")
                if feat not in self.categorical_features \
                        and not (feat == self.dataset.class_label or
                                 (self.dataset.match_label and feat == self.dataset.match_label)):
                    self.categorical_features.append(feat)
                    logging.warning("Textual Unknown Feature Added as Categorical")

                # Not encoding anything except class labels and binary text categorical variable
                # to preserve label in figures

                if feat == self.dataset.class_label:
                    logging.info('\t' + feat)
                    self.dataset.data[feat], labels = pd.factorize(self.dataset.data[feat])
                    ord_label.loc[feat] = [list(labels), list(range(len(labels)))]
                elif self.dataset.data[feat].nunique() <= 2:
                    logging.info('\t' + feat)
                    self.dataset.data[feat], labels = pd.factorize(self.dataset.data[feat])
                    ord_label.loc[feat] = [list(labels), list(range(len(labels)))]
                else:
                    # Do we fake numerical encode a dataset?
                    # labels = pd.factorize(self.dataset.data[feat])
                    # ord_label.loc[feat] = [list(labels), list(range(len(labels)))]
                    pass

            ord_label.to_csv(self.experiment_path + '/' + self.dataset.name +
                             '/exploratory/Numerical_Encoding_Map.csv')

            with open(self.experiment_path + '/' + self.dataset.name +
                      '/exploratory/ordinal_encoding.pickle', 'wb') as outfile:
                pickle.dump(ord_label, outfile)
        else:
            logging.info("No textual categorical features, skipping label encoding")

    def drop_ignored_rowcols(self, ignored_features=None):
        """
        Basic data cleaning: Drops any instances with a missing outcome
        value as well as any features (ignore_features) specified by user
        """
        # Remove features that are specified to be dropped
        if ignored_features is None:
            ignored_features = self.ignore_features
        for feat in ignored_features:
            if feat in self.categorical_features:
                self.categorical_features.remove(feat)
            if feat in self.quantitative_features:
                self.quantitative_features.remove(feat)
        self.dataset.clean_data(self.ignore_features)

    def drop_invariant(self):
        """
        Basic data cleaning: Drops any invariant features found by pandas
        """
        try:
            invariant_columns = list(self.dataset.data.columns[self.dataset.data.nunique(dropna=True) <= 1])
        except Exception:
            invariant_columns = []
        if invariant_columns:
            logging.info("Dropping the following Invariant Columns:")
            for feat in invariant_columns:
                logging.info('\t' + feat)
                if feat in self.categorical_features:
                    self.categorical_features.remove(feat)
                if feat in self.quantitative_features:
                    self.quantitative_features.remove(feat)
                if feat in self.engineered_features:
                    self.engineered_features.remove(feat)
                if feat in self.one_hot_features:
                    self.one_hot_features.remove(feat)
        self.dataset.data.drop(invariant_columns, axis=1, inplace=True)

    def feature_engineering(self):
        """
        Feature Engineering - Missingness as a feature (missingness feature engineering phase)

        Using the used run parameter we define the minimum missingness of a variable at which
        streamline will automatically engineer a new feature (i.e. 0 not missing vs. 1 missing).

        This parameter would have value of 0-1 and default of 0.5 meaning any feature with a
        missingness of >50% will have a corresponding missingness feature added.

        This new feature would have the inserted label of “Miss_”+originalFeatureName.
        The list of feature names for which a missingness feature was constructed
        is saved in self.engineered_features. In the ‘apply’ phase, we use this feature list
        to build similar new missingness features added to the replication dataset.
        """

        logging.info("Running Feature Engineering")

        # Calculating missingness for values in a feature
        missingness = self.dataset.data.isnull().sum() / len(self.dataset.data)

        # Finding features with missingness greater than featureeng_missingness
        high_missingness_features = missingness[missingness > self.featureeng_missingness]
        high_missingness_features = list(high_missingness_features.index)
        # self.high_missingness_features = high_missingness_features
        self.engineered_features = ['Miss_' + feat for feat in high_missingness_features]

        # For each Feature with high missingness creating a categorical feature.
        for feat in high_missingness_features:
            self.dataset.data['Miss_' + feat] = self.dataset.data[feat].isnull().astype(int)
            self.categorical_features.append('Miss_' + feat)

        if high_missingness_features:
            logging.info("Engineering the following Features for missingness:")
            for feat in high_missingness_features:
                logging.info('\t Miss_' + feat)

            with open(self.experiment_path + '/' + self.dataset.name +
                      '/exploratory/engineered_features.pickle', 'wb') as outfile:
                pickle.dump(high_missingness_features, outfile)

            with open(self.experiment_path + '/' + self.dataset.name +
                      '/exploratory/Missingness_Engineered_Features.csv', 'w') as outfile:
                outfile.write("\n".join(self.engineered_features))
        else:
            logging.info("No Features with high missingness found")

    def feature_removal(self):
        original_features = self.dataset.get_headers()
        self.dataset.data.dropna(thresh=int(self.dataset.data.shape[0] * self.cleaning_missingness) - 1,
                                 axis=1, inplace=True)
        new_features = self.dataset.get_headers()
        removed_variables = [item for item in original_features if item not in new_features]
        for feat in removed_variables:
            if feat in self.categorical_features:
                self.categorical_features.remove(feat)
            if feat in self.engineered_features:
                self.engineered_features.remove(feat)
            if feat in self.one_hot_features:
                self.one_hot_features.remove(feat)
            if feat in self.quantitative_features:
                self.quantitative_features.remove(feat)

        if removed_variables:
            logging.info("Removing the following Features due to Missingness:")
            for feat in removed_variables:
                logging.info('\t' + feat)
            with open(self.experiment_path + '/' + self.dataset.name +
                      '/exploratory/removed_features.pickle', 'wb') as outfile:
                pickle.dump(removed_variables, outfile)
            with open(self.experiment_path + '/' + self.dataset.name +
                      '/exploratory/Missingness_Feature_Cleaning.csv', 'w') as outfile:
                outfile.write("\n".join(removed_variables))
        else:
            logging.info("Not removing any features due to high missingness")

    def instance_removal(self):
        """
        dropping instances with feature/columns missingness greater that cleaning missingness percentage
        """
        f_count = self.dataset.data.shape[1] - 1
        if not (self.dataset.instance_label is None):
            f_count -= 1
        if not (self.dataset.match_label is None):
            f_count -= 1
        self.dataset.data = self.dataset.data[self.dataset.data.isnull().sum(axis=1) <
                                              int(self.cleaning_missingness * f_count)]

    def categorical_feature_encoding(self):
        """
        Categorical feature encoding using sklearn onehot encoder
        not used/implemented
        """
        # enc = OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False)
        # enc.fit(self.dataset.feature_only_data(), self.dataset.data[self.dataset.class_label])
        # logging.warning(enc.categories_)
        # feature_only_data = pd.DataFrame(enc.transform(self.dataset.feature_only_data()),
        #                                  columns=enc.categories_)
        # label_data = self.dataset.non_feature_data()
        # logging.warning(type(feature_only_data))
        # self.dataset.data = pd.concat([feature_only_data, label_data], axis=1)
        # with open(self.experiment_path + '/' + self.dataset.name
        #           + '/exploratory/one_hot_encoder.pickle') as file:
        #     pickle.dump(enc, file)
        raise NotImplementedError

    def categorical_feature_encoding_pandas(self):
        """
        Categorical feature encoding using pandas get_dummies function
        """
        # Identify non-binary categorical features to apply one-hot-encoding to
        non_binary_categorical = list()
        for feat in self.categorical_features:
            if feat in self.dataset.data.columns:
                if self.dataset.data[feat].nunique() > 2:
                    non_binary_categorical.append(feat)

        # Apply one-hot encoding
        if len(non_binary_categorical) > 0:
            logging.info("One-hot encoding the following features:")
            for feat in non_binary_categorical:
                logging.info('\t' + feat)
            # Run one-hot encoding
            one_hot_df = pd.get_dummies(self.dataset.data[non_binary_categorical],
                                        columns=non_binary_categorical)
            # Ryan - make it so all new features have same naming convention
            self.one_hot_features = list(one_hot_df.columns)
            # Remove original feature from dataset
            self.dataset.data.drop(non_binary_categorical, axis=1, inplace=True)
            # Add new one-hot-encoded features to the right columns of the dataset
            self.dataset.data = pd.concat([self.dataset.data, one_hot_df], axis=1)
            for feat in non_binary_categorical:
                if feat in self.categorical_features:
                    self.categorical_features.remove(feat)
            self.categorical_features += self.one_hot_features

            with open(self.experiment_path + '/' + self.dataset.name +
                      '/exploratory/one_hot_feature.pickle', 'wb') as outfile:
                pickle.dump(self.one_hot_features, outfile)
        else:
            logging.info("No non-binary categorical features, skipping categorical encoding")

    def drop_highly_correlated_features(self):
        # Ryan - if we are recalculating the correlation matrix this is
        # wasted time since it was already calculated for initial correlation plot.
        df_corr = self.dataset.feature_only_data().corr()
        df_corr_org = df_corr.copy(deep=True)

        # calculate the correlation matrix and reshape
        df_corr = df_corr.stack().reset_index()

        # rename the columns
        df_corr.columns = ['Removed_Feature', 'Correlated_Feature', 'Correlation']

        # create a mask to identify rows with duplicate features as mentioned above
        mask_dups = (df_corr[['Removed_Feature', 'Correlated_Feature']].apply(frozenset, axis=1).duplicated()) | (
                df_corr['Removed_Feature'] == df_corr['Correlated_Feature'])

        # apply the mask to clean the correlation dataframe
        df_corr = df_corr[~mask_dups]

        df_corr = df_corr.sort_values(by='Correlation', key=abs, ascending=False)

        logging.info('Top 10 Correlated Features')
        logging.info("\n" + df_corr.head(10).to_string())

        df_corr = df_corr[abs(df_corr['Correlation']) >= self.correlation_removal_threshold]

        features_to_drop = list(df_corr['Removed_Feature'])

        for feat in features_to_drop:
            if feat not in self.dataset.data.columns:
                features_to_drop.remove(feat)

        self.dataset.clean_data(features_to_drop)

        if len(features_to_drop) > 0:
            logging.info("Removing the following Features due to high correlation:")
            for feat in features_to_drop:
                logging.info(feat)
            for feat in features_to_drop:
                if feat in self.categorical_features:
                    self.categorical_features.remove(feat)
                if feat in self.engineered_features:
                    self.engineered_features.remove(feat)
                if feat in self.one_hot_features:
                    self.one_hot_features.remove(feat)
                if feat in self.quantitative_features:
                    self.quantitative_features.remove(feat)

            with open(self.experiment_path + '/' + self.dataset.name +
                      '/exploratory/correlated_features.pickle', 'wb') as outfile:
                pickle.dump(features_to_drop, outfile)

            all_features = set(self.dataset.get_headers())
            features_kept = list(all_features - set(features_to_drop))

            # logging.warning(df_corr_org.columns)

            with open(self.experiment_path + '/' + self.dataset.name +
                      '/exploratory/correlation_feature_cleaning.csv', 'w', newline='') as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Retained Feature', 'Deleted Features', ])
                for feat in features_kept:
                    corr_feat = list(df_corr_org[abs(df_corr_org[feat]) >= self.correlation_removal_threshold].index)
                    corr_feat.remove(feat)
                    if len(corr_feat) != 0:
                        writer.writerow([feat, ] + corr_feat)
        else:
            logging.info("No Features with correlation higher that parameter")

    def second_eda(self, top_features=20):
        # Running EDA after all the new data processing/manipulation
        logging.info("Running Basic Exploratory Analysis...")

        # Describe and save description if user specified
        if "Describe" in self.explorations:
            self.dataset.describe_data(self.experiment_path)
            total_missing = self.dataset.missingness_counts(self.experiment_path)
            plot = False
            if "Describe" in self.plots:
                plot = True
                self.dataset.missing_count_plot(self.experiment_path)
            self.counts_summary(total_missing, plot)

        # Export feature correlation plot if user specified
        if "Feature Correlation" in self.explorations:
            logging.info("Generating Feature Correlation Heatmap...")
            if "Feature Correlation" in self.plots:
                plot = True
                x_data = self.dataset.feature_only_data()
                self.dataset.feature_correlation(self.experiment_path, x_data, plot=plot, show_plots=self.show_plots)
        del x_data

        # Conduct uni-variate analyses of association between individual features and class
        if "Univariate Analysis" in self.explorations:
            logging.info("Running Univariate Analyses...")
            sorted_p_list = self.univariate_analysis(top_features)
            # Export uni-variate association plots (for significant features) if user specifies
            if "Univariate Analysis" in self.plots:
                logging.info("Generating Univariate Analysis Plots...")
                self.univariate_plots(sorted_p_list)

        pd.DataFrame(self.categorical_features, columns=['Feature']).to_csv(
            self.experiment_path + '/' + self.dataset.name +
            '/exploratory/processed_categorical_features.csv', index=False)
        pd.DataFrame(self.quantitative_features, columns=['Feature']).to_csv(
            self.experiment_path + '/' + self.dataset.name +
            '/exploratory/processed_quantitative_features.csv', index=False)

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

    def test_selector(self, feature_name):
        """
        Selects and applies appropriate univariate association test for a given feature. Returns resulting p-value

        Args:
            feature_name: name of feature column operation is running on
        """
        # test_name, test_stat = None, None
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
        if feature_name in self.dataset.categorical_variables:
            plt.savefig(self.experiment_path + '/' + self.dataset.name
                        + '/exploratory/univariate_analyses/' + 'Barplot_' +
                        str(new_feature_name) + ".png", bbox_inches="tight", format='png')
            plt.close('all')
        else:
            plt.savefig(self.experiment_path + '/' + self.dataset.name
                        + '/exploratory/univariate_analyses/' + 'Boxplot_' +
                        str(new_feature_name) + ".png", bbox_inches="tight", format='png')
            plt.close('all')
        # plt.cla() # not required

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

    def start(self, top_features=20):
        self.run(top_features)

    def join(self):
        pass
