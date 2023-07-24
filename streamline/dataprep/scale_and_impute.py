import os
import time
import pickle
import random
import logging
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from streamline.utils.job import Job


class ScaleAndImpute(Job):
    """
    Data Processing Job Class for Scaling and Imputation of CV Datasets
    """
    def __init__(self, cv_train_path, cv_test_path, experiment_path, scale_data=True, impute_data=True,
                 multi_impute=True, overwrite_cv=True, class_label="Class", instance_label=None, random_state=None):
        """

        Args:
            cv_train_path:
            cv_test_path:
            experiment_path:
            scale_data:
            impute_data:
            multi_impute:
            overwrite_cv:
            class_label:
            instance_label:
            random_state:
        """
        super().__init__()
        self.cv_train_path = cv_train_path
        self.cv_test_path = cv_test_path
        self.experiment_path = experiment_path
        self.scale_data = scale_data
        self.impute_data = impute_data
        self.multi_impute = multi_impute
        self.overwrite_cv = overwrite_cv
        self.class_label = class_label
        self.instance_label = instance_label
        self.categorical_variables = None
        self.dataset_name = None
        self.cv_count = None
        self.random_state = random_state

    def run(self):
        """
        Run all elements of the data preprocessing: data scaling and missing value imputation
        (mode imputation for categorical features and MICE-based iterative imputing for
        quantitative features)
        """
        # Set random seeds for repeatability
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        # Load target training and testing datasets
        data_train, data_test = self.load_data()
        # Grab header labels for features only
        header = data_train.columns.values.tolist()
        header.remove(self.class_label)
        if not (self.instance_label is None):
            header.remove(self.instance_label)
        logging.info('Preparing Train and Test for: ' + str(self.dataset_name) + "_CV_" + str(self.cv_count))
        # Temporarily separate class column to be merged back into training and testing datasets later
        y_train = data_train[self.class_label]
        y_test = data_test[self.class_label]
        # If present, temporarily separate instance label to be merged back into training and testing datasets later
        if not (self.instance_label is None):
            i_train = data_train[self.instance_label]
            i_test = data_test[self.instance_label]
        # Create features-only version of training and testing datasets for scaling and imputation
        if self.instance_label is None:
            x_train = data_train.drop([self.class_label], axis=1)  # exclude class column
            x_test = data_test.drop([self.class_label], axis=1)  # exclude class column
        else:
            x_train = data_train.drop([self.class_label, self.instance_label], axis=1)  # exclude class column
            x_test = data_test.drop([self.class_label, self.instance_label], axis=1)  # exclude class column
        del data_train  # memory cleanup
        del data_test  # memory cleanup
        # Load previously identified list of categorical variables
        # and create an index list to identify respective columns
        file = open(self.experiment_path + '/' + self.dataset_name + '/exploratory/categorical_variables.pickle', 'rb')
        self.categorical_variables = pickle.load(file)
        # Impute Missing Values in training and testing data if specified by user
        if self.impute_data:
            logging.info('Imputing Missing Values...')
            # Confirm that there are missing values in original dataset to bother with imputation
            data_counts = pd.read_csv(self.experiment_path + '/' + self.dataset_name + '/exploratory/DataCounts.csv',
                                      na_values='NA', sep=',')
            missing_values = int(data_counts['Count'].values[4])
            if missing_values != 0:
                x_train, x_test = self.impute_cv_data(x_train, x_test)
                x_train = pd.DataFrame(x_train, columns=header)
                x_test = pd.DataFrame(x_test, columns=header)
            else:  # No missing data found in dataset
                logging.info('Notice: No missing values found. Imputation skipped.')
        # Scale training and testing datasets if specified by user
        if self.scale_data:
            logging.info('Scaling Data Values...')
            x_train, x_test = self.data_scaling(x_train, x_test)
        # Remerge features with class and instance label in training and testing data
        if self.instance_label is None:
            data_train = pd.concat([
                pd.DataFrame(y_train, columns=[self.class_label]),
                pd.DataFrame(x_train, columns=header)
            ],
                axis=1, sort=False)
            data_test = pd.concat([
                pd.DataFrame(y_test, columns=[self.class_label]),
                pd.DataFrame(x_test, columns=header)
            ],
                axis=1, sort=False)
        else:
            data_train = pd.concat([
                pd.DataFrame(y_train, columns=[self.class_label]),
                pd.DataFrame(i_train, columns=[self.instance_label]),
                pd.DataFrame(x_train, columns=header)
            ],
                axis=1, sort=False)
            data_test = pd.concat([
                pd.DataFrame(y_test, columns=[self.class_label]),
                pd.DataFrame(i_test, columns=[self.instance_label]),
                pd.DataFrame(x_test, columns=header)
            ],
                axis=1, sort=False)
        del x_train  # memory cleanup
        del x_test  # memory cleanup

        # Export imputed and/or scaled cv data
        logging.info('Saving Processed Train and Test Data...')
        if self.impute_data or self.scale_data:
            self.write_cv_files(data_train, data_test)
        # Save phase runtime
        self.save_runtime()
        # Print phase completion
        logging.info(self.dataset_name + " Phase 2 complete")
        job_file = open(
            self.experiment_path + '/jobsCompleted/job_preprocessing_'
            + self.dataset_name + '_' + str(self.cv_count) + '.txt', 'w')
        job_file.write('complete')
        job_file.close()

    def load_data(self):
        """
        Load the target training and testing datasets and return respective dataframes,
        feature header labels, dataset name, and specific cv partition number for this dataset pair.
        """
        # Grab path name components
        self.dataset_name = self.cv_train_path.split('/')[-3]
        self.cv_count = self.cv_train_path.split('/')[-1].split("_")[-2]
        # Load training and testing datasets
        data_train = pd.read_csv(self.cv_train_path, na_values='NA', sep=',')
        data_test = pd.read_csv(self.cv_test_path, na_values='NA', sep=',')
        return data_train, data_test

    def impute_cv_data(self, x_train, x_test):
        """
        Begin by imputing categorical variables with simple 'mode' imputation

        Args:
            x_train: pandas dataframe with train set data
            x_test: pandas dataframe with test set data

        Returns: Imputed x_train and x_test

        """
        mode_dict = {}
        for c in x_train.columns:
            if c in self.categorical_variables:
                train_mode = x_train[c].mode().iloc[0]
                x_train[c].fillna(train_mode, inplace=True)
                mode_dict[c] = train_mode
        for c in x_test.columns:
            if c in self.categorical_variables:
                x_test[c].fillna(mode_dict[c], inplace=True)
        # Save impute map for downstream use.
        outfile = open(
            self.experiment_path + '/' + self.dataset_name
            + "/scale_impute/categorical_imputer_cv" + str(self.cv_count) + '.pickle', "wb")
        pickle.dump(mode_dict, outfile)
        outfile.close()

        if self.multi_impute:
            # Impute quantitative features (x) using iterative imputer (multiple imputation)
            imputer = IterativeImputer(random_state=self.random_state, max_iter=30)
            imputer = imputer.fit(x_train)
            x_train = imputer.transform(x_train)
            x_test = imputer.transform(x_test)
            # Save impute map for downstream use.
            outfile = open(
                self.experiment_path + '/' + self.dataset_name +
                '/scale_impute/ordinal_imputer_cv' + str(self.cv_count) + '.pickle', 'wb')
            pickle.dump(imputer, outfile)
            outfile.close()
        else:  # Impute quantitative features (x) with simple median imputation
            median_dict = {}
            for c in x_train.columns:
                if not (c in self.categorical_variables):
                    train_median = x_train[c].median()
                    x_train[c].fillna(train_median, inplace=True)
                    median_dict[c] = train_median
            for c in x_test.columns:
                if not (c in self.categorical_variables):
                    x_test[c].fillna(median_dict[c], inplace=True)
            # Save impute map for downstream use.
            outfile = open(
                self.experiment_path + '/' + self.dataset_name
                + '/scale_impute/ordinal_imputer_cv' + str(self.cv_count) + '.pickle', 'wb')
            pickle.dump(median_dict, outfile)
            outfile.close()

        return x_train, x_test

    def data_scaling(self, x_train, x_test):
        """
        Conducts data scaling using scikit-learn StandardScalar method which standardizes featuers by removing
        the mean and scaling to unit variance.

        This scaling transformation is determined (i.e. fit) based on the training dataset alone
        then the same scaling is applied (i.e. transform) to both the training and testing datasets.
        The fit scaling is pickled so that it can be applied identically to data in the future for model application.

        Args:
            x_train: pandas dataframe with train set data
            x_test: pandas dataframe with test set data

        Returns: Scaled x_train and x_test

        """
        # number of decimal places to round scaled values to
        # (Important to avoid float round errors, and thus pipeline reproducibility)
        decimal_places = 7

        # Scale features (x) using training data
        scaler = StandardScaler()
        scaler.fit(x_train)

        x_train = pd.DataFrame(scaler.transform(x_train).round(decimal_places), columns=x_train.columns)
        # Avoid float value rounding errors with large numbers of decimal places.
        # Important for pipeline reproducibility
        # x_train = x_train.round(decimal_places)
        # Scale features (x) using fit scalar in corresponding testing dataset

        x_test = pd.DataFrame(scaler.transform(x_test).round(decimal_places), columns=x_test.columns)
        # Avoid float value rounding errors with large numbers of decimal places.
        # Important for pipeline reproducibility
        # x_test = x_test.round(decimal_places)

        # Save scalar for future use
        outfile = open(self.experiment_path + '/' + self.dataset_name
                       + '/scale_impute/scaler_cv' + str(self.cv_count) + '.pickle', 'wb')
        pickle.dump(scaler, outfile)
        outfile.close()
        return x_train, x_test

    def write_cv_files(self, data_train, data_test):
        """
        Exports new training and testing datasets following imputation and/or scaling.
        Includes option to overwrite original dataset (to save space) or preserve a copy of
        training and testing dataset with CVOnly (for comparison and quality control).

        Args:
            data_train: pandas dataframe with train set data
            data_test: pandas dataframe with test set data

        Returns: None

        """
        if self.overwrite_cv:
            # Remove old CV files
            os.remove(self.cv_train_path)
            os.remove(self.cv_test_path)
        else:
            # Rename old CV files
            os.rename(self.cv_train_path,
                      self.experiment_path + '/' + self.dataset_name
                      + '/CVDatasets/' + self.dataset_name + '_CVOnly_'
                      + str(self.cv_count) + "_Train.csv")
            os.rename(self.cv_test_path,
                      self.experiment_path + '/' + self.dataset_name
                      + '/CVDatasets/' + self.dataset_name + '_CVOnly_'
                      + str(self.cv_count) + "_Test.csv")

        # Write new CV files
        data_train.to_csv(self.cv_train_path, index=False)
        data_test.to_csv(self.cv_test_path, index=False)

    def save_runtime(self):
        """ Save runtime for this phase """
        if not os.path.exists(self.experiment_path + '/' + self.dataset_name
                              + '/runtime/'):
            os.mkdir(self.experiment_path + '/' + self.dataset_name
                     + '/runtime/')
        runtime_file = open(self.experiment_path + '/' + self.dataset_name
                            + '/runtime/runtime_preprocessing'
                            + self.cv_count + '.txt', 'w+')
        runtime_file.write(str(time.time() - self.job_start_time))
        runtime_file.close()
