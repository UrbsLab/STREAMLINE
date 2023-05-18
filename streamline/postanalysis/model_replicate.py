import csv
import glob
import logging
import os
import pickle

import pandas as pd

from streamline.dataprep.data_process import DataProcess
from streamline.modeling.basemodel import BaseModel
from streamline.modeling.utils import ABBREVIATION, SUPPORTED_MODELS, is_supported_model
from streamline.postanalysis.statistics import StatsJob
from streamline.utils.dataset import Dataset
from streamline.utils.job import Job


# Evaluation metrics
# from scipy import interp,stats


class ReplicateJob(Job):
    """
    This 'Job' script conducts exploratory analysis on the new replication dataset then
    applies and evaluates all trained models on one or more previously unseen hold-out
    or replication study dataset(s). It also generates new evaluation figure.
    It does not deal with model feature importance estimation as this is a part of model training interpretation only.
    This script is run once for each replication dataset in rep_data_path.
    """

    def __init__(self, dataset_filename, dataset_for_rep, full_path, class_label, instance_label, match_label,
                 ignore_features=None, algorithms=None, exclude=("XCS", "eLCS"), cv_partitions=3,
                 export_feature_correlations=True, plot_roc=True, plot_prc=True, plot_metric_boxplots=True,
                 categorical_cutoff=10, sig_cutoff=0.05, featureeng_missingness=0.5, scale_data=True, impute_data=True,
                 multi_impute=True, show_plots=False, scoring_metric='balanced_accuracy', random_state=None):
        super().__init__()
        self.dataset_filename = dataset_filename
        self.dataset_for_rep = dataset_for_rep

        self.full_path = full_path
        self.class_label = class_label
        self.instance_label = instance_label
        self.match_label = match_label

        if algorithms is None:
            self.algorithms = SUPPORTED_MODELS
            if exclude is not None:
                for algorithm in exclude:
                    try:
                        self.algorithms.remove(algorithm)
                    except Exception:
                        Exception("Unknown algorithm in exclude: " + str(algorithm))
        else:
            self.algorithms = list()
            for algorithm in algorithms:
                self.algorithms.append(is_supported_model(algorithm))

        self.plot_roc = plot_roc
        self.plot_prc = plot_prc
        self.plot_metric_boxplots = plot_metric_boxplots
        self.export_feature_correlations = export_feature_correlations
        self.show_plots = show_plots
        self.cv_partitions = cv_partitions

        self.categorical_cutoff = categorical_cutoff
        self.sig_cutoff = sig_cutoff
        self.featureeng_missingness = featureeng_missingness
        self.scale_data = scale_data
        self.impute_data = impute_data
        self.scoring_metric = scoring_metric
        self.multi_impute = multi_impute
        self.ignore_features = ignore_features
        self.random_state = random_state

        self.train_name = self.full_path.split('/')[-1]
        self.experiment_path = '/'.join(self.full_path.split('/')[:-1])
        # replication dataset being analyzed in this job
        self.apply_name = self.dataset_filename.split('/')[-1].split('.')[0]

    def run(self):

        # Load Replication Dataset
        rep_data = Dataset(self.dataset_filename, self.class_label, self.match_label, self.instance_label)
        rep_feature_list = list(rep_data.data.columns.values)
        rep_feature_list.remove(self.class_label)
        if self.match_label is not None:
            rep_feature_list.remove(self.match_label)
        if self.instance_label is not None:
            rep_feature_list.remove(self.instance_label)

        # Load original training dataset (could include 'match label')
        # replication dataset file extension
        train_data = Dataset(self.dataset_for_rep, self.class_label, self.match_label, self.instance_label)
        # train_data.clean_data(ignore_features=self.ignore_features)

        all_train_feature_list = list(train_data.data.columns.values)
        all_train_feature_list.remove(self.class_label)
        if self.match_label is not None:
            all_train_feature_list.remove(self.match_label)
        if self.instance_label is not None:
            all_train_feature_list.remove(self.instance_label)

        # Confirm that all features in original training data appear in replication datasets
        if not (set(all_train_feature_list).issubset(set(rep_feature_list))):
            raise Exception('Error: One or more features in training dataset did not appear in replication dataset!')

        # Grab and order replication data columns to match training data columns
        rep_data.data = rep_data.data[train_data.data.columns]

        # Create Folder hierarchy
        if not os.path.exists(self.full_path + "/applymodel/" + self.apply_name + '/' + 'exploratory'):
            os.mkdir(self.full_path + "/applymodel/" + self.apply_name + '/' + 'exploratory')
        if not os.path.exists(self.full_path + "/applymodel/" + self.apply_name + '/' + 'model_evaluation'):
            os.mkdir(self.full_path + "/applymodel/" + self.apply_name + '/' + 'model_evaluation')
        if not os.path.exists(
                self.full_path + "/applymodel/" + self.apply_name + '/' + 'model_evaluation' + '/' + 'pickled_metrics'):
            os.mkdir(
                self.full_path + "/applymodel/" + self.apply_name + '/' + 'model_evaluation' + '/' + 'pickled_metrics')

        # Load previously identified list of categorical
        # variables and create an index list to identify respective columns
        file = open(self.full_path + '/exploratory/categorical_variables.pickle', 'rb')
        categorical_variables = pickle.load(file)

        rep_data.categorical_variables = categorical_variables

        eda = DataProcess(rep_data, self.full_path, ignore_features=self.ignore_features,
                          categorical_features=categorical_variables, explorations=[], plots=[],
                          categorical_cutoff=self.categorical_cutoff, sig_cutoff=self.sig_cutoff,
                          featureeng_missingness=self.featureeng_missingness,
                          random_state=self.random_state, show_plots=self.show_plots)

        # Arguments changed to send to correct locations describe_data(self)
        eda.dataset.name = 'applymodel/' + self.apply_name

        eda.identify_feature_types()

        transition_df = pd.DataFrame(columns=['Instances', 'Total Features',
                                              'Categorical Features',
                                              'Quantitative Features', 'Missing Values',
                                              'Missing Percent', 'Class 0', 'Class 1'])

        transition_df.loc["Original"] = eda.counts_summary(save=False)

        # ExploratoryAnalysis - basic data cleaning
        eda.drop_ignored_rowcols()

        transition_df.loc["C1"] = eda.counts_summary(save=False)

        eda.dataset.initial_eda(self.experiment_path + '/' + self.train_name)

        # Missingness Feature Reconstruction
        # Read all engineered feature names
        try:
            with open(self.experiment_path + '/' + self.train_name +
                      '/exploratory/engineered_variables.pickle', 'rb') as infile:
                eda.engineered_features = pickle.load(infile)
        except FileNotFoundError:
            eda.engineered_features = list()

        # Recreate missingness features in apply phase
        for feat in eda.engineered_features:
            eda.dataset.data['miss_' + feat] = eda.dataset.data[feat].isnull().astype(int)
        eda.engineered_features = ['miss_' + feat for feat in eda.engineered_features]

        transition_df.loc["E1"] = eda.counts_summary(save=False)

        try:
            # Removing dropped features
            with open(self.experiment_path + '/' + self.train_name +
                      '/exploratory/removed_variables.pickle', 'rb') as infile:
                removed_features = pickle.load(infile)
            eda.dataset.clean_data(removed_features)
        except FileNotFoundError:
            pass

        transition_df.loc["C2"] = eda.counts_summary(save=False)

        try:
            with open(self.experiment_path + '/' + self.train_name +
                      '/exploratory/post_processed_vars.pickle', 'rb') as infile:
                post_processed_vars = pickle.load(infile)
        except Exception as e:
            raise e

        non_binary_categorical = list()
        for feat in eda.categorical_features:
            if feat in eda.dataset.data.columns:
                if eda.dataset.data[feat].nunique() > 2:
                    non_binary_categorical.append(feat)
        # logging.warning(non_binary_categorical)
        if len(non_binary_categorical) > 0:
            one_hot_df = pd.get_dummies(eda.dataset.data[non_binary_categorical], columns=non_binary_categorical)
            eda.one_hot_features = one_hot_df.columns
            eda.dataset.data.drop(non_binary_categorical, axis=1, inplace=True)
            eda.dataset.data = pd.concat([eda.dataset.data, one_hot_df], axis=1)
        # adding features not seen in test data
        for feat in post_processed_vars:
            if feat not in list(eda.dataset.data.columns):
                eda.dataset.data[feat] = 0

        try:
            with open(self.experiment_path + '/' + self.train_name +
                      '/exploratory/correlated_features.pickle', 'rb') as infile:
                correlated_features = pickle.load(infile)
        except FileNotFoundError:
            correlated_features = list()

        # removing extra features
        for feat in eda.dataset.data.columns:
            if feat not in post_processed_vars and feat not in correlated_features:
                eda.dataset.data.drop(feat, axis=1)

        transition_df.loc["E2"] = eda.counts_summary(save=False)

        # Removing highly correlated features
        eda.dataset.clean_data(correlated_features)

        transition_df.loc["C4"] = eda.counts_summary(save=False)

        eda.dataset.data = eda.dataset.data[post_processed_vars]

        transition_df.to_csv(self.full_path + "/applymodel/" + self.apply_name + '/exploratory/'
                             + 'DataProcessSummary.csv', index=True)

        # Pickle list of feature names to be treated as categorical variables
        with open(self.full_path + "/applymodel/" + self.apply_name +
                  '/exploratory/categorical_variables.pickle', 'wb') as outfile:
            pickle.dump(eda.categorical_features, outfile)

        # Pickle list of processed feature names
        with open(self.full_path + "/applymodel/" + self.apply_name +
                  '/exploratory/post_processed_vars.pickle', 'wb') as outfile:
            pickle.dump(list(eda.dataset.data.columns), outfile)
        with open(self.full_path + "/applymodel/" + self.apply_name +
                  '/exploratory/ProcessedFeatureNames.csv', 'w') as outfile:
            writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(list(eda.dataset.data.columns))

        # Export basic exploratory analysis files
        eda.dataset.describe_data(self.experiment_path + '/' + self.train_name)

        total_missing = eda.dataset.missingness_counts(self.experiment_path + '/' + self.train_name)

        eda.counts_summary(total_missing, plot=True)

        # Create features-only version of dataset for some operations
        x_rep_data = eda.dataset.feature_only_data()

        # Export feature correlation plot if user specified
        if self.export_feature_correlations:
            eda.dataset.feature_correlation(self.experiment_path + '/' + self.train_name, x_rep_data, show_plots=False)
        del x_rep_data  # memory cleanup

        # Rep Data Preparation for each Training Partition Model set
        # (rep data will potentially be scaled, imputed and feature
        # selected in the same was as was done for each corresponding CV training partition)
        master_list = []  # Will hold all evalDict's, one for each cv dataset.

        cv_dataset_paths = list(glob.glob(self.full_path + "/CVDatasets/*_CV_*Train.csv"))
        cv_partitions = len(cv_dataset_paths)
        for cv_count in range(0, cv_partitions):
            # Get corresponding training CV dataset
            cv_train_path = self.full_path + "/CVDatasets/" + self.train_name + '_CV_' + str(cv_count) + '_Train.csv'
            cv_train_data = pd.read_csv(cv_train_path, na_values='NA', sep=",")
            # Get List of features in cv dataset
            # (if feature selection took place this may only include a subset of original training data features)
            train_feature_list = list(cv_train_data.columns.values)
            train_feature_list.remove(self.class_label)
            if self.instance_label is not None:
                if self.instance_label in train_feature_list:
                    train_feature_list.remove(self.instance_label)
            if self.match_label is not None:
                train_feature_list.remove(self.match_label)
            # Working copy of original dataframe -
            # a new version will be created for each CV partition to be applied to each corresponding set of models
            cv_rep_data = rep_data.data.copy()
            # Impute dataframe based on training imputation

            # if self.ignore_features is not None:
            #     for feature in self.ignore_features:
            #         if feature in all_train_feature_list:
            #             feature_name_list.remove(feature)
            #
            # if removed_features:
            #     for feature in removed_features:
            #         if feature in all_train_feature_list:
            #             feature_name_list.remove(feature)
            #
            # if correlated_features:
            #     for feature in correlated_features:
            #         if feature in all_train_feature_list:
            #             feature_name_list.remove(feature)
            # one_hot_list = list()
            # for var in post_processed_vars:
            #     if var not in all_train_feature_list:
            #         one_hot_list.append(var)
            #
            # feature_name_list = all_train_feature_list + engineered_features + one_hot_list

            feature_name_list = list(post_processed_vars)
            feature_name_list.remove(eda.dataset.class_label)
            if eda.dataset.instance_label:
                feature_name_list.remove(eda.dataset.instance_label)
            if eda.dataset.match_label:
                feature_name_list.remove(eda.dataset.match_label)

            if self.impute_data:
                try:
                    # assumes imputation was actually run in training (i.e. user had impute_data setting as 'True')
                    cv_rep_data = self.impute_rep_data(cv_count, cv_rep_data, feature_name_list)
                except Exception as e:
                    # If there was no missing data in respective dataset,
                    # thus no imputation files were created, bypass loading of imputation data.
                    # Requires new replication data to have no missing values, as there is no
                    # established internal scheme to conduct imputation.
                    # logging.warning(e)
                    logging.warning("Notice: Imputation was not conducted for the following target dataset, "
                                    "so imputation was not conducted for replication data: "
                                    + str(self.apply_name))
                    raise e

            # Scale dataframe based on training scaling
            if self.scale_data:
                try:
                    # assumes imputation was actually run in training (i.e. user had impute_data setting as 'True')
                    cv_rep_data = self.scale_rep_data(cv_count, cv_rep_data, feature_name_list)
                except Exception as e:
                    # If there was no missing data in respective dataset,
                    # thus no imputation files were created, bypass loading of imputation data.
                    # Requires new replication data to have no missing values, as there is no
                    # established internal scheme to conduct imputation.
                    # logging.warning(e)
                    logging.warning("Notice: Scaling was not conducted for the following target dataset, "
                                    "so scaling was not conducted for replication data: "
                                    + str(self.apply_name))
                    raise e

            # Conduct feature selection based on training selection
            # (Filters out any features not in the final cv training dataset)
            cv_rep_data = cv_rep_data[cv_train_data.columns]
            del cv_train_data  # memory cleanup
            # Prep data for evaluation
            if self.instance_label is not None:
                cv_rep_data = cv_rep_data.drop(self.instance_label, axis=1)
            x_test = cv_rep_data.drop(self.class_label, axis=1).values
            y_test = cv_rep_data[self.class_label].values
            # Unpickle algorithm info from training phases of pipeline

            eval_dict = dict()
            for algorithm in self.algorithms:
                ret = self.eval_model(algorithm, cv_count, x_test, y_test)
                eval_dict[algorithm] = ret
                pickle.dump(ret, open(self.full_path + "/applymodel/"
                                      + self.apply_name + '/model_evaluation/pickled_metrics/'
                                      + ABBREVIATION[algorithm] + '_CV_'
                                      + str(cv_count) + "_metrics.pickle", 'wb'))
                # includes everything from training except feature importance values
            master_list.append(eval_dict)  # update master list with evalDict for this CV model

        stats = StatsJob(self.full_path + '/applymodel/' + self.apply_name,
                         self.algorithms, self.class_label, self.instance_label, self.scoring_metric,
                         cv_partitions=self.cv_partitions, top_features=40, sig_cutoff=self.sig_cutoff,
                         metric_weight='balanced_accuracy', scale_data=self.scale_data,
                         plot_roc=self.plot_roc, plot_prc=self.plot_prc, plot_fi_box=False,
                         plot_metric_boxplots=self.plot_metric_boxplots, show_plots=self.show_plots)

        result_table, metric_dict = stats.primary_stats(master_list, rep_data.data)

        stats.do_plot_roc(result_table)
        stats.do_plot_prc(result_table, rep_data.data, True)

        metrics = list(metric_dict[self.algorithms[0]].keys())

        stats.save_metric_stats(metrics, metric_dict)

        if self.plot_metric_boxplots:
            stats.metric_boxplots(metrics, metric_dict)

        # Save Kruskal Wallis, Mann Whitney, and Wilcoxon Rank Sum Stats
        if len(self.algorithms) > 1:
            kruskal_summary = stats.kruskal_wallis(metrics, metric_dict)
            stats.mann_whitney_u(metrics, metric_dict, kruskal_summary)
            stats.wilcoxon_rank(metrics, metric_dict, kruskal_summary)

        # Print phase completion
        print(self.apply_name + " phase 9 complete")
        job_file = open(self.experiment_path + '/jobsCompleted/job_apply_' + self.apply_name + '.txt', 'w')
        job_file.write('complete')
        job_file.close()

    def impute_rep_data(self, cv_count, cv_rep_data, all_train_feature_list):
        # Impute categorical features (i.e. those included in the mode_dict)
        impute_cat_info = self.full_path + '/scale_impute/categorical_imputer_cv' + str(
            cv_count) + '.pickle'  # Corresponding pickle file name with scalingInfo
        infile = open(impute_cat_info, 'rb')
        mode_dict = pickle.load(infile)
        infile.close()
        for c in cv_rep_data.columns:
            if c in mode_dict:  # was the given feature identified as and treated as categorical during training?
                cv_rep_data[c].fillna(mode_dict[c], inplace=True)

        impute_rep_df = None

        impute_oridinal_info = self.full_path + '/scale_impute/ordinal_imputer_cv' + str(
            cv_count) + '.pickle'  # Corresponding pickle file name with scalingInfo
        if self.multi_impute:  # multiple imputation of quantitative features
            infile = open(impute_oridinal_info, 'rb')
            imputer = pickle.load(infile)
            infile.close()
            inst_rep = None
            # Prepare data for scikit imputation
            if self.instance_label is None or self.instance_label == 'None':
                x_rep = cv_rep_data.drop([self.class_label], axis=1).values
            else:
                x_rep = cv_rep_data.drop([self.class_label, self.instance_label], axis=1).values
                inst_rep = cv_rep_data[self.instance_label].values  # pull out instance labels in case they include text
            y_rep = cv_rep_data[self.class_label].values
            x_rep_impute = imputer.transform(x_rep)
            # Recombine x and y
            if self.instance_label is None or self.instance_label == 'None':
                impute_rep_df = pd.concat([pd.DataFrame(y_rep, columns=[self.class_label]),
                                           pd.DataFrame(x_rep_impute, columns=all_train_feature_list)], axis=1,
                                          sort=False)
            else:
                impute_rep_df = pd.concat(
                    [pd.DataFrame(y_rep, columns=[self.class_label]),
                     pd.DataFrame(inst_rep, columns=[self.instance_label]),
                     pd.DataFrame(x_rep_impute, columns=all_train_feature_list)], axis=1, sort=False)
        else:  # simple (median) imputation of quantitative features
            infile = open(impute_oridinal_info, 'rb')
            median_dict = pickle.load(infile)
            infile.close()
            for c in cv_rep_data.columns:
                if c in median_dict:  # was the given feature identified as and treated as categorical during training?
                    cv_rep_data[c].fillna(median_dict[c], inplace=True)
        return impute_rep_df

    def scale_rep_data(self, cv_count, cv_rep_data, all_train_feature_list):
        # Corresponding pickle file name with scalingInfo
        scale_info = self.full_path + '/scale_impute/scaler_cv' + str(
            cv_count) + '.pickle'
        infile = open(scale_info, 'rb')
        scaler = pickle.load(infile)
        decimal_places = 7
        infile.close()
        inst_rep = None
        # Scale target replication data
        if self.instance_label is None or self.instance_label == 'None':
            x_rep = cv_rep_data.drop([self.class_label], axis=1)
        else:
            x_rep = cv_rep_data.drop([self.class_label, self.instance_label], axis=1)
            inst_rep = cv_rep_data[self.instance_label]  # pull out instance labels in case they include text
        y_rep = cv_rep_data[self.class_label]
        # Scale features (x)
        x_rep_scaled = pd.DataFrame(scaler.transform(x_rep).round(decimal_places), columns=x_rep.columns)
        # Recombine x and y
        if self.instance_label is None or self.instance_label == 'None':
            scale_rep_df = pd.concat([pd.DataFrame(y_rep, columns=[self.class_label]),
                                      pd.DataFrame(x_rep_scaled, columns=all_train_feature_list)], axis=1, sort=False)
        else:
            scale_rep_df = pd.concat(
                [pd.DataFrame(y_rep, columns=[self.class_label]), pd.DataFrame(inst_rep, columns=[self.instance_label]),
                 pd.DataFrame(x_rep_scaled, columns=all_train_feature_list)], axis=1, sort=False)
        return scale_rep_df

    def eval_model(self, algorithm, cv_count, x_test, y_test):
        model_info = self.full_path + '/models/pickledModels/' + ABBREVIATION[algorithm] + '_' \
                     + str(cv_count) + '.pickle'
        # Corresponding pickle file name with scalingInfo
        infile = open(model_info, 'rb')
        model = pickle.load(infile)
        infile.close()
        # Prediction evaluation
        m = BaseModel(None, algorithm, scoring_metric=self.scoring_metric)
        m.model = model
        m.model_name = algorithm
        m.small_name = ABBREVIATION[algorithm]

        metric_list, fpr, tpr, roc_auc, prec, recall, \
            prec_rec_auc, ave_prec, probas_ = m.model_evaluation(x_test, y_test)

        return [metric_list, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, None, probas_]
