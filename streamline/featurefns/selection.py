import os
import csv
import time
import copy
import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import median
from streamline.utils.job import Job


class FeatureSelection(Job):
    def __int__(self, full_path, n_splits, algorithms,
                class_label, instance_label):
        self.full_path = full_path
        self.algorithms = algorithms
        self.n_splits = n_splits
        self.class_label = class_label
        self.instance_label = instance_label

    def run(self, export_scores=True, top_features=20, max_features_to_keep=2000,
            filter_poor_features=True, overwrite_cv=False):
        """
        Run all elements of the feature selection: reports average feature importance scores across
        CV sets and applies collective feature selection to generate new feature selected datasets
        """
        # def job(full_path,do_mutual_info,do_multisurf,max_features_to_keep,
        #         filter_poor_features,top_features,export_scores,class_label,
        #         instance_label,cv_partitions,overwrite_cv,jupyterRun):

        self.job_start_time = time.time()
        dataset_name = self.full_path.split('/')[-1]
        selected_feature_lists = {}
        meta_feature_ranks = {}
        algorithms = []
        # total_features = 0
        logging.info('Plotting Feature Importance Scores...')
        # Manage and summarize mutual information feature importance scores
        if "Mutual Information" in algorithms:
            selected_feature_lists, meta_feature_ranks = self.report_ave_fs("Mutual Information",
                                                                            "mutualinformation", top_features,
                                                                            selected_feature_lists, meta_feature_ranks,
                                                                            export_scores)
        # Manage and summarize MultiSURF feature importance scores
        if "MultiSURF" in algorithms:
            selected_feature_lists, meta_feature_ranks = self.report_ave_fs("MultiSURF", "multisurf", top_features,
                                                                            selected_feature_lists, meta_feature_ranks,
                                                                            export_scores)
        # Conduct collective feature selection
        logging.info('Applying collective feature selection...')
        if len(algorithms) != 0:
            if eval(str(filter_poor_features)):
                # Identify top feature subset for each cv
                cv_selected_list, informative_feature_counts, uninformative_feature_counts = \
                    self.select_features(selected_feature_lists,
                                         max_features_to_keep, meta_feature_ranks)
                # Save count of features identified as informative for each CV partitions
                self.report_informative_features(informative_feature_counts, uninformative_feature_counts)
                # Generate new datasets with selected feature subsets
                self.gen_filtered_datasets(cv_selected_list, self.full_path + '/CVDatasets',
                                           dataset_name, overwrite_cv)
        # Save phase runtime
        self.save_runtime(self.full_path)
        # Print phase completion
        logging.info(dataset_name + " Phase 4 Complete")
        experiment_path = '/'.join(self.full_path.split('/')[:-1])
        job_file = open(experiment_path + '/jobsCompleted/job_featureselection_' + dataset_name + '.txt', 'w')
        job_file.write('complete')
        job_file.close()

    def report_informative_features(self, informative_feature_counts, uninformative_feature_counts):
        """
        Saves counts of informative vs uninformative features (i.e. those with feature
        importance scores <= 0) in an csv file.
        """
        counts = {'Informative': informative_feature_counts, 'Uninformative': uninformative_feature_counts}
        count_df = pd.DataFrame(counts)
        count_df.to_csv(self.full_path + "/feature_selection/InformativeFeatureSummary.csv",
                        index_label='CV_Partition')

    def report_ave_fs(self, algorithm, algorithmlabel, top_features,
                      selected_feature_lists, meta_feature_ranks, export_scores, show=False):
        """
        Loads feature importance results from phase 3, stores sorted feature importance scores for all
        cvs, creates a list of all feature names that have a feature importance score greater than 0
        (i.e. some evidence that it may be informative), and creates a barplot of average
        feature importance scores.

        Args:
            algorithm:
            algorithmlabel:
            top_features:
            selected_feature_lists:
            meta_feature_ranks:
            export_scores:
            show:

        Returns:

        """
        # Load and manage feature importance scores ------------------------------------------------------------------
        counter = 0
        cv_keep_list = []
        feature_name_ranks = []  # stores sorted feature importance dictionaries for all CVs
        cv_score_dict = {}
        for i in range(0, self.n_splits):
            score_info = self.full_path + "/feature_selection/" + algorithmlabel + "/pickledForPhase4/" + str(
                i) + '.pickle'
            file = open(score_info, 'rb')
            raw_data = pickle.load(file)
            file.close()
            score_dict = raw_data[1]  # dictionary of feature importance scores (original feature order)
            score_sorted_features = raw_data[2]  # dictionary of feature importance scores (in decreasing order)
            feature_name_ranks.append(score_sorted_features)
            # update cv_score_dict so there is a list of scores (from CV runs) for each feature
            if counter == 0:
                cv_score_dict = copy.deepcopy(score_dict)
                for each in cv_score_dict:
                    cv_score_dict[each] = [cv_score_dict[each]]
            else:
                for each in raw_data[1]:
                    cv_score_dict[each].append(score_dict[each])
            counter += 1
            """
            # Update score_dict so it includes feature importance sums across all cvs.
            if counter == 0:
                scoreSum = copy.deepcopy(scoreDict)
            else:
                for each in raw_data[1]:
                    scoreSum[each] += score_dict[each]
            """
            keep_list = []
            for each in score_dict:
                if score_dict[each] > 0:
                    keep_list.append(each)
            cv_keep_list.append(keep_list)
        selected_feature_lists[algorithm] = cv_keep_list  # stores feature names to keep for all algorithms and CVs
        # stores sorted feature importance dictionaries for all algorithms and CVs
        meta_feature_ranks[algorithm] = feature_name_ranks

        # Generate barplot of average scores------------------------------------------------------------------------
        if eval(export_scores):
            # Get median score for each features
            for v in cv_score_dict:
                cv_score_dict[v] = median(cv_score_dict[v])
            logging.info(str(cv_score_dict))
            """
            # Make the sum of scores an average
            for v in scoreSum:
                scoreSum[v] = scoreSum[v] / float(cv_partitions)
            """
            # Sort averages (decreasing order and print top 'n' and plot top 'n'
            f_names = []
            f_scores = []
            for each in cv_score_dict:
                f_names.append(each)
                f_scores.append(cv_score_dict[each])
            names_scores = {'Names': f_names, 'Scores': f_scores}
            ns = pd.DataFrame(names_scores)
            ns = ns.sort_values(by='Scores', ascending=False)
            # Select top 'n' to report and plot
            ns = ns.head(top_features)
            # Visualize sorted feature scores
            ns['Scores'].plot(kind='barh', figsize=(6, 12))
            plt.ylabel('Features')
            plt.xlabel(str(algorithm) + ' Median Score')
            plt.yticks(np.arange(len(ns['Names'])), ns['Names'])
            plt.title('Sorted Median ' + str(algorithm) + ' Scores')
            plt.savefig((self.full_path+"/feature_selection/"+algorithmlabel+"/TopAverageScores.png"),
                        bbox_inches="tight")
            if eval(str(show)):
                plt.show()
            else:
                plt.close('all')
        return selected_feature_lists, meta_feature_ranks

    def select_features(self, selected_feature_lists, max_features_to_keep, meta_feature_ranks):
        """
        Function to select features

        Identifies feature to keep for each cv.
        If more than one feature importance algorithm was applied, collective feature selection
        is applied so that the union of informative features is preserved.
        Overall, only informative features (i.e. those with a score > 0 are preserved).
        If there are more informative features than the max_features_to_keep,
        then only those top scoring features are preserved.
        To reduce the feature list to some max limit, we alternate between algorithm ranked feature
        lists grabbing the top features from each until the max limit is reached.

        Args:
            selected_feature_lists:
            max_features_to_keep:
            meta_feature_ranks:

        Returns:
            cv_selected_List, informative_feature_counts, uninformative_feature_counts
            list of final selected features for each cv

        """
        cv_selected_list = []  # final list of selected features for each cv (list of lists)
        num_algorithms = len(self.algorithms)
        informative_feature_counts = []
        uninformative_feature_counts = []
        total_features = len(meta_feature_ranks[self.algorithms[0]][0])
    #     'Interesting' features determined by union of feature selection results (from different algorithms)
        if num_algorithms > 1:
            for i in range(self.n_splits):
                # grab first algorithm's lists of feature names to keep
                # Determine union
                union_list = selected_feature_lists[self.algorithms[0]][i]
                for j in range(1, num_algorithms):  # number of union comparisons
                    union_list = list(set(union_list) | set(selected_feature_lists[self.algorithms[j]][i]))
                informative_feature_counts.append(len(union_list))
                uninformative_feature_counts.append(total_features-len(union_list))
                # Further reduce selected feature set if it is larger than max_features_to_keep
                if len(union_list) > max_features_to_keep:  # Apply further filtering if more than max features remains
                    # Create score list dictionary with indexes in union list
                    new_feature_list = []
                    k = 0
                    while len(new_feature_list) < max_features_to_keep:
                        for each in meta_feature_ranks:
                            target_feature = meta_feature_ranks[each][i][k]
                            if target_feature not in new_feature_list:
                                new_feature_list.append(target_feature)
                            if len(new_feature_list) < max_features_to_keep:
                                break
                        k += 1
                    union_list = new_feature_list
                union_list.sort()  # Added to ensure script random seed reproducibility
                cv_selected_list.append(union_list)
        else:  # Only one algorithm applied (collective feature selection not applied)
            for i in range(self.n_splits):
                feature_list = selected_feature_lists[self.algorithms[0]][i]  # grab first algorithm's lists
                informative_feature_counts.append(len(feature_list))
                uninformative_feature_counts.append(total_features-informative_feature_counts)
                # Apply further filtering if more than max features remains
                if len(feature_list) > max_features_to_keep:
                    # Create score list dictionary with indexes in union list
                    new_feature_list = []
                    k = 0
                    while len(new_feature_list) < max_features_to_keep:
                        target_feature = meta_feature_ranks[self.algorithms[0]][i][k]
                        new_feature_list.append(target_feature)
                        k += 1
                    feature_list = new_feature_list
                cv_selected_list.append(feature_list)
        return cv_selected_list, informative_feature_counts, uninformative_feature_counts

    def gen_filtered_datasets(self, cv_selected_list, path_to_csv, dataset_name, overwrite_cv):
        """
        Takes the lists of final features to be kept and creates new filtered cv training and
        testing datasets including only those features.

        Args:
            cv_selected_list:
            path_to_csv:
            dataset_name:
            overwrite_cv:

        Returns:

        """
        # create lists to hold training and testing set dataframes.
        train_list = []
        test_list = []
        for i in range(self.n_splits):
            # Load training partition
            train_set = pd.read_csv(path_to_csv + '/' + dataset_name + '_CV_' + str(i)
                                    + "_Train.csv", na_values='NA', sep=",")
            train_list.append(train_set)
            # Load testing partition
            test_set = pd.read_csv(path_to_csv+'/'+dataset_name+'_CV_' + str(i)
                                   + "_Test.csv", na_values='NA', sep=",")
            test_list.append(test_set)
            # Training datasets
            label_list = [self.class_label]
            if not(self.instance_label is None):
                label_list.append(self.instance_label)
            label_list = label_list + cv_selected_list[i]
            td_train = train_list[i][label_list]
            td_test = test_list[i][label_list]
            if eval(overwrite_cv):
                # Remove old CV files
                os.remove(path_to_csv+'/'+dataset_name+'_CV_' + str(i) + "_Train.csv")
                os.remove(path_to_csv+'/'+dataset_name+'_CV_' + str(i) + "_Test.csv")
            else:
                # Rename old CV files
                os.rename(path_to_csv+'/'+dataset_name+'_CV_' + str(i) +
                          "_Train.csv", path_to_csv+'/'+dataset_name+'_CVPre_' + str(i) + "_Train.csv")
                os.rename(path_to_csv+'/'+dataset_name+'_CV_' + str(i) +
                          "_Test.csv", path_to_csv+'/'+dataset_name+'_CVPre_' + str(i) + "_Test.csv")
            # Write new CV files
            with open(path_to_csv+'/'+dataset_name+'_CV_' + str(i) + "_Train.csv", mode='w', newline="") as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(td_train.columns.values.tolist())
                for row in td_train.values:
                    writer.writerow(row)
            file.close()
            with open(path_to_csv+'/'+dataset_name+'_CV_' + str(i) + "_Test.csv", mode='w', newline="") as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(td_test.columns.values.tolist())
                for row in td_test.values:
                    writer.writerow(row)
            file.close()

    def save_runtime(self, full_path):
        """ Save phase runtime"""
        runtime_file = open(full_path + '/runtime/runtime_featureselection.txt', 'w')
        runtime_file.write(str(time.time() - self.job_start_time))
        runtime_file.close()
