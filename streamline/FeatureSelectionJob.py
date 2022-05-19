"""
File:FeatureSelectionJob.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 4 of STREAMLINE - This 'Job' script is called by FeatureSelectionMain.py and generates an average summary of feature importances
            across all CV datasets from phase 3 and conducts collective feature selection to remove features prior to modeling that show no association
            with class, or reduce the feature space down to some maximum number of most informative features. It is run for a single dataset from the
            original target dataset folder (data_path) in Phase 1 (i.e. feature selection completed for all cv
            training and testing datasets).
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import time
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import copy
import pandas as pd
import os
import csv
import sys

def job(full_path,do_mutual_info,do_multisurf,max_features_to_keep,filter_poor_features,top_features,export_scores,class_label,instance_label,cv_partitions,overwrite_cv,jupyterRun):
    """ Run all elements of the feature selection: reports average feature importance scores across CV sets and applies collective feature selection to generate new feature selected datasets """
    job_start_time = time.time() #for tracking phase runtime
    dataset_name = full_path.split('/')[-1]
    selected_feature_lists = {}
    meta_feature_ranks = {}
    algorithms = []
    totalFeatures = 0
    if eval(jupyterRun):
        print('Plotting Feature Importance Scores...')
    #Manage and summarize mutual information feature importance scores
    if eval(do_mutual_info):
        algorithms.append('Mutual Information')
        selected_feature_lists,meta_feature_ranks = reportAveFS("Mutual Information","mutualinformation",cv_partitions,top_features,full_path,selected_feature_lists,meta_feature_ranks,export_scores,jupyterRun)
    #Manage and summarize MultiSURF feature importance scores
    if eval(do_multisurf):
        algorithms.append('MultiSURF')
        selected_feature_lists,meta_feature_ranks = reportAveFS("MultiSURF","multisurf",cv_partitions,top_features,full_path,selected_feature_lists,meta_feature_ranks,export_scores,jupyterRun)
    # Conduct collective feature selection
    if eval(jupyterRun):
        print('Applying collective feature selection...')
    if len(algorithms) != 0:
        if eval(filter_poor_features):
            #Identify top feature subset for each cv
            cv_selected_list, informativeFeatureCounts, uninformativeFeatureCounts = selectFeatures(algorithms,cv_partitions,selected_feature_lists,max_features_to_keep,meta_feature_ranks)
            # Save count of features identified as informative for each CV partitions
            reportInformativeFeatures(informativeFeatureCounts,uninformativeFeatureCounts,full_path)
            #Generate new datasets with selected feature subsets
            genFilteredDatasets(cv_selected_list,class_label,instance_label,cv_partitions,full_path+'/CVDatasets',dataset_name,overwrite_cv)
    # Save phase runtime
    saveRuntime(full_path,job_start_time)
    # Print phase completion
    print(dataset_name + " phase 4 complete")
    experiment_path = '/'.join(full_path.split('/')[:-1])
    job_file = open(experiment_path + '/jobsCompleted/job_featureselection_' + dataset_name + '.txt', 'w')
    job_file.write('complete')
    job_file.close()

def reportInformativeFeatures(informativeFeatureCounts,uninformativeFeatureCounts,full_path):
    """ Saves counts of informative vs uninformative features (i.e. those with feature importance scores <= 0) in an csv file. """
    counts = {'Informative':informativeFeatureCounts, 'Uninformative':uninformativeFeatureCounts}
    count_df = pd.DataFrame(counts)
    count_df.to_csv(full_path+"/feature_selection/InformativeFeatureSummary.csv",index_label='CV_Partition')

def reportAveFS(algorithm,algorithmlabel,cv_partitions,top_features,full_path,selected_feature_lists,meta_feature_ranks,export_scores,jupyterRun):
    """ Loads feature importance results from phase 3, stores sorted feature importance scores for all cvs, creates a list of all feature names
    that have a feature importance score greater than 0 (i.e. some evidence that it may be informative), and creates a barplot of average
    feature importance scores. """
    #Load and manage feature importance scores ------------------------------------------------------------------
    counter = 0
    cv_keep_list = []
    feature_name_ranks = [] #stores sorded feature importance dictionaries for all CVs
    for i in range(0,cv_partitions):
        scoreInfo = full_path+"/feature_selection/"+algorithmlabel+"/pickledForPhase4/"+str(i)+'.pickle'
        file = open(scoreInfo, 'rb')
        rawData = pickle.load(file)
        file.close()
        scoreDict = rawData[1] #dictionary of feature importance scores (original feature order)
        score_sorted_features = rawData[2] #dictionary of feature importances scores (in decreasing order)
        feature_name_ranks.append(score_sorted_features)
        #Update scoreDict so it includes feature importance sums across all cvs.
        if counter == 0:
            scoreSum = copy.deepcopy(scoreDict)
        else:
            for each in rawData[1]:
                scoreSum[each] += scoreDict[each]
        counter += 1
        keep_list = []
        for each in scoreDict:
            if scoreDict[each] > 0:
                keep_list.append(each)
        cv_keep_list.append(keep_list)
    selected_feature_lists[algorithm] = cv_keep_list #stores feature names to keep for all algorithms and CVs
    meta_feature_ranks[algorithm] = feature_name_ranks #stores sorted feature importance dicitonaries for all algorithms and CVs
    #Generate barplot of average scores------------------------------------------------------------------------
    if eval(export_scores):
        # Make the sum of scores an average
        for v in scoreSum:
            scoreSum[v] = scoreSum[v] / float(cv_partitions)
        # Sort averages (decreasing order and print top 'n' and plot top 'n'
        f_names = []
        f_scores = []
        for each in scoreSum:
            f_names.append(each)
            f_scores.append(scoreSum[each])
        names_scores = {'Names': f_names, 'Scores': f_scores}
        ns = pd.DataFrame(names_scores)
        ns = ns.sort_values(by='Scores', ascending=False)
        # Select top 'n' to report and plot
        ns = ns.head(top_features)
        # Visualize sorted feature scores
        ns['Scores'].plot(kind='barh', figsize=(6, 12))
        plt.ylabel('Features')
        plt.xlabel(str(algorithm) + ' Score')
        plt.yticks(np.arange(len(ns['Names'])), ns['Names'])
        plt.title('Sorted ' + str(algorithm) + ' Scores')
        plt.savefig((full_path+"/feature_selection/"+algorithmlabel+"/TopAverageScores.png"), bbox_inches="tight")
        if eval(jupyterRun):
            plt.show()
        else:
            plt.close('all')
    return selected_feature_lists,meta_feature_ranks

def selectFeatures(algorithms, cv_partitions, selectedFeatureLists, max_features_to_keep, metaFeatureRanks): 
    """ Identifies features to keep for each cv. If more than one feature importance algorithm was applied, collective feature selection
        is applied so that the union of informative features is preserved. Overall, only informative features (i.e. those with a score > 0
        are preserved). If there are more informative features than the max_features_to_keep, then only those top scoring features are preserved.
        To reduce the feature list to some max limit, we alternate between algorithm ranked feature lists grabbing the top features from each
        until the max limit is reached."""
    cv_Selected_List = []  # final list of selected features for each cv (list of lists)
    numAlgorithms = len(algorithms)
    informativeFeatureCounts = []
    uninformativeFeatureCounts = []
    totalFeatures = len(metaFeatureRanks[algorithms[0]][0])
    if numAlgorithms > 1:  # 'Interesting' features determined by union of feature selection results (from different algorithms)
        for i in range(cv_partitions):
            unionList = selectedFeatureLists[algorithms[0]][i]  # grab first algorithm's lists of feature names to keep
            # Determine union
            for j in range(1, numAlgorithms):  # number of union comparisons
                unionList = list(set(unionList) | set(selectedFeatureLists[algorithms[j]][i]))
            informativeFeatureCounts.append(len(unionList))
            uninformativeFeatureCounts.append(totalFeatures-len(unionList))
            #Further reduce selected feature set if it is larger than max_features_to_keep
            if len(unionList) > max_features_to_keep:  # Apply further filtering if more than max features remains
                # Create score list dictionary with indexes in union list
                newFeatureList = []
                k = 0
                while len(newFeatureList) < max_features_to_keep:
                    for each in metaFeatureRanks:
                        targetFeature = metaFeatureRanks[each][i][k]
                        if not targetFeature in newFeatureList:
                            newFeatureList.append(targetFeature)
                        if len(newFeatureList) < max_features_to_keep:
                            break
                    k += 1
                unionList = newFeatureList
            unionList.sort()  # Added to ensure script random seed reproducibility
            cv_Selected_List.append(unionList)
    else:  # Only one algorithm applied (collective feature selection not applied)
        for i in range(cv_partitions):
            featureList = selectedFeatureLists[algorithms[0]][i]  # grab first algorithm's lists
            informativeFeatureCounts.append(len(featureList))
            uninformativeFeatureCounts.append(totalFeatures-informativeFeatureCounts)
            if len(featureList) > max_features_to_keep:  # Apply further filtering if more than max features remains
                # Create score list dictionary with indexes in union list
                newFeatureList = []
                k = 0
                while len(newFeatureList) < max_features_to_keep:
                    targetFeature = metaFeatureRanks[algorithms[0]][i][k]
                    newFeatureList.append(targetFeature)
                    k += 1
                featureList = newFeatureList
            cv_Selected_List.append(featureList)
    return cv_Selected_List, informativeFeatureCounts, uninformativeFeatureCounts #list of final selected features for each cv

def genFilteredDatasets(cv_selected_list,class_label,instance_label,cv_partitions,path_to_csv,dataset_name,overwrite_cv):
    """ Takes the lists of final features to be kept and creates new filtered cv training and testing datasets including only those features."""
    #create lists to hold training and testing set dataframes.
    trainList = []
    testList = []
    for i in range(cv_partitions):
        #Load training partition
        trainSet = pd.read_csv(path_to_csv+'/'+dataset_name+'_CV_' + str(i) +"_Train.csv", na_values='NA', sep = ",")
        trainList.append(trainSet)
        #Load testing partition
        testSet = pd.read_csv(path_to_csv+'/'+dataset_name+'_CV_' + str(i) +"_Test.csv", na_values='NA', sep = ",")
        testList.append(testSet)
        #Training datasets
        labelList = [class_label]
        if instance_label != 'None':
            labelList.append(instance_label)
        labelList = labelList + cv_selected_list[i]
        td_train = trainList[i][labelList]
        td_test = testList[i][labelList]
        if eval(overwrite_cv):
            #Remove old CV files
            os.remove(path_to_csv+'/'+dataset_name+'_CV_' + str(i) +"_Train.csv")
            os.remove(path_to_csv+'/'+dataset_name+'_CV_' + str(i) + "_Test.csv")
        else:
            #Rename old CV files
            os.rename(path_to_csv+'/'+dataset_name+'_CV_' + str(i) +"_Train.csv",path_to_csv+'/'+dataset_name+'_CVPre_' + str(i) +"_Train.csv")
            os.rename(path_to_csv+'/'+dataset_name+'_CV_' + str(i) + "_Test.csv",path_to_csv+'/'+dataset_name+'_CVPre_' + str(i) +"_Test.csv")
        #Write new CV files
        with open(path_to_csv+'/'+dataset_name+'_CV_' + str(i) +"_Train.csv",mode='w', newline="") as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(td_train.columns.values.tolist())
            for row in td_train.values:
                writer.writerow(row)
        file.close()
        with open(path_to_csv+'/'+dataset_name+'_CV_' + str(i) +"_Test.csv",mode='w', newline="") as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(td_test.columns.values.tolist())
            for row in td_test.values:
                writer.writerow(row)
        file.close()

def saveRuntime(full_path,job_start_time):
    """ Save phase runtime"""
    runtime_file = open(full_path + '/runtime/runtime_featureselection.txt', 'w')
    runtime_file.write(str(time.time() - job_start_time))
    runtime_file.close()

if __name__ == '__main__':
    job(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], int(sys.argv[6]),sys.argv[7], sys.argv[8],sys.argv[9],int(sys.argv[10]),sys.argv[11],sys.argv[12])
