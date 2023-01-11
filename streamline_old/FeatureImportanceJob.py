"""
File:FeatureImportanceJob.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 3 of STREAMLINE - This 'Job' script is called by FeatureImportanceMain.py and conducts filter-based feature importance estimation.
            It runs one of the two implemented feature importance evaluators (Mutual information or MultiSURF) on a single CV training/testing dataset pair
            for one of the original datasets in the target dataset folder (data_path).
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import sys
import random
import numpy as np
import time
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from skrebate import MultiSURF, TURF
import csv
import pickle
import os

def job(cv_train_path,experiment_path,random_state,class_label,instance_label,instance_subset,algorithm,njobs,use_TURF,TURF_pct,jupyterRun):
    """ Run all elements of the feature importance evaluation: applies either mutual information and multisurf and saves a sorted dictionary of features with associated scores """
    job_start_time = time.time() #for tracking phase runtime
    # Set random seeds for replicatability
    random.seed(random_state)
    np.random.seed(random_state)
    # Load and format data for analysis
    dataset_name,dataFeatures,dataOutcome,header,cvCount = prepareData(cv_train_path,instance_label,class_label)
    if eval(jupyterRun):
        print('Prepared Train and Test for: '+str(dataset_name)+ "_CV_"+str(cvCount))
    # Apply mutual information if specified by user
    if algorithm == 'mi':
        if eval(jupyterRun):
            print('Running Mutual Information...')
        scores,outpath,alg_name = runMutualInformation(experiment_path,dataset_name,cvCount,dataFeatures,dataOutcome,random_state)
    # Apply MultiSURF if specified by user
    elif algorithm == 'ms':
        if eval(jupyterRun):
            print('Running MultiSURF...')
        scores,outpath,alg_name = runMultiSURF(dataFeatures,dataOutcome,instance_subset,experiment_path,dataset_name,cvCount,use_TURF,TURF_pct,njobs)
    else:
        raise Exception("Feature importance algorithm not found")
    if eval(jupyterRun):
        print('Sort and pickle feature importance scores...')
    # Save sorted feature importance scores:
    scoreDict, score_sorted_features = sort_save_fi_scores(scores, header, outpath, alg_name)
    # Pickle feature importance information to be used in Phase 4 (feature selection)
    pickleScores(experiment_path,dataset_name,alg_name,scores,scoreDict,score_sorted_features,cvCount)
    # Save phase runtime
    saveRuntime(experiment_path,dataset_name,job_start_time,alg_name,cvCount)
    # Print phase completion
    print(dataset_name+" CV"+str(cvCount)+" phase 3 "+alg_name+" evaluation complete")
    job_file = open(experiment_path + '/jobsCompleted/job_'+alg_name+'_' + dataset_name + '_'+str(cvCount)+'.txt', 'w')
    job_file.write('complete')
    job_file.close()

def prepareData(cv_train_path,instance_label,class_label):
    """ Loads target cv training dataset, separates class from features and removes instance labels."""
    dataset_name = cv_train_path.split('/')[-3]
    data = pd.read_csv(cv_train_path, sep=',')
    if instance_label != 'None':
        dataFeatures = data.drop([class_label,instance_label], axis=1).values
    else:
        dataFeatures = data.drop([class_label], axis=1).values
    dataOutcome = data[class_label].values
    header = data.columns.values.tolist()
    header.remove(class_label)
    if instance_label != 'None':
        header.remove(instance_label)
    cvCount = cv_train_path.split('/')[-1].split("_")[-2]
    return dataset_name,dataFeatures,dataOutcome,header,cvCount

def runMutualInformation(experiment_path,dataset_name,cvCount,dataFeatures,dataOutcome,random_state):
    """ Run mutual information on target training dataset and return scores as well as file path/name information. """
    alg_name = "mutualinformation"
    outpath = experiment_path + '/' + dataset_name + "/feature_selection/"+alg_name+"/"+alg_name+"_scores_cv_" + str(cvCount) + '.csv'
    scores = mutual_info_classif(dataFeatures, dataOutcome, random_state=random_state)
    return scores,outpath,alg_name

def runMultiSURF(dataFeatures,dataOutcome,instance_subset,experiment_path,dataset_name,cvCount,use_TURF,TURF_pct,njobs):
    """ Run multiSURF (a Relief-based feature importance algorithm able to detect both univariate and interaction effects) and return scores as well as file path/name information """
    #Format instance sampled dataset (prevents MultiSURF from running a very long time in large instance spaces)
    formatted = np.insert(dataFeatures, dataFeatures.shape[1], dataOutcome, 1)
    choices = np.random.choice(formatted.shape[0],min(instance_subset,formatted.shape[0]),replace=False)
    newL = []
    for i in choices:
        newL.append(formatted[i])
    formatted = np.array(newL)
    dataFeatures = np.delete(formatted,-1,axis=1)
    dataPhenotypes = formatted[:,-1]
    #Run MultiSURF
    alg_name = "multisurf"
    outpath = experiment_path + '/' + dataset_name + "/feature_selection/"+alg_name+"/"+alg_name+"_scores_cv_" + str(cvCount) + '.csv'
    if eval(use_TURF):
        clf = TURF(MultiSURF(n_jobs=njobs),pct=TURF_pct).fit(dataFeatures,dataPhenotypes)
    else:
        clf = MultiSURF(n_jobs=njobs).fit(dataFeatures, dataPhenotypes)
    scores = clf.feature_importances_
    return scores,outpath,alg_name

def pickleScores(experiment_path,dataset_name,outname,scores,scoreDict,score_sorted_features,cvCount):
    """ Pickle the scores, score dicitonary and features sorted by score to be used primarily in phase 4 (feature selection) of pipeline"""
    #Save Scores to pickled file for later use
    if not os.path.exists(experiment_path + '/' + dataset_name + "/feature_selection/"+outname+"/pickledForPhase4"):
        os.mkdir(experiment_path + '/' + dataset_name + "/feature_selection/"+outname+"/pickledForPhase4")
    outfile = open(experiment_path + '/' + dataset_name + "/feature_selection/"+outname+"/pickledForPhase4/"+str(cvCount)+'.pickle','wb')
    pickle.dump([scores,scoreDict,score_sorted_features],outfile)
    outfile.close()

def saveRuntime(experiment_path,dataset_name,job_start_time,outname,cvCount):
    """ Save phase runtime """
    runtime_file = open(experiment_path + '/' + dataset_name + '/runtime/runtime_'+outname+'_CV_'+str(cvCount)+'.txt', 'w')
    runtime_file.write(str(time.time() - job_start_time))
    runtime_file.close()

def sort_save_fi_scores(scores, ordered_feature_names, filename, algo_name):
    """ Creates a feature score dictionary and a dictionary sorted by decreasing feature importance scores."""
    # Put list of scores in dictionary
    scoreDict = {}
    i = 0
    for each in ordered_feature_names:
        scoreDict[each] = scores[i]
        i += 1
    # Sort features by decreasing score
    score_sorted_features = sorted(scoreDict, key=lambda x: scoreDict[x], reverse=True)
    # Save scores to 'formatted' file
    with open(filename,mode='w', newline="") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Sorted "+algo_name+" Scores"])
        for k in score_sorted_features:
            writer.writerow([k,scoreDict[k]])
    file.close()
    return scoreDict, score_sorted_features

########################################################################################################################
if __name__ == '__main__':
    job(sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4],sys.argv[5],int(sys.argv[6]),sys.argv[7],int(sys.argv[8]),sys.argv[9],float(sys.argv[10]),sys.argv[11])
