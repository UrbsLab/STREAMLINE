"""
File: DataPreprocessingJob.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 2 of STREAMLINE - This 'Job' script is called by DataPreprocessingMain.py and optionally conducts data scaling followed by missing value imputation.
            It runs on a single CV training/testing dataset pair for one of the original datasets in the target dataset folder (data_path).
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as scs
import random
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import csv
import time
import pickle

def job(cv_train_path,cv_test_path,experiment_path,scale_data,impute_data,overwrite_cv,categorical_cutoff,class_label,instance_label,random_state,multi_impute,jupyterRun):
    """ Run all elements of the data preprocessing: data scaling and missing value imputation (mode imputation for categorical features and MICE-based iterative imputing for quantitative features)"""
    job_start_time = time.time() #for tracking phase runtime
    #Set random seeds for replicatability
    random.seed(random_state)
    np.random.seed(random_state)
    #Load target training and testing datsets
    data_train,data_test,header,dataset_name,cvCount = loadData(cv_train_path,cv_test_path,experiment_path,class_label,instance_label)
    if eval(jupyterRun):
        print('Preparing Train and Test for: '+str(dataset_name)+ "_CV_"+str(cvCount))
    #Temporarily separate class column to be merged back into training and testing datasets later
    y_train = data_train[class_label]
    y_test = data_test[class_label]
    #If present, temporarily separate instance label to be merged back into training and testing datasets later
    if not instance_label == "None":
        i_train = data_train[instance_label]
        i_test = data_test[instance_label]
    #Create features-only version of training and testing datasets for scaling and imputation
    if instance_label == "None":
        x_train = data_train.drop([class_label],axis=1) #exclude class column
        x_test = data_test.drop([class_label],axis=1) #exclude class column
    else:
        x_train = data_train.drop([class_label,instance_label],axis=1) #exclude class column
        x_test = data_test.drop([class_label,instance_label],axis=1) #exclude class column
    del data_train #memory cleanup
    del data_test #memory cleanup
    #Load previously identified list of categorical variables and create an index list to identify respective columns
    file = open(experiment_path + '/' + dataset_name + '/exploratory/categorical_variables.pickle','rb')
    categorical_variables = pickle.load(file)
    #Impute Missing Values in training and testing data if specified by user
    if eval(impute_data):
        if eval(jupyterRun):
            print('Imputing Missing Values...')
        #Confirm that there are missing values in original dataset to bother with imputation
        dataCounts = pd.read_csv(experiment_path + '/' + dataset_name + '/exploratory/DataCounts.csv',na_values='NA',sep=',')
        missingValues = int(dataCounts['Count'].values[4])
        if missingValues != 0:
            x_train,x_test = imputeCVData(categorical_variables,x_train,x_test,random_state,experiment_path,dataset_name,cvCount,multi_impute)
            x_train = pd.DataFrame(x_train, columns=header)
            x_test = pd.DataFrame(x_test, columns=header)
        else:
            if eval(jupyterRun):
                print('Notice: No missing values found. Imputation skipped.')
    #Scale training and testing datasets if specified by user
    if eval(scale_data):
        if eval(jupyterRun):
            print('Scaling Data Values...')
        x_train,x_test = dataScaling(x_train,x_test,experiment_path,dataset_name,cvCount)
    #Remerge features with class and instance label in training and testing data
    if instance_label == None or instance_label == 'None':
        data_train = pd.concat([pd.DataFrame(y_train, columns=[class_label]), pd.DataFrame(x_train, columns=header)],axis=1, sort=False)
    else:
        data_train = pd.concat([pd.DataFrame(y_train, columns=[class_label]), pd.DataFrame(i_train, columns=[instance_label]),pd.DataFrame(x_train, columns=header)], axis=1, sort=False)
    del x_train #memory cleanup
    if instance_label == None or instance_label == 'None':
        data_test = pd.concat([pd.DataFrame(y_test, columns=[class_label]), pd.DataFrame(x_test, columns=header)],axis=1, sort=False)
    else:
        data_test = pd.concat([pd.DataFrame(y_test, columns=[class_label]), pd.DataFrame(i_test, columns=[instance_label]),pd.DataFrame(x_test, columns=header)], axis=1, sort=False)
    del x_test #memory cleanup
    #Export imputed and/or scaled cv data
    if eval(jupyterRun):
        print('Saving Processed Train and Test Data...')
    if eval(impute_data) or eval(scale_data):
        writeCVFiles(overwrite_cv,cv_train_path,cv_test_path,experiment_path,dataset_name,cvCount,data_train,data_test)
    #Save phase runtime
    saveRuntime(experiment_path,dataset_name,job_start_time)
    #Print phase completion
    print(dataset_name+" phase 2 complete")
    job_file = open(experiment_path + '/jobsCompleted/job_preprocessing_'+dataset_name+'_'+str(cvCount)+'.txt', 'w')
    job_file.write('complete')
    job_file.close()

def loadData(cv_train_path,cv_test_path,experiment_path,class_label,instance_label):
    """ Load the target training and testing datasets and return respective dataframes, feature header labels, dataset name, and specific cv partition number for this dataset pair. """
    #Grab path name components
    dataset_name = cv_train_path.split('/')[-3]
    cvCount = cv_train_path.split('/')[-1].split("_")[-2]
    #Create folder to store scaling and imputing files
    if not os.path.exists(experiment_path + '/' + dataset_name + '/scale_impute'):
        os.mkdir(experiment_path + '/' + dataset_name + '/scale_impute')
    #Load training and testing datasets
    data_train = pd.read_csv(cv_train_path,na_values='NA',sep=',')
    data_test = pd.read_csv(cv_test_path,na_values='NA',sep=',')
    #Grab header labels for features only
    header = data_train.columns.values.tolist()
    header.remove(class_label)
    if instance_label != 'None':
        header.remove(instance_label)
    return data_train,data_test,header,dataset_name,cvCount

def imputeCVData(categorical_variables,x_train,x_test,random_state,experiment_path,dataset_name,cvCount,multi_impute):
    # Begin by imputing categorical variables with simple 'mode' imputation
    mode_dict = {}
    for c in x_train.columns:
        if c in categorical_variables:
            train_mode = x_train[c].mode().iloc[0]
            x_train[c].fillna(train_mode, inplace=True)
            mode_dict[c] = train_mode
    for c in x_test.columns:
        if c in categorical_variables:
            x_test[c].fillna(mode_dict[c], inplace=True)
    #Save impute map for downstream use.
    outfile = open(experiment_path + '/' + dataset_name + '/scale_impute/categorical_imputer_cv' + str(cvCount)+'.pickle','wb')
    pickle.dump(mode_dict, outfile)
    outfile.close()

    if eval(multi_impute):
        # Impute quantitative features (x) using iterative imputer (multiple imputation)
        imputer = IterativeImputer(random_state=random_state,max_iter=30).fit(x_train)
        x_train = imputer.transform(x_train)
        x_test = imputer.transform(x_test)
        #Save impute map for downstream use.
        outfile = open(experiment_path + '/' + dataset_name + '/scale_impute/ordinal_imputer_cv' + str(cvCount)+'.pickle','wb')
        pickle.dump(imputer, outfile)
        outfile.close()
    else: #Impute quantitative features (x) with simple mean imputation
        median_dict = {}
        for c in x_train.columns:
            if not c in categorical_variables:
                train_median = x_train[c].median()
                x_train[c].fillna(train_median, inplace=True)
                median_dict[c] = train_median
        for c in x_test.columns:
            if not c in categorical_variables:
                x_test[c].fillna(median_dict[c], inplace=True)
        #Save impute map for downstream use.
        outfile = open(experiment_path + '/' + dataset_name + '/scale_impute/ordinal_imputer_cv' + str(cvCount)+'.pickle','wb')
        pickle.dump(median_dict, outfile)
        outfile.close()

    return x_train,x_test

def dataScaling(x_train,x_test,experiment_path,dataset_name,cvCount):
    """ Conducts data scaling using scikit learn's StandardScalar method which standardizes featuers by removing the mean and scaling to unit variance.
        This scaling transformation is determined (i.e. fit) based on the training dataset alone then the same scaling is applied (i.e. transform) to
        both the training and testing datasets. The fit scaling is pickled so that it can be applied identically to data in the future for model application."""
    scale_train_df = None
    scale_test_df = None
    decimal_places = 7 # number of decimal places to round scaled values to (Important to avoid float round errors, and thus pipeline reproducibility)
    # Scale features (x) using training data
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = pd.DataFrame(scaler.transform(x_train).round(decimal_places), columns=x_train.columns)
    #x_train = xtrain.round(decimal_places) # Avoid float value rounding errors with large numbers of decimal places. Important for pipeline replicability
    # Scale features (x) using fit scalar in corresponding testing dataset
    x_test = pd.DataFrame(scaler.transform(x_test).round(decimal_places), columns=x_test.columns)
    #x_test = xtest.round(decimal_places) # Avoid float value rounding errors with large numbers of decimal places. Important for pipeline replicability
    #Save scalar for future use
    outfile = open(experiment_path + '/' + dataset_name + '/scale_impute/scaler_cv'+str(cvCount)+'.pickle', 'wb')
    pickle.dump(scaler, outfile)
    outfile.close()
    return x_train, x_test

def writeCVFiles(overwrite_cv,cv_train_path,cv_test_path,experiment_path,dataset_name,cvCount,data_train,data_test):
    """ Exports new training and testing datasets following imputation and/or scaling. Includes option to overwrite original dataset (to save space) or preserve a copy of training and testing dataset with CVOnly (for comparison and quality control)."""
    if eval(overwrite_cv):
        #Remove old CV files
        os.remove(cv_train_path)
        os.remove(cv_test_path)
    else:
        #Rename old CV files
        os.rename(cv_train_path,experiment_path + '/' + dataset_name + '/CVDatasets/'+dataset_name+'_CVOnly_' + str(cvCount) +"_Train.csv")
        os.rename(cv_test_path,experiment_path + '/' + dataset_name + '/CVDatasets/'+dataset_name+'_CVOnly_' + str(cvCount) +"_Test.csv")
    #Write new CV files
    with open(cv_train_path,mode='w', newline="") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(data_train.columns.values.tolist())
        for row in data_train.values:
            writer.writerow(row)
    file.close()
    with open(cv_test_path,mode='w', newline="") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(data_test.columns.values.tolist())
        for row in data_test.values:
            writer.writerow(row)
    file.close()

def saveRuntime(experiment_path,dataset_name,job_start_time):
    """ Save runtime for this phase """
    runtime_file = open(experiment_path + '/' + dataset_name + '/runtime/runtime_preprocessing.txt','w')
    runtime_file.write(str(time.time()-job_start_time))
    runtime_file.close()

if __name__ == '__main__':
    job(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],int(sys.argv[7]),sys.argv[8],sys.argv[9],int(sys.argv[10]),sys.argv[11],sys.argv[12])
