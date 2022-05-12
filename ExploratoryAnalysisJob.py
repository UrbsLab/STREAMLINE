"""
File: ExploratoryAnalysisJob.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 1 of STREAMLINE - This 'Job' script is called by ExploratoryAnalysisMain.py and conducts a basic exploratory analysis and cross validation partitioning
             for a single dataset within the target dataset folder. This code as been arranged so that it could also be called and run within the included jupyer notebook.
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

def job(dataset_path,experiment_path,cv_partitions,partition_method,categorical_cutoff,export_feature_correlations,export_univariate_plots,class_label,instance_label,match_label,random_state,ignore_features_path,categorical_feature_path,sig_cutoff,jupyterRun):
    """ Prepares ignore_features and categorical_feature_headers lists then calls the exploratory analyisis method."""
    #Allows user to specify features in target datasets that should be excluded during pipeline analysis
    if ignore_features_path == 'None':
        ignore_features = []
    else:
        ignore_features = pd.read_csv(ignore_features_path,sep=',')
        ignore_features = list(ignore_features)
    #Allows user to specify features that should be treated as categorical whenever possible, rather than relying on pipelines automated strategy for distinguishing categorical vs. quantitative features using the categorical_cutoff parameter.
    if categorical_feature_path == 'None':
        categorical_feature_headers = []
    else:
        categorical_feature_headers = pd.read_csv(categorical_feature_path,sep=',')
        categorical_feature_headers = list(categorical_feature_headers)
    runExplore(dataset_path,experiment_path,cv_partitions,partition_method,categorical_cutoff,export_feature_correlations,export_univariate_plots,class_label,instance_label,match_label,random_state,ignore_features,categorical_feature_headers,sig_cutoff,jupyterRun)

def runExplore(dataset_path,experiment_path,cv_partitions,partition_method,categorical_cutoff,export_feature_correlations,export_univariate_plots,class_label,instance_label,match_label,random_state,ignore_features,categorical_feature_headers,sig_cutoff,jupyterRun):
    """ Run all elements of the exploratory analysis: basic data cleaning, automated identification of categorical vs. quantitative features, basic data summary (e.g. sample size, feature type counts, class counts)"""
    job_start_time = time.time() #for tracking phase runtime
    topFeatures = 20 # only used with jupyter notebook reporting of univariate analyses.
    #Set random seeds for replicatability
    random.seed(random_state)
    np.random.seed(random_state)
    #Make analysis folder for target dataset and a folder for the respective exploratory analysis within it
    dataset_name,dataset_ext = makeFolders(dataset_path,experiment_path)
    #Load target dataset
    if eval(jupyterRun):
        print("Loading Dataset: "+str(dataset_name))
    data = loadData(dataset_path,dataset_ext)
    #Basic data cleaning
    if eval(jupyterRun):
        print("Cleaning Dataset...")
    data = removeRowsColumns(data,class_label,ignore_features)
    #Account for possibility that only one dataset in folder has a match label. Check for presence of match label (this allows multiple datasets to be analyzed in the pipeline where not all of them have match labels if specified)
    if not match_label == 'None':
        dfHeader = list(data.columns.values)
        if not match_label in dfHeader:
            match_label = 'None'
            partition_method = 'S'
            print("Warning: Specified 'Match label' could not be found in dataset. Analysis moving forward assuming there is no 'match label' column using stratified (S) CV partitioning.")
    #Create features-only version of dataset for some operations
    if instance_label == "None" and match_label == "None":
        x_data = data.drop([class_label],axis=1) #exclude class column
    elif not instance_label == "None" and match_label == "None":
        x_data = data.drop([class_label,instance_label],axis=1) #exclude class column
    elif instance_label == "None" and not match_label == "None":
        x_data = data.drop([class_label,match_label],axis=1) #exclude class column
    else:
        x_data = data.drop([class_label,instance_label,match_label],axis=1) #exclude class column
    #Automatically identify categorical vs. quantitative features/variables
    if eval(jupyterRun):
        print("Identifying Feature Types...")
    categorical_variables = idFeatureTypes(x_data,categorical_feature_headers,categorical_cutoff,experiment_path,dataset_name)
    #Export basic exploratory analysis files
    if eval(jupyterRun):
        print("Running Basic Exploratory Analysis...")
    describeData(data,experiment_path,dataset_name)
    totalMissing = missingnessCounts(data,experiment_path,dataset_name,jupyterRun)
    countsSummary(data,class_label,experiment_path,dataset_name,instance_label,match_label,categorical_variables,totalMissing,jupyterRun)

    #Export feature correlation plot if user specified
    if eval(export_feature_correlations):
        if eval(jupyterRun):
            print("Generating Feature Correlation Heatmap...")
        featureCorrelationPlot(x_data,experiment_path,dataset_name,jupyterRun)
    #Export feature labels from data header as a reference to be used later in the pipeline
    reportHeaders(x_data,experiment_path,dataset_name)
    del x_data #memory cleanup
    #Conduct univariate analyses of association between individual features and class
    if eval(jupyterRun):
        print("Running Univariate Analyses...")
    sorted_p_list = univariateAnalysis(data,experiment_path,dataset_name,class_label,instance_label,match_label,categorical_variables,jupyterRun,topFeatures)
    #Export univariate association plots (for significant features) if user specifies
    if eval(export_univariate_plots):
        if eval(jupyterRun):
            print("Generating Univariate Analysis Plots...")
        univariatePlots(data,sorted_p_list,class_label,categorical_variables,experiment_path,dataset_name,sig_cutoff)
    #Generate and export cross validation datasets (i.e. training and testing sets)
    if eval(jupyterRun):
        print("Generating and Saving CV Datasets...")
    train_dfs,test_dfs = cv_partitioner(data,cv_partitions,partition_method,class_label,match_label,random_state)
    saveCVDatasets(experiment_path,dataset_name,train_dfs,test_dfs)
    #Save phase runtime
    saveRuntime(experiment_path,dataset_name,job_start_time)
    #Print phase completion
    print(dataset_name+" phase 1 complete")
    job_file = open(experiment_path + '/jobsCompleted/job_exploratory_'+dataset_name+'.txt', 'w')
    job_file.write('complete')
    job_file.close()

def makeFolders(dataset_path,experiment_path):
    """ Make analysis folder for target dataset and a folder for the respective exploratory analysis within it"""
    dataset_name = dataset_path.split('/')[-1].split('.')[0]
    dataset_ext = dataset_path.split('/')[-1].split('.')[-1]
    if not os.path.exists(experiment_path + '/' + dataset_name):
        os.mkdir(experiment_path + '/' + dataset_name)
    if not os.path.exists(experiment_path + '/' + dataset_name + '/exploratory'):
        os.mkdir(experiment_path + '/' + dataset_name + '/exploratory')
    return dataset_name,dataset_ext

def loadData(dataset_path,dataset_ext):
    """ Load the target dataset given the dataset file path and respective file extension"""
    if dataset_ext == 'csv':
        data = pd.read_csv(dataset_path,na_values='NA',sep=',')
    else: # txt file
        data = pd.read_csv(dataset_path,na_values='NA',sep='\t')
    return data

def removeRowsColumns(data,class_label,ignore_features):
    """ Basic data cleaning: Drops any instances with a missing outcome value as well as any features (ignore_features) specified by user"""
    #Remove instances with missing outcome values
    data = data.dropna(axis=0,how='any',subset=[class_label])
    data = data.reset_index(drop=True)
    data[class_label] = data[class_label].astype(dtype='int8')
    #Remove columns to be ignored in analysis
    data = data.drop(ignore_features,axis=1)
    return data

def idFeatureTypes(x_data,categorical_feature_headers,categorical_cutoff,experiment_path,dataset_name):
    """ Takes a dataframe (of independent variables) with column labels and returns a list of column names identified as
    being categorical based on user defined cutoff (categorical_cutoff). """
    #Identify categorical variables in dataset
    if len(categorical_feature_headers) == 0: #Runs unless user has specified a predifined list of variables to treat as categorical
        categorical_variables = []
        for each in x_data:
            if x_data[each].nunique() <= categorical_cutoff or not pd.api.types.is_numeric_dtype(x_data[each]):
                categorical_variables.append(each)
    else:
        categorical_variables = categorical_feature_headers
    #Pickle list of feature names to be treated as categorical variables
    outfile = open(experiment_path + '/' + dataset_name + '/exploratory/categorical_variables.pickle', 'wb')
    pickle.dump(categorical_variables, outfile)
    outfile.close()
    return categorical_variables

def describeData(data,experiment_path,dataset_name):
    """Conduct and export basic dataset descriptions including basic column statistics, column variable types (i.e. int64 vs. float64),
        and unique value counts for each column"""
    data.describe().to_csv(experiment_path + '/' + dataset_name + '/exploratory/'+'DescribeDataset.csv')
    data.dtypes.to_csv(experiment_path + '/' + dataset_name + '/exploratory/'+'DtypesDataset.csv',header=['DataType'],index_label='Variable')
    data.nunique().to_csv(experiment_path + '/' + dataset_name + '/exploratory/'+'NumUniqueDataset.csv',header=['Count'],index_label='Variable')

def missingnessCounts(data,experiment_path,dataset_name,jupyterRun):
    """ Count and export missing values for all data columns. Also plots a histogram of missingness across all data columns."""
    #Assess Missingness in all data columns
    missing_count = data.isnull().sum()
    totalMissing = data.isnull().sum().sum()
    missing_count.to_csv(experiment_path + '/' + dataset_name + '/exploratory/'+'DataMissingness.csv',header=['Count'],index_label='Variable')
    #Plot a histogram of the missingness observed over all columns in the dataset
    #plt.hist(missing_count,bins=data.shape[0]) #To view the full spectrum of posssible missingness
    plt.hist(missing_count,bins=100)
    plt.xlabel("Missing Value Counts")
    plt.ylabel("Frequency")
    plt.title("Histogram of Missing Value Counts in Dataset")
    plt.savefig(experiment_path + '/' + dataset_name + '/exploratory/'+'DataMissingnessHistogram.png',bbox_inches='tight')
    if eval(jupyterRun):
        plt.show()
    else:
        plt.close('all')
    return totalMissing

def countsSummary(data,class_label,experiment_path,dataset_name,instance_label,match_label,categorical_variables,totalMissing,jupyterRun):
    """ Reports various dataset counts: i.e. number of instances, total features, categorical features, quantitative features, and class counts.
        Also saves a simple bar graph of class counts."""
    #Calculate, print, and export instance and feature counts
    fCount = data.shape[1]-1
    if not instance_label == 'None':
        fCount -= 1
    if not match_label == 'None':
        fCount -=1
    percentMissing = int(totalMissing)/float(data.shape[0]*fCount)
    if jupyterRun:
        print('Data Counts: ----------------')
        print('Instance Count = '+str(data.shape[0]))
        print('Feature Count = '+str(fCount))
        print('    Categorical  = '+str(len(categorical_variables)))
        print('    Quantitative = '+str(fCount - len(categorical_variables)))
        print('Missing Count = '+str(totalMissing))
        print('    Missing Percent = '+str(percentMissing))
    summary = [['instances',data.shape[0]],['features',fCount],['categorical_features',len(categorical_variables)],['quantitative_features',fCount - len(categorical_variables)],['missing_values',totalMissing],['missing_percent',round(percentMissing,5)]]
    dfSummary = pd.DataFrame(summary, columns = ['Variable','Count'])
    dfSummary.to_csv(experiment_path + '/' + dataset_name + '/exploratory/'+'DataCounts.csv',index=None)
    #Calculate, print, and export class counts
    class_counts = data[class_label].value_counts()
    class_counts.to_csv(experiment_path + '/' + dataset_name + '/exploratory/'+'ClassCounts.csv',header=['Count'],index_label='Class')
    print('Class Counts: ----------------')
    print(class_counts)
    #Generate and export class count bar graph
    class_counts.plot(kind='bar')
    plt.ylabel('Count')
    plt.title('Class Counts')
    plt.savefig(experiment_path + '/' + dataset_name + '/exploratory/'+'ClassCountsBarPlot.png',bbox_inches='tight')
    if eval(jupyterRun):
        plt.show()
    else:
        plt.close('all')

def featureCorrelationPlot(x_data,experiment_path,dataset_name,jupyterRun):
    """ Calculates feature correlations via pearson correlation and explorts a respective heatmap visualization. Due to computational expense
        this may not be recommended for datasets with a large number of instances and/or features unless needed. The generated heatmap will be
        difficult to read with a large number of features in the target dataset."""
    #Calculate correlation matrix
    corrmat = x_data.corr(method='pearson')
    #Generate and export correlation heatmap
    f,ax=plt.subplots(figsize=(40,20))
    sns.heatmap(corrmat,vmax=1,square=True)
    plt.savefig(experiment_path + '/' + dataset_name + '/exploratory/'+'FeatureCorrelations.png',bbox_inches='tight')
    if eval(jupyterRun):
        plt.show()
    else:
        plt.close('all')

def reportHeaders(x_data,experiment_path,dataset_name):
    """ Exports dataset header labels for use as a reference later in the pipeline. """
    #Get and Export Original Headers
    headers = x_data.columns.values.tolist()
    with open(experiment_path + '/' + dataset_name + '/exploratory/OriginalFeatureNames.csv',mode='w', newline="") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
    file.close()

def univariateAnalysis(data,experiment_path,dataset_name,class_label,instance_label,match_label,categorical_variables,jupyterRun,topFeatures):
    """ Calculates univariate association significance between each individual feature and class outcome. Assumes categorical outcome using Chi-square test for
        categorical features and Mann-Whitney Test for quantitative features. """
    try: #Try loop added to deal with versions specific change to using mannwhitneyu in scipy and avoid STREAMLINE crash in those circumstances.
        #Create folder for univariate analysis results
        if not os.path.exists(experiment_path + '/' + dataset_name + '/exploratory/univariate_analyses'):
            os.mkdir(experiment_path + '/' + dataset_name + '/exploratory/univariate_analyses')
        #Generate dictionary of p-values for each feature using appropriate test (via test_selector)
        p_value_dict = {}
        for column in data:
            if column != class_label and column != instance_label:
                p_value_dict[column] = test_selector(column,class_label,data,categorical_variables)
        sorted_p_list = sorted(p_value_dict.items(),key = lambda item:item[1])
        #Save p-values to file
        pval_df = pd.DataFrame.from_dict(p_value_dict, orient='index')
        pval_df.to_csv(experiment_path + '/' + dataset_name + '/exploratory/univariate_analyses/Univariate_Significance.csv',index_label='Feature',header=['p-value'])
        #Print results for top features across univariate analyses
        if eval(jupyterRun):
            fCount = data.shape[1]-1
            if not instance_label == 'None':
                fCount -= 1
            if not match_label == 'None':
                fCount -=1
            min_num = min(topFeatures,fCount)
            sorted_p_list_temp = sorted_p_list[: min_num]
            print('Plotting top significant '+ str(min_num) + ' features.')
            print('###################################################')
            print('Significant Univariate Associations:')
            for each in sorted_p_list_temp[:min_num]:
                print(each[0]+": (p-val = "+str(each[1]) +")")
    except:
        sorted_p_list = [] #won't actually be sorted
        print('WARNING: Exploratory univariate analysis failed due to scipy package version error when running mannwhitneyu test. To fix, we recommend updating scipy to version 1.8.0 or greater using: pip install --upgrade scipy')
        for column in data:
            if column != class_label and column != instance_label:
                sorted_p_list.append([column,'None'])
    return sorted_p_list

def test_selector(featureName, class_label, data, categorical_variables):
    """ Selects and applies appropriate univariate association test for a given feature. Returns resulting p-value"""
    p_val = 0
    # Feature and Outcome are discrete/categorical/binary
    if featureName in categorical_variables:
        # Calculate Contingency Table - Counts
        table = pd.crosstab(data[featureName], data[class_label])
        # Univariate association test (Chi Square Test of Independence - Non-parametric)
        c, p, dof, expected = scs.chi2_contingency(table)
        p_val = p
    # Feature is continuous and Outcome is discrete/categorical/binary
    else:
        # Univariate association test (Mann-Whitney Test - Non-parametric)
        try: #works in scipy 1.5.0
            c, p = scs.mannwhitneyu(x=data[featureName].loc[data[class_label] == 0], y=data[featureName].loc[data[class_label] == 1])
        except: #for scipy 1.8.0
            c, p = scs.mannwhitneyu(x=data[featureName].loc[data[class_label] == 0], y=data[featureName].loc[data[class_label] == 1],nan_policy='omit')
        p_val = p
    return p_val

def univariatePlots(data,sorted_p_list,class_label,categorical_variables,experiment_path,dataset_name,sig_cutoff):
    """ Checks whether p-value of each feature is less than or equal to significance cutoff. If so, calls graph_selector to generate an appropriate plot."""
    for i in sorted_p_list: #each feature in sorted p-value dictionary
        if i[1] == 'None':
            pass
        else:
            for j in data: #each feature
                if j == i[0] and i[1] <= sig_cutoff: #ONLY EXPORTS SIGNIFICANT FEATURES
                    graph_selector(j,class_label,data,categorical_variables,experiment_path,dataset_name)

def graph_selector(featureName, class_label, data, categorical_variables,experiment_path,dataset_name):
    """ Assuming a categorical class outcome, a barplot is generated given a categorical feature, and a boxplot is generated given a quantitative feature. """
    if featureName in categorical_variables: # Feature and Outcome are discrete/categorical/binary
        # Generate contingency table count bar plot. ------------------------------------------------------------------------
        # Calculate Contingency Table - Counts
        table = pd.crosstab(data[featureName], data[class_label])
        geom_bar_data = pd.DataFrame(table)
        mygraph = geom_bar_data.plot(kind='bar')
        plt.ylabel('Count')
        new_feature_name = featureName.replace(" ","")       # Deal with the dataset specific characters causing problems in this dataset.
        new_feature_name = new_feature_name.replace("*","")  # Deal with the dataset specific characters causing problems in this dataset.
        new_feature_name = new_feature_name.replace("/","")  # Deal with the dataset specific characters causing problems in this dataset.
        plt.savefig(experiment_path + '/' + dataset_name + '/exploratory/univariate_analyses/'+'Barplot_'+str(new_feature_name)+".png",bbox_inches="tight", format='png')
        plt.close('all')
    else: # Feature is continuous and Outcome is discrete/categorical/binary
        # Generate boxplot-----------------------------------------------------------------------------------------------------
        mygraph = data.boxplot(column=featureName, by=class_label)
        plt.ylabel(featureName)
        plt.title('')
        new_feature_name = featureName.replace(" ","")       # Deal with the dataset specific characters causing problems in this dataset.
        new_feature_name = new_feature_name.replace("*","")  # Deal with the dataset specific characters causing problems in this dataset.
        new_feature_name = new_feature_name.replace("/","")  # Deal with the dataset specific characters causing problems in this dataset.
        plt.savefig(experiment_path + '/' + dataset_name + '/exploratory/univariate_analyses/'+'Boxplot_'+str(new_feature_name)+".png",bbox_inches="tight", format='png')
        plt.close('all')

def cv_partitioner(data, cv_partitions, partition_method, class_label, match_label, randomSeed):
    """ Takes data frame (data), number of cv partitions, partition method (R, S, or M), class label,
    and the column name used for matched CV. Returns list of training and testing dataframe partitions."""
    # Shuffle instances to avoid potential order biases in creating partitions
    data = data.sample(frac=1, random_state=randomSeed).reset_index(drop=True)
    # Convert data frame to list of lists (save header for later)
    header = list(data.columns.values)
    datasetList = list(list(x) for x in zip(*(data[x].values.tolist() for x in data.columns)))
    outcomeIndex = data.columns.get_loc(class_label) # Get classIndex
    if not match_label =='None':
        matchIndex = data.columns.get_loc(match_label) # Get match variable column index
    classList = pd.unique(data[class_label]).tolist()
    del data #memory cleanup
    # Initialize partitions-----------------------------
    partList = []  # Will store partitions
    for x in range(cv_partitions):
        partList.append([])
    # Random Partitioning Method----------------------------------------------------------------
    if partition_method == 'R':
        currPart = 0
        counter = 0
        for row in datasetList:
            partList[currPart].append(row)
            counter += 1
            currPart = counter % cv_partitions
    # Stratified Partitioning Method------------------------------------------------------------
    elif partition_method == 'S':
        # Create data sublists, each having all rows with the same class
        byClassRows = [[] for i in range(len(classList))]  # create list of empty lists (one for each class)
        for row in datasetList:
            # find index in classList corresponding to the class of the current row.
            cIndex = classList.index(row[outcomeIndex])
            byClassRows[cIndex].append(row)
        for classSet in byClassRows:
            currPart = 0
            counter = 0
            for row in classSet:
                partList[currPart].append(row)
                counter += 1
                currPart = counter % cv_partitions
    # Matched partitioning method ---------------------------------------------------------------
    elif partition_method == 'M':
        # Create data sublists, each having all rows with the same match identifier
        matchList = []
        for each in datasetList:
            if each[matchIndex] not in matchList:
                matchList.append(each[matchIndex])
        byMatchRows = [[] for i in range(len(matchList))]  # create list of empty lists (one for each match group)
        for row in datasetList:
            # find index in matchList corresponding to the matchset of the current row.
            mIndex = matchList.index(row[matchIndex])
            row.pop(matchIndex)  # remove match column from partition output
            byMatchRows[mIndex].append(row)
        currPart = 0
        counter = 0
        for matchSet in byMatchRows:  # Go through each unique set of matched instances
            for row in matchSet:  # put all of the instances
                partList[currPart].append(row)
            # move on to next matchset being placed in the next partition.
            counter += 1
            currPart = counter % cv_partitions
        header.pop(matchIndex)  # remove match column from partition output
    else:
        raise Exception('Error: Requested partition method not found.')
    del datasetList # memory cleanup
    # Create (cv_partitions) training and testing sets from partitions -------------------------------------------
    train_dfs = []
    test_dfs = []
    for part in range(0, cv_partitions):
        testList = partList[part]  # Assign testing set as the current partition
        trainList = []
        tempList = []
        for x in range(0, cv_partitions):
            tempList.append(x)
        tempList.pop(part)
        for v in tempList:  # for each training partition
            trainList.extend(partList[v])
        train_dfs.append(pd.DataFrame(trainList, columns=header))
        test_dfs.append(pd.DataFrame(testList, columns=header))
    del partList #memory cleanup
    return train_dfs, test_dfs

def saveCVDatasets(experiment_path,dataset_name,train_dfs,test_dfs):
    """ Saves individual training and testing CV datasets as .csv files"""
    #Generate folder to contain generated CV datasets
    if not os.path.exists(experiment_path + '/' + dataset_name + '/CVDatasets'):
        os.mkdir(experiment_path + '/' + dataset_name + '/CVDatasets')
    #Export training datasets
    counter = 0
    for each in train_dfs:
        a = each.values
        with open(experiment_path + '/' + dataset_name + '/CVDatasets/'+dataset_name+'_CV_' + str(counter) +"_Train.csv", mode="w", newline="") as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(each.columns.values.tolist())
            for row in a:
                writer.writerow(row)
        counter += 1
    #Export testing datasets
    counter = 0
    for each in test_dfs:
        a = each.values
        with open(experiment_path + '/' + dataset_name + '/CVDatasets/'+dataset_name+'_CV_' + str(counter) +"_Test.csv", mode="w", newline="") as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(each.columns.values.tolist())
            for row in a:
                writer.writerow(row)
        file.close()
        counter += 1

def saveRuntime(experiment_path,dataset_name,job_start_time):
    """ Export runtime for this phase of the pipeline on current target dataset"""
    if not os.path.exists(experiment_path + '/' + dataset_name + '/runtime'):
        os.mkdir(experiment_path + '/' + dataset_name + '/runtime')
    runtime_file = open(experiment_path + '/' + dataset_name + '/runtime/runtime_exploratory.txt','w')
    runtime_file.write(str(time.time()-job_start_time))
    runtime_file.close()

if __name__ == '__main__':
    job(sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4],int(sys.argv[5]),sys.argv[6],sys.argv[7],sys.argv[8],sys.argv[9],sys.argv[10],int(sys.argv[11]),sys.argv[12],sys.argv[13],float(sys.argv[14]),sys.argv[15])
