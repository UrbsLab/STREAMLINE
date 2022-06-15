"""
File: DataCompareJob.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 7 of STREAMLINE - This 'Job' script is called by DataCompareMain.py which runs non-parametric statistical analysis
comparing ML algorithm performance between all target datasets included in the original Phase 1 data folder, for each evaluation metric.
Also compares the best overall model for each target dataset, for each evaluation metric. This runs once for the entire pipeline analysis.
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import os
import sys
import glob
import pandas as pd
from scipy import stats
import copy
import matplotlib.pyplot as plt
import numpy as np
import pickle

def job(experiment_path,sig_cutoff,jupyterRun):
    """ Run all elements of data comparison once for the entire analysis pipeline: runs non-parametric statistical analysis
    comparing ML algorithm performance between all target datasets included in the original Phase 1 data folder, for each
    evaluation metric. Also compares the best overall model for each target dataset, for each evaluation metric."""
    # Get dataset paths for all completed dataset analyses in experiment folder
    datasets = os.listdir(experiment_path)
    experiment_name = experiment_path.split('/')[-1] #Name of experiment folder
    removeList = removeList = ['metadata.pickle','metadata.csv','algInfo.pickle','jobsCompleted','logs','jobs','DatasetComparisons','UsefulNotebooks',experiment_name+'_ML_Pipeline_Report.pdf']
    for text in removeList:
        if text in datasets:
            datasets.remove(text)
    datasets = sorted(datasets) #ensures consistent ordering of datasets and assignment of temporary identifier
    dataset_directory_paths = []
    for dataset in datasets:
        full_path = experiment_path + "/" + dataset
        dataset_directory_paths.append(full_path)
    #Unpickle algorithm information from previous phase
    file = open(experiment_path+'/'+"algInfo.pickle", 'rb')
    algInfo = pickle.load(file)
    file.close()
    # Get ML modeling algorithms that were applied in analysis pipeline
    algorithms = []
    name_to_abbrev = {}
    colors = {}
    for key in algInfo:
        if algInfo[key][0]: # If that algorithm was used
            algorithms.append(key)
            name_to_abbrev[key] = (algInfo[key][1])
            colors[key] = (algInfo[key][2])
    abbrev_to_name = dict([(value, key) for key, value in name_to_abbrev.items()])
    # Get list of metric names
    data = pd.read_csv(dataset_directory_paths[0] + '/model_evaluation/Summary_performance_mean.csv', sep=',')
    metrics = data.columns.values.tolist()[1:]
    # Create directory to store dataset statistical comparisons
    if not os.path.exists(experiment_path+'/DatasetComparisons'):
        os.mkdir(experiment_path+'/DatasetComparisons')
    if eval(jupyterRun):
        print('Running Statistical Significance Comparisons Between Multiple Datasets...')
    #Run Kruscall Wallis test (for each algorithm) to determine if there was a significant difference in any metric performance between all analyzed datasets
    kruscallWallis(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff)
    #Run MannWhitney U test (for each algorithm) to determine if there was a significant difference between any pair of datasets for any metric. Runs for all pairs even if kruscall wallis not significant for given metric.
    mannWhitneyU(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff)
    #Run Wilcoxon Rank Sum test (for each algorithm) to determine if there was a significant difference between any pair of datasets for any metric. Runs for all pairs even if kruscall wallis not significant for given metric.
    wilcoxonRank(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff)
    #Run Kruscall Wallist test for each metric comparing all datasets using the best performing algorithm (based on given metric).
    global_data = bestKruscallWallis(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff)
    #Run MannWhitney U test for each metric comparing pairs of datsets using the best performing algorithm (based on given metric).
    bestMannWhitneyU(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff,global_data)
    #Run Wilcoxon Rank sum test for each metric comparing pairs of datsets using the best performing algorithm (based on given metric).
    bestWilcoxonRank(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff,global_data)
    #Generate boxplots comparing average algorithm performance (for a given metric) across all dataset comparisons
    if eval(jupyterRun):
        print('Generate Boxplots Comparing Dataset Performance...')
    dataCompareBPAll(experiment_path,metrics,dataset_directory_paths,algorithms,colors,jupyterRun)
    #Generate boxplots comparing a specific algorithm's CV performance (for AUC_ROC or AUC_PRC) across all dataset comparisons
    dataCompareBP(experiment_path,metrics,dataset_directory_paths,algorithms,name_to_abbrev,jupyterRun)
    # Print phase completion
    print("Phase 7 complete")
    job_file = open(experiment_path + '/jobsCompleted/job_data_compare' + '.txt', 'w')
    job_file.write('complete')
    job_file.close()

def kruscallWallis(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff):
    """ For each algorithm apply non-parametric Kruskal Wallis one-way ANOVA on ranks. Determines if there is a statistically significant difference in performance between original target datasets across CV runs.
    Completed for each standard metric separately."""
    label = ['Statistic', 'P-Value', 'Sig(*)']
    i = 1
    for dataset in datasets:
        label.append('Median_D' + str(i))
        #label.append('Mean_D' + str(i))
        #label.append('Std_D' + str(i))
        i += 1
    for algorithm in algorithms:
        kruskal_summary = pd.DataFrame(index=metrics, columns=label)
        for metric in metrics:
            tempArray = []
            medList = []
            #aveList = []
            #sdList = []
            for dataset_path in dataset_directory_paths:
                filename = dataset_path+'/model_evaluation/'+name_to_abbrev[algorithm]+'_performance.csv'
                td = pd.read_csv(filename)
                tempArray.append(td[metric])
                medList.append(td[metric].median())
                #aveList.append(td[metric].mean())
                #sdList.append(td[metric].std())
            try: #Run Kruscall Wallis
                result = stats.kruskal(*tempArray)
            except:
                result = ['NA',1]
            kruskal_summary.at[metric, 'Statistic'] = str(round(result[0], 6))
            kruskal_summary.at[metric, 'P-Value'] = str(round(result[1], 6))
            if result[1] < sig_cutoff:
                kruskal_summary.at[metric, 'Sig(*)'] = str('*')
            else:
                kruskal_summary.at[metric, 'Sig(*)'] = str('')
            #for j in range(len(aveList)):
            for j in range(len(medList)):
                kruskal_summary.at[metric, 'Median_D' + str(j+1)] = str(round(medList[j], 6))
                #kruskal_summary.at[metric, 'Mean_D' + str(j+1)] = str(round(aveList[j], 6))
                #kruskal_summary.at[metric, 'Std_D' + str(j+1)] = str(round(sdList[j], 6))
        #Export analysis summary to .csv file
        kruskal_summary.to_csv(experiment_path+'/DatasetComparisons/KruskalWallis_'+algorithm+'.csv')

def wilcoxonRank(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff):
    """ For each algorithm, apply non-parametric Wilcoxon Rank Sum (pairwise comparisons). This tests individual algorithm pairs of original target datasets (for each metric)
    to determine if there is a statistically significant difference in performance across CV runs. Test statistic will be zero if all scores from one set are
    larger than the other."""
    label = ['Metric', 'Data1', 'Data2', 'Statistic', 'P-Value', 'Sig(*)']
    for i in range(1,3):
        label.append('Median_Data' + str(i))
        #label.append('Mean_Data' + str(i))
        #label.append('Std_Data' + str(i))
    for algorithm in algorithms:
        master_list = []
        for metric in metrics:
            for x in range(0,len(dataset_directory_paths)-1):
                for y in range(x+1,len(dataset_directory_paths)):
                    tempList = []
                    #Grab info on first dataset
                    file1 = dataset_directory_paths[x]+'/model_evaluation/'+name_to_abbrev[algorithm]+'_performance.csv'
                    td1 = pd.read_csv(file1)
                    set1 = td1[metric]
                    med1 = td1[metric].median()
                    #ave1 = td1[metric].mean()
                    #sd1 = td1[metric].std()
                    #Grab info on second dataset
                    file2 = dataset_directory_paths[y] + '/model_evaluation/' + name_to_abbrev[algorithm] + '_performance.csv'
                    td2 = pd.read_csv(file2)
                    set2 = td2[metric]
                    med2 = td2[metric].median()
                    #ave2 = td2[metric].mean()
                    #sd2 = td2[metric].std()
                    #handle error when metric values are equal for both algorithms
                    if set1.equals(set2):  # Check if all nums are equal in sets
                        result = ['NA', 1]
                    else:
                        result = stats.wilcoxon(set1, set2)
                    #Summarize test information in list
                    tempList.append(str(metric))
                    tempList.append('D'+str(x+1))
                    tempList.append('D'+str(y+1))
                    if set1.equals(set2):
                        tempList.append(result[0])
                    else:
                        tempList.append(str(round(result[0], 6)))
                    tempList.append(str(round(result[1], 6)))
                    if result[1] < sig_cutoff:
                        tempList.append(str('*'))
                    else:
                        tempList.append(str(''))
                    tempList.append(str(round(med1, 6)))
                    tempList.append(str(round(med2, 6)))
                    #tempList.append(str(round(ave1, 6)))
                    #tempList.append(str(round(sd1, 6)))
                    #tempList.append(str(round(ave2, 6)))
                    #tempList.append(str(round(sd2, 6)))
                    master_list.append(tempList)
        #Export test results
        df = pd.DataFrame(master_list)
        df.columns = label
        df.to_csv(experiment_path+'/DatasetComparisons/WilcoxonRank_'+algorithm+'.csv',index=False)

def mannWhitneyU(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff):
    """ For each algorithm, apply non-parametric Mann Whitney U-test (pairwise comparisons). Mann Whitney tests dataset pairs (for each metric)
    to determine if there is a statistically significant difference in performance across CV runs. Test statistic will be zero if all scores from one set are
    larger than the other."""
    label = ['Metric', 'Data1', 'Data2', 'Statistic', 'P-Value', 'Sig(*)']
    for i in range(1,3):
        label.append('Median_Data' + str(i))
        #label.append('Mean_Data' + str(i))
        #label.append('Std_Data' + str(i))
    for algorithm in algorithms:
        master_list = []
        for metric in metrics:
            for x in range(0,len(dataset_directory_paths)-1):
                for y in range(x+1,len(dataset_directory_paths)):
                    tempList = []
                    #Grab info on first dataset
                    file1 = dataset_directory_paths[x]+'/model_evaluation/'+name_to_abbrev[algorithm]+'_performance.csv'
                    td1 = pd.read_csv(file1)
                    set1 = td1[metric]
                    med1 = td1[metric].median()
                    #ave1 = td1[metric].mean()
                    #sd1 = td1[metric].std()
                    #Grab info on second dataset
                    file2 = dataset_directory_paths[y] + '/model_evaluation/' + name_to_abbrev[algorithm] + '_performance.csv'
                    td2 = pd.read_csv(file2)
                    set2 = td2[metric]
                    med2 = td2[metric].median()
                    #ave2 = td2[metric].mean()
                    #sd2 = td2[metric].std()
                    #handle error when metric values are equal for both algorithms
                    if set1.equals(set2):  # Check if all nums are equal in sets
                        result = ['NA', 1]
                    else:
                        result = stats.mannwhitneyu(set1, set2)
                    #Summarize test information in list
                    tempList.append(str(metric))
                    tempList.append('D'+str(x+1))
                    tempList.append('D'+str(y+1))
                    if set1.equals(set2):
                        tempList.append(result[0])
                    else:
                        tempList.append(str(round(result[0], 6)))
                    tempList.append(str(round(result[1], 6)))
                    if result[1] < sig_cutoff:
                        tempList.append(str('*'))
                    else:
                        tempList.append(str(''))
                    tempList.append(str(round(med1, 6)))
                    tempList.append(str(round(med2, 6)))
                    #tempList.append(str(round(ave1, 6)))
                    #tempList.append(str(round(sd1, 6)))
                    #tempList.append(str(round(ave2, 6)))
                    #tempList.append(str(round(sd2, 6)))
                    master_list.append(tempList)
        #Export test results
        df = pd.DataFrame(master_list)
        df.columns = label
        df.to_csv(experiment_path+'/DatasetComparisons/MannWhitney_'+algorithm+'.csv',index=False)

def bestKruscallWallis(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff):
    """ For best performing algorithm on a given metric and dataset, apply non-parametric Kruskal Wallis one-way ANOVA on ranks.
    Determines if there is a statistically significant difference in performance between original target datasets across CV runs
    on best algorithm for given metric."""
    label = ['Statistic', 'P-Value', 'Sig(*)']
    i = 1
    for dataset in datasets:
        label.append('Best_Alg_D' + str(i))
        label.append('Median_D' + str(i))
        #label.append('Mean_D' + str(i))
        #label.append('Std_D' + str(i))
        i += 1
    kruskal_summary = pd.DataFrame(index=metrics, columns=label)
    global_data = []
    for metric in metrics:
        best_list = []
        best_data = []
        for dataset_path in dataset_directory_paths:
            alg_med = []
            #alg_ave = []
            #alg_st = []
            alg_data = []
            for algorithm in algorithms:
                filename = dataset_path+'/model_evaluation/'+name_to_abbrev[algorithm]+'_performance.csv'
                td = pd.read_csv(filename)
                alg_med.append(td[metric].median())
                #alg_ave.append(td[metric].mean())
                #alg_st.append(td[metric].std())
                alg_data.append(td[metric])
            # Find best algorithm for given metric based on average
            #best_ave = max(alg_ave)
            best_med = max(alg_med)
            #best_index = alg_ave.index(best_ave)
            best_index = alg_med.index(best_med)
            #best_sd = alg_st[best_index]
            best_alg = algorithms[best_index]
            best_data.append(alg_data[best_index])
            #best_list.append([best_alg, best_ave, best_sd])
            best_list.append([best_alg, best_med])
        global_data.append([best_data, best_list])
        try:
            result = stats.kruskal(*best_data)
            kruskal_summary.at[metric, 'Statistic'] = str(round(result[0], 6))
            kruskal_summary.at[metric, 'P-Value'] = str(round(result[1], 6))
            if result[1] < sig_cutoff:
                kruskal_summary.at[metric, 'Sig(*)'] = str('*')
            else:
                kruskal_summary.at[metric, 'Sig(*)'] = str('')
        except ValueError:
            kruskal_summary.at[metric, 'Statistic'] = str(round('NA', 6))
            kruskal_summary.at[metric, 'P-Value'] = str(round('NA', 6))
            kruskal_summary.at[metric, 'Sig(*)'] = str('')
        for j in range(len(best_list)):
            kruskal_summary.at[metric, 'Best_Alg_D' + str(j+1)] = str(best_list[j][0])
            kruskal_summary.at[metric, 'Median_D' + str(j+1)] = str(round(best_list[j][1], 6))
            #kruskal_summary.at[metric, 'Mean_D' + str(j+1)] = str(round(best_list[j][1], 6))
            #kruskal_summary.at[metric, 'Std_D' + str(j+1)] = str(round(best_list[j][2], 6))
    #Export analysis summary to .csv file
    kruskal_summary.to_csv(experiment_path + '/DatasetComparisons/BestCompare_KruskalWallis.csv')
    return global_data

def bestMannWhitneyU(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff,global_data):
    """ For best performing algorithm on a given metric and dataset, apply non-parametric Mann Whitney U-test (pairwise comparisons). Mann Whitney tests dataset pairs (for each metric)
    to determine if there is a statistically significant difference in performance across CV runs. Test statistic will be zero if all scores from one set are
    larger than the other."""
    label = ['Metric', 'Data1', 'Data2', 'Statistic', 'P-Value', 'Sig(*)']
    for i in range(1,3):
        label.append('Best_Alg_Data' + str(i))
        label.append('Median_Data' + str(i))
        #label.append('Mean_Data' + str(i))
        #label.append('Std_Data' + str(i))
    master_list = []
    j = 0
    for metric in metrics:
        for x in range(0, len(datasets) - 1):
            for y in range(x + 1, len(datasets)):
                tempList = []
                set1 = global_data[j][0][x]
                med1 = global_data[j][1][x][1]
                #ave1 = global_data[j][1][x][1]
                #sd1 = global_data[j][1][x][2]
                set2 = global_data[j][0][y]
                med2 = global_data[j][1][y][1]
                #ave2 = global_data[j][1][y][1]
                #sd2 = global_data[j][1][y][2]
                #handle error when metric values are equal for both algorithms
                if set1.equals(set2):  # Check if all nums are equal in sets
                    result = ['NA', 1]
                else:
                    result = stats.mannwhitneyu(set1, set2)
                #Summarize test information in list
                tempList.append(str(metric))
                tempList.append('D'+str(x+1))
                tempList.append('D'+str(y+1))
                if set1.equals(set2):
                    tempList.append(result[0])
                else:
                    tempList.append(str(round(result[0], 6)))
                tempList.append(str(round(result[1], 6)))
                if result[1] < sig_cutoff:
                    tempList.append(str('*'))
                else:
                    tempList.append(str(''))
                tempList.append(global_data[j][1][x][0])
                tempList.append(str(round(med1, 6)))
                #tempList.append(str(round(ave1, 6)))
                #tempList.append(str(round(sd1, 6)))
                tempList.append(global_data[j][1][y][0])
                tempList.append(str(round(med2, 6)))
                #tempList.append(str(round(ave2, 6)))
                #tempList.append(str(round(sd2, 6)))
                master_list.append(tempList)
        j += 1
    #Export analysis summary to .csv file
    df = pd.DataFrame(master_list)
    df.columns = label
    df.to_csv(experiment_path + '/DatasetComparisons/BestCompare_MannWhitney.csv',index=False)

def bestWilcoxonRank(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff,global_data):
    """ For best performing algorithm on a given metric and dataset, apply non-parametric Mann Whitney U-test (pairwise comparisons). Mann Whitney tests dataset pairs (for each metric)
    to determine if there is a statistically significant difference in performance across CV runs. Test statistic will be zero if all scores from one set are
    larger than the other."""
    # Best Mann Whitney (Pairwise comparisons)
    label = ['Metric', 'Data1', 'Data2', 'Statistic', 'P-Value', 'Sig(*)']
    for i in range(1,3):
        label.append('Best_Alg_Data' + str(i))
        label.append('Median_Data' + str(i))
        #label.append('Mean_Data' + str(i))
        #label.append('Std_Data' + str(i))
    master_list = []
    j = 0
    for metric in metrics:
        for x in range(0, len(datasets) - 1):
            for y in range(x + 1, len(datasets)):
                tempList = []
                set1 = global_data[j][0][x]
                med1 = global_data[j][1][x][1]
                #ave1 = global_data[j][1][x][1]
                #sd1 = global_data[j][1][x][2]
                set2 = global_data[j][0][y]
                med2 = global_data[j][1][y][1]
                #ave2 = global_data[j][1][y][1]
                #sd2 = global_data[j][1][y][2]
                #handle error when metric values are equal for both algorithms
                if set1.equals(set2):  # Check if all nums are equal in sets
                    result = ['NA', 1]
                else:
                    result = stats.wilcoxon(set1, set2)
                #Summarize test information in list
                tempList.append(str(metric))
                tempList.append('D'+str(x+1))
                tempList.append('D'+str(y+1))
                if set1.equals(set2):
                    tempList.append(result[0])
                else:
                    tempList.append(str(round(result[0], 6)))
                tempList.append(str(round(result[1], 6)))
                if result[1] < sig_cutoff:
                    tempList.append(str('*'))
                else:
                    tempList.append(str(''))
                tempList.append(global_data[j][1][x][0])
                tempList.append(str(round(med1, 6)))
                #tempList.append(str(round(ave1, 6)))
                #tempList.append(str(round(sd1, 6)))
                tempList.append(global_data[j][1][y][0])
                tempList.append(str(round(med2, 6)))
                #tempList.append(str(round(ave2, 6)))
                #tempList.append(str(round(sd2, 6)))
                master_list.append(tempList)
        j += 1
    #Export analysis summary to .csv file
    df = pd.DataFrame(master_list)
    df.columns = label
    df.to_csv(experiment_path + '/DatasetComparisons/BestCompare_WilcoxonRank.csv',index=False)

def dataCompareBPAll(experiment_path,metrics,dataset_directory_paths,algorithms,colors,jupyterRun):
    """ Generate a boxplot comparing algorithm performance (CV average of each target metric) across all target datasets to be compared."""
    #colors = ['grey','black','yellow','orange','bisque','purple','aqua','blue','red','firebrick','deepskyblue','seagreen','lightcoral']
    if not os.path.exists(experiment_path+'/DatasetComparisons/dataCompBoxplots'):
        os.mkdir(experiment_path+'/DatasetComparisons/dataCompBoxplots')
    for metric in metrics: #One boxplot generated for each available metric
        df = pd.DataFrame()
        data_name_list = []
        alg_values_dict = {}
        for algorithm in algorithms: #Dictionary of all algorithms run that will each have a list of respective mean metric value
            alg_values_dict[algorithm] = [] #Used to generate algorithm lines on top of boxplot
        for each in dataset_directory_paths: #each target dataset
            data_name_list.append(each.split('/')[-1])
            data = pd.read_csv(each + '/model_evaluation/Summary_performance_mean.csv', sep=',', index_col=0)
            rownames = data.index.values # makes a list of algorithm names from file
            rownames = list(rownames)
            #Grab data in metric column
            col = data[metric] #Dataframe of average target metric values for each algorithm
            colList = data[metric].tolist() #List of average target metric values for each algorithm
            #for j in range(len(colList)): #For each algorithm
            for j in range(len(rownames)): #For each algorithm
                alg_values_dict[rownames[j]].append(colList[j])
            # Create dataframe of average target metric where columns are datasets, and rows are algorithms
            df = pd.concat([df, col], axis=1)
        df.columns = data_name_list
        # Generate boxplot (with legend for each box) ---------------------------------------
        # Plot boxplots
        boxplot = df.boxplot(column=data_name_list,rot=90)
        # Plot lines for each algorithm (to illustrate algorithm performance trajectories between datasets)
        for i in range(len(algorithms)):
            plt.plot(np.arange(len(dataset_directory_paths))+1,alg_values_dict[algorithms[i]], color = colors[algorithms[i]], label=algorithms[i])
        #Specify plot labels
        plt.ylabel(str(metric))
        plt.xlabel('Dataset')
        plt.legend(loc="upper left", bbox_to_anchor=(1.01,1))
        #Export and/or show plot
        plt.savefig(experiment_path+'/DatasetComparisons/dataCompBoxplots/DataCompareAllModels_'+metric+'.png', bbox_inches="tight")
        if eval(jupyterRun):
            plt.show()
        else:
            plt.close('all')

def dataCompareBP(experiment_path,metrics,dataset_directory_paths,algorithms,name_to_abbrev,jupyterRun):
    """ Generate a boxplot comparing average algorithm performance (for a given target metric) across all target datasets to be compared."""
    metricList = ['ROC AUC','PRC AUC'] #Hard coded
    if not os.path.exists(experiment_path+'/DatasetComparisons/dataCompBoxplots'):
        os.mkdir(experiment_path+'/DatasetComparisons/dataCompBoxplots')
    for algorithm in algorithms:
        for metric in metricList:
            df = pd.DataFrame()
            data_name_list = []
            for each in dataset_directory_paths:
                data_name_list.append(each.split('/')[-1])
                data = pd.read_csv(each + '/model_evaluation/'+name_to_abbrev[algorithm]+'_performance.csv', sep=',')
                #Grab data in metric column
                col = data[metric]
                df = pd.concat([df, col], axis=1)
            df.columns = data_name_list
            # Generate boxplot (with legend for each box)
            boxplot = df.boxplot(column=data_name_list,rot=90)
            #Specify plot labels
            plt.ylabel(str(metric))
            plt.xlabel('Dataset')
            #Export and/or show plot
            plt.savefig(experiment_path+'/DatasetComparisons/dataCompBoxplots/DataCompare_'+name_to_abbrev[algorithm]+'_'+metric+'.png', bbox_inches="tight")
            if eval(jupyterRun):
                print(algorithm)
                plt.show()
            else:
                plt.close('all')

if __name__ == '__main__':
    job(sys.argv[1],float(sys.argv[2]),sys.argv[3])
