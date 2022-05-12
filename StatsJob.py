"""
File: StatsJob.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 6 of STREAMLINE - This 'Job' script is called by StatsMain.py and creates summaries of ML classification evaluation statistics
            (means and standard deviations), ROC and PRC plots (comparing CV performance in the same ML algorithm and comparing average performance
            between ML algorithms), model feature importance averages over CV runs, boxplots comparing ML algorithms for each metric, Kruskal Wallis
            and Mann Whitney statistical comparsions between ML algorithms, model feature importance boxplots for each algorithm, and composite feature
            importance plots summarizing model feature importance across all ML algorithms. It is run for a single dataset from the original target
            dataset folder (data_path) in Phase 1 (i.e. stats summary completed for all cv datasets).
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import sys
import time
import pandas as pd
import glob
import numpy as np
from scipy import stats
#from scipy import interp,stats
import matplotlib.pyplot as plt
from matplotlib import rc
import os
from sklearn.metrics import auc
import csv
from statistics import mean,stdev
import pickle
import copy
from sklearn import tree
from subprocess import call
from IPython.display import display_pdf

def job(full_path,plot_ROC,plot_PRC,plot_FI_box,class_label,instance_label,cv_partitions,scale_data,plot_metric_boxplots,primary_metric,top_model_features,sig_cutoff,jupyterRun):
    """ Run all elements of stats summary and analysis for one one the original phase 1 datasets: summaries of average and standard deviations for all metrics and modeling algorithms,
    ROC and PRC plots (comparing CV performance in the same ML algorithm and comparing average performance between ML algorithms), model feature importance averages over CV runs,
    boxplots comparing ML algorithms for each metric, Kruskal Wallis and Mann Whitney statistical comparsions between ML algorithms, model feature importance boxplots for each
    algorithm, and composite feature importance plots summarizing model feature importance across all ML algorithms"""
    job_start_time = time.time() #for tracking phase runtime
    data_name = full_path.split('/')[-1]
    experiment_path = '/'.join(full_path.split('/')[:-1])
    if eval(jupyterRun):
        print('Running Statistics Summary for '+str(data_name))
    #Unpickle algorithm information from previous phase
    file = open(experiment_path+'/'+"algInfo.pickle", 'rb')
    algInfo = pickle.load(file)
    file.close()
    #Translate metric name from scikitlearn standard (currently balanced accuracy is hardcoded for use in generating FI plots due to no-skill normalization)
    metric_term_dict = {'balanced_accuracy': 'Balanced Accuracy','accuracy': 'Accuracy','f1': 'F1_Score','recall': 'Sensitivity (Recall)','precision': 'Precision (PPV)','roc_auc': 'ROC_AUC'}
    primary_metric = metric_term_dict[primary_metric]
    #Get algorithms run, specify algorithm abbreviations, colors to use for algorithms in plots, and original ordered feature name list
    algorithms,abbrev,colors,original_headers = preparation(full_path,algInfo)
    #Gather and summarize all evaluation metrics for each algorithm across all CVs. Returns result_table used to plot average ROC and PRC plots and metric_dict organizing all metrics over all algorithms and CVs.
    result_table,metric_dict = primaryStats(algorithms,original_headers,cv_partitions,full_path,data_name,instance_label,class_label,abbrev,colors,plot_ROC,plot_PRC,jupyterRun)
    #Plot ROC and PRC curves comparing average ML algorithm performance (averaged over all CVs)
    if eval(jupyterRun):
        print('Generating ROC and PRC plots...')
    doPlotROC(result_table,colors,full_path,jupyterRun)
    doPlotPRC(result_table,colors,full_path,data_name,instance_label,class_label,jupyterRun)
    #Make list of metric names
    if eval(jupyterRun):
        print('Saving Metric Summaries...')
    metrics = list(metric_dict[algorithms[0]].keys())
    #Save metric means and standard deviations
    saveMetricMeans(full_path,metrics,metric_dict)
    saveMetricStd(full_path,metrics,metric_dict)
    #Generate boxplots comparing algorithm performance for each standard metric, if specified by user
    if eval(plot_metric_boxplots):
        if eval(jupyterRun):
            print('Generating Metric Boxplots...')
        metricBoxplots(full_path,metrics,algorithms,metric_dict,jupyterRun)
    #Calculate and export Kruskal Wallis, Mann Whitney, and wilcoxon Rank sum stats if more than one ML algorithm has been run (for the comparison) - note stats are based on comparing the multiple CV models for each algorithm.
    if len(algorithms) > 1:
        if eval(jupyterRun):
            print('Running Non-Parametric Statistical Significance Analysis...')
        kruskal_summary = kruskalWallis(full_path,metrics,algorithms,metric_dict,sig_cutoff)
        wilcoxonRank(full_path,metrics,algorithms,metric_dict,kruskal_summary,sig_cutoff)
        mannWhitneyU(full_path,metrics,algorithms,metric_dict,kruskal_summary,sig_cutoff)
    #Prepare for feature importance visualizations
    if eval(jupyterRun):
        print('Preparing for Model Feature Importance Plotting...')
    fi_df_list,fi_ave_list,fi_ave_norm_list,ave_metric_list,all_feature_list,non_zero_union_features,non_zero_union_indexes = prepFI(algorithms,full_path,abbrev,metric_dict,'Balanced Accuracy')
    #Select 'top' features for composite vizualization
    featuresToViz = selectForCompositeViz(top_model_features,non_zero_union_features,non_zero_union_indexes,algorithms,ave_metric_list,fi_ave_norm_list)
    #Generate FI boxplots for each modeling algorithm if specified by user
    if eval(plot_FI_box):
        if eval(jupyterRun):
            print('Generating Feature Importance Boxplots and Histograms...')
        doFIBoxplots(full_path,fi_df_list,fi_ave_list,algorithms,original_headers,top_model_features,jupyterRun)
        doFI_Histogram(full_path, fi_ave_list, algorithms, jupyterRun)
    #Visualize composite FI - Currently set up to only use Balanced Accuracy for composite FI plot visualization
    if eval(jupyterRun):
        print('Generating Composite Feature Importance Plots...')
    #Take top feature names to vizualize and get associated feature importance values for each algorithm, and original data ordered feature names list
    top_fi_ave_norm_list,all_feature_listToViz = getFI_To_Viz_Sorted(featuresToViz,all_feature_list,algorithms,fi_ave_norm_list) #If we want composite FI plots to be displayed in descenting total bar height order.
    #Generate Normalized composite FI plot
    composite_FI_plot(top_fi_ave_norm_list, algorithms, list(colors.values()), all_feature_listToViz, 'Norm',full_path,jupyterRun, 'Normalized Feature Importance')
    #Fractionate FI scores for normalized and fractionated composite FI plot
    fracLists = fracFI(top_fi_ave_norm_list)
    #Generate Normalized and Fractioned composite FI plot
    composite_FI_plot(fracLists, algorithms, list(colors.values()), all_feature_listToViz, 'Norm_Frac',full_path,jupyterRun, 'Normalized and Fractioned Feature Importance')
    #Weight FI scores for normalized and (model performance) weighted composite FI plot
    weightedLists,weights = weightFI(ave_metric_list,top_fi_ave_norm_list)
    #Generate Normalized and Weighted Compount FI plot
    composite_FI_plot(weightedLists, algorithms, list(colors.values()), all_feature_listToViz, 'Norm_Weight',full_path,jupyterRun, 'Normalized and Weighted Feature Importance')
    #Weight the Fractionated FI scores for normalized,fractionated, and weighted compount FI plot
    weightedFracLists = weightFracFI(fracLists,weights)
    #Generate Normalized, Fractionated, and Weighted Compount FI plot
    composite_FI_plot(weightedFracLists, algorithms, list(colors.values()), all_feature_listToViz, 'Norm_Frac_Weight',full_path,jupyterRun, 'Normalized, Fractioned, and Weighted Feature Importance')
    #Export phase runtime
    saveRuntime(full_path,job_start_time)
    #Parse all pipeline runtime files into a single runtime report
    parseRuntime(full_path,algorithms,abbrev)
    # Print phase completion
    print(data_name + " phase 5 complete")
    job_file = open(experiment_path + '/jobsCompleted/job_stats_' + data_name + '.txt', 'w')
    job_file.write('complete')
    job_file.close()

def preparation(full_path,algInfo):
    """ Creates directory for all results files, decodes included ML modeling algorithms that were run, specifies figure abbreviations for algorithms
    and color to use for each algorithm in plots, and loads original ordered feature name list to use as a reference to facilitate combining feature
    importance results across cv runs where different features may have been dropped during the feature selection phase."""
    #Create Directory
    if not os.path.exists(full_path+'/model_evaluation'):
        os.mkdir(full_path+'/model_evaluation')

    #Extract the original algorithm name, abreviated name, and color to use for each algortithm run by users
    algorithms = []
    abbrev = {}
    colors = {}
    for key in algInfo:
        if algInfo[key][0]: # If that algorithm was used
            algorithms.append(key)
            abbrev[key] = (algInfo[key][1])
            colors[key] = (algInfo[key][2])

    original_headers = pd.read_csv(full_path+"/exploratory/OriginalFeatureNames.csv",sep=',').columns.values.tolist() #Get Original Headers
    return algorithms,abbrev,colors,original_headers

def primaryStats(algorithms,original_headers,cv_partitions,full_path,data_name,instance_label,class_label,abbrev,colors,plot_ROC,plot_PRC,jupyterRun):
    """ Combine classification metrics and model feature importance scores as well as ROC and PRC plot data across all CV datasets.
    Generate ROC and PRC plots comparing separate CV models for each individual modeling algorithm."""
    result_table = []
    metric_dict = {}
    for algorithm in algorithms: #completed for each individual ML modeling algorithm
        alg_result_table = [] #stores values used in ROC and PRC plots
        # Define evaluation stats variable lists
        s_bac = [] # balanced accuracies
        s_ac = [] # standard accuracies
        s_f1 = [] # F1 scores
        s_re = [] # recall values
        s_sp = [] # specificities
        s_pr = [] # precision values
        s_tp = [] # true positives
        s_tn = [] # true negatives
        s_fp = [] # false positives
        s_fn = [] # false negatives
        s_npv = [] # negative predictive values
        s_lrp = [] # likelihood ratio positive values
        s_lrm = [] # likelihood ratio negative values
        # Define feature importance lists
        FI_all = [] # used to save model feature importances individually for each cv within single summary file (all original features in dataset prior to feature selection included)
        # Define ROC plot variable lists
        tprs = [] # stores interpolated true postitive rates for average CV line in ROC
        aucs = [] #stores individual CV areas under ROC curve to calculate average
        mean_fpr = np.linspace(0, 1, 100) #used to plot average of CV line in ROC plot
        mean_recall = np.linspace(0, 1, 100) #used to plot average of CV line in PRC plot
        # Define PRC plot variable lists
        precs = [] #stores interpolated precision values for average CV line in PRC
        praucs = [] #stores individual CV areas under PRC curve to calculate average
        aveprecs = [] #stores individual CV average precisions for PRC to calculate CV average
        #Gather statistics over all CV partitions
        for cvCount in range(0,cv_partitions):
            #Unpickle saved metrics from previous phase
            result_file = full_path+'/model_evaluation/pickled_metrics/'+abbrev[algorithm]+"_CV_"+str(cvCount)+"_metrics.pickle"
            file = open(result_file, 'rb')
            results = pickle.load(file) #[metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi, probas_]
            file.close()
            #Separate pickled results
            metricList = results[0]
            fpr = results[1]
            tpr = results[2]
            roc_auc = results[3]
            prec = results[4]
            recall = results[5]
            prec_rec_auc = results[6]
            ave_prec = results[7]
            fi = results[8]
            #Separate metrics from metricList
            s_bac.append(metricList[0])
            s_ac.append(metricList[1])
            s_f1.append(metricList[2])
            s_re.append(metricList[3])
            s_sp.append(metricList[4])
            s_pr.append(metricList[5])
            s_tp.append(metricList[6])
            s_tn.append(metricList[7])
            s_fp.append(metricList[8])
            s_fn.append(metricList[9])
            s_npv.append(metricList[10])
            s_lrp.append(metricList[11])
            s_lrm.append(metricList[12])
            #update list that stores values used in ROC and PRC plots
            alg_result_table.append([fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec]) # alg_result_table.append([fpr, tpr, roc_auc, recall, prec, prec_rec_auc, ave_prec])
            # Update ROC plot variable lists needed to plot all CVs in one ROC plot
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            #tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(roc_auc)
            # Update PRC plot variable lists needed to plot all CVs in one PRC plot
            precs.append(np.interp(mean_recall, recall, prec))
            #precs.append(interp(mean_recall, recall, prec))
            #precs.append(interp(mean_recall, prec, recall))
            praucs.append(prec_rec_auc)
            aveprecs.append(ave_prec)
            # Format feature importance scores as list (takes into account that all features are not in each CV partition)
            tempList = []
            j = 0
            headers = pd.read_csv(full_path+'/CVDatasets/'+data_name+'_CV_'+str(cvCount)+'_Test.csv').columns.values.tolist()
            if instance_label != 'None':
                headers.remove(instance_label)
            headers.remove(class_label)
            for each in original_headers:
                if each in headers:  # Check if current feature from original dataset was in the partition
                    # Deal with features not being in original order (find index of current feature list.index()
                    f_index = headers.index(each)
                    tempList.append(fi[f_index])
                else:
                    tempList.append(0)
                j += 1
            FI_all.append(tempList)

        if jupyterRun:
            print(algorithm)
        #Define values for the mean ROC line (mean of individual CVs)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        #Generate ROC Plot (including individual CV's lines, average line, and no skill line) - based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html-----------------------
        if eval(plot_ROC):
            # Set figure dimensions
            plt.rcParams["figure.figsize"] = (6,6)
            # Plot individual CV ROC lines
            for i in range(cv_partitions):
                plt.plot(alg_result_table[i][0], alg_result_table[i][1], lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.3f)' % (i, alg_result_table[i][2]))
            # Plot no-skill line
            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black',label='No-Skill', alpha=.8)
            # Plot average line for all CVs
            std_auc = np.std(aucs) # AUC standard deviations across CVs
            plt.plot(mean_fpr, mean_tpr, color=colors[algorithm],label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),lw=2, alpha=.8)
            # Plot standard deviation grey zone of curves
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
            #Specify plot axes,labels, and legend
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="upper left", bbox_to_anchor=(1.01,1))
            #Export and/or show plot
            plt.savefig(full_path+'/model_evaluation/'+abbrev[algorithm]+"_ROC.png", bbox_inches="tight")
            if eval(jupyterRun):
                plt.show()
            else:
                plt.close('all')

        #Define values for the mean PRC line (mean of individual CVs)
        mean_prec = np.mean(precs, axis=0)
        mean_pr_auc = np.mean(praucs)
        #Generate PRC Plot (including individual CV's lines, average line, and no skill line)------------------------------------------------------------------------------------------------------------------
        if eval(plot_PRC):
            # Set figure dimensions
            plt.rcParams["figure.figsize"] = (6,6)
            # Plot individual CV PRC lines
            for i in range(cv_partitions):
                plt.plot(alg_result_table[i][4], alg_result_table[i][3], lw=1, alpha=0.3, label='PRC fold %d (AUC = %0.3f)' % (i, alg_result_table[i][5]))
            #Estimate no skill line based on the fraction of cases found in the first test dataset
            test = pd.read_csv(full_path + '/CVDatasets/' + data_name + '_CV_0_Test.csv') #Technically there could be a unique no-skill line for each CV dataset based on final class balance (however only one is needed, and stratified CV attempts to keep partitions with similar/same class balance)
            testY = test[class_label].values
            noskill = len(testY[testY == 1]) / len(testY)  # Fraction of cases
            # Plot no-skill line
            plt.plot([0, 1], [noskill, noskill], color='black', linestyle='--', label='No-Skill', alpha=.8)
            # Plot average line for all CVs
            std_pr_auc = np.std(praucs)
            plt.plot(mean_recall, mean_prec, color=colors[algorithm],label=r'Mean PRC (AUC = %0.3f $\pm$ %0.3f)' % (mean_pr_auc, std_pr_auc),lw=2, alpha=.8)
            # Plot standard deviation grey zone of curves
            std_prec = np.std(precs, axis=0)
            precs_upper = np.minimum(mean_prec + std_prec, 1)
            precs_lower = np.maximum(mean_prec - std_prec, 0)
            plt.fill_between(mean_recall, precs_lower, precs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
            #Specify plot axes,labels, and legend
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('Recall (Sensitivity)')
            plt.ylabel('Precision (PPV)')
            plt.legend(loc="upper left", bbox_to_anchor=(1.01,1))
            #Export and/or show plot
            plt.savefig(full_path+'/model_evaluation/'+abbrev[algorithm]+"_PRC.png", bbox_inches="tight")
            if eval(jupyterRun):
                plt.show()
            else:
                plt.close('all')

        #Export and save all CV metric stats for each individual algorithm  -----------------------------------------------------------------------------
        results = {'Balanced Accuracy': s_bac, 'Accuracy': s_ac, 'F1 Score': s_f1, 'Sensitivity (Recall)': s_re, 'Specificity': s_sp,'Precision (PPV)': s_pr, 'TP': s_tp, 'TN': s_tn, 'FP': s_fp, 'FN': s_fn, 'NPV': s_npv, 'LR+': s_lrp, 'LR-': s_lrm, 'ROC AUC': aucs,'PRC AUC': praucs, 'PRC APS': aveprecs}
        dr = pd.DataFrame(results)
        filepath = full_path+'/model_evaluation/'+abbrev[algorithm]+"_performance.csv"
        dr.to_csv(filepath, header=True, index=False)
        metric_dict[algorithm] = results

        #Save Average FI Stats
        save_FI(FI_all, abbrev[algorithm], original_headers, full_path)

        #Store ave metrics for creating global ROC and PRC plots later
        mean_ave_prec = np.mean(aveprecs)
        #result_dict = {'algorithm':algorithm,'fpr':mean_fpr, 'tpr':mean_tpr, 'auc':mean_auc, 'prec':mean_prec, 'pr_auc':mean_pr_auc, 'ave_prec':mean_ave_prec}
        result_dict = {'algorithm':algorithm,'fpr':mean_fpr, 'tpr':mean_tpr, 'auc':mean_auc, 'prec':mean_prec, 'recall':mean_recall, 'pr_auc':mean_pr_auc, 'ave_prec':mean_ave_prec}
        result_table.append(result_dict)
    #Result table later used to create global ROC an PRC plots comparing average ML algorithm performance.
    result_table = pd.DataFrame.from_dict(result_table)
    result_table.set_index('algorithm',inplace=True)
    return result_table,metric_dict

def save_FI(FI_all,algorithm,globalFeatureList,full_path):
    """ Creates directory to store model feature importance results and, for each algorithm, exports a file of feature importance scores from each CV. """
    dr = pd.DataFrame(FI_all)
    if not os.path.exists(full_path+'/model_evaluation/feature_importance/'):
        os.mkdir(full_path+'/model_evaluation/feature_importance/')
    filepath = full_path+'/model_evaluation/feature_importance/'+algorithm+"_FI.csv"
    dr.to_csv(filepath, header=globalFeatureList, index=False)

def doPlotROC(result_table,colors,full_path,jupyterRun):
    """ Generate ROC plot comparing average ML algorithm performance (over all CV training/testing sets)"""
    count = 0
    #Plot curves for each individual ML algorithm
    for i in result_table.index:
        #plt.plot(result_table.loc[i]['fpr'],result_table.loc[i]['tpr'], color=colors[i],label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
        plt.plot(result_table.loc[i]['fpr'],result_table.loc[i]['tpr'], color=colors[i],label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
        count += 1
    # Set figure dimensions
    plt.rcParams["figure.figsize"] = (6,6)
    # Plot no-skill line
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='No-Skill', alpha=.8)
    #Specify plot axes,labels, and legend
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01,1))
    #Export and/or show plot
    plt.savefig(full_path+'/model_evaluation/Summary_ROC.png', bbox_inches="tight")
    if eval(jupyterRun):
        plt.show()
    else:
        plt.close('all')

def doPlotPRC(result_table,colors,full_path,data_name,instance_label,class_label,jupyterRun):
    """ Generate PRC plot comparing average ML algorithm performance (over all CV training/testing sets)"""
    count = 0
    #Plot curves for each individual ML algorithm
    for i in result_table.index:
        plt.plot(result_table.loc[i]['recall'],result_table.loc[i]['prec'], color=colors[i],label="{}, AUC={:.3f}, APS={:.3f}".format(i, result_table.loc[i]['pr_auc'],result_table.loc[i]['ave_prec']))
        count += 1
    #Estimate no skill line based on the fraction of cases found in the first test dataset
    test = pd.read_csv(full_path+'/CVDatasets/'+data_name+'_CV_0_Test.csv')
    if instance_label != 'None':
        test = test.drop(instance_label, axis=1)
    testY = test[class_label].values
    noskill = len(testY[testY == 1]) / len(testY)  # Fraction of cases
    # Plot no-skill line
    plt.plot([0, 1], [noskill, noskill], color='black', linestyle='--',label='No-Skill', alpha=.8)
    #Specify plot axes,labels, and legend
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Recall (Sensitivity)", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("Precision (PPV)", fontsize=15)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01,1))
    #Export and/or show plot
    plt.savefig(full_path+'/model_evaluation/Summary_PRC.png', bbox_inches="tight")
    if eval(jupyterRun):
        plt.show()
    else:
        plt.close('all')

def saveMetricMeans(full_path,metrics,metric_dict):
    """ Exports csv file with average metric values (over all CVs) for each ML modeling algorithm"""
    with open(full_path+'/model_evaluation/Summary_performance_mean.csv',mode='w', newline="") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        e = ['']
        e.extend(metrics)
        writer.writerow(e) #Write headers (balanced accuracy, etc.)
        for algorithm in metric_dict:
            astats = []
            for l in list(metric_dict[algorithm].values()):
                l = [float(i) for i in l]
                meani = mean(l)
                std = stdev(l)
                astats.append(str(meani))
            toAdd = [algorithm]
            toAdd.extend(astats)
            writer.writerow(toAdd)
    file.close()

def saveMetricStd(full_path,metrics,metric_dict):
    """ Exports csv file with metric value standard deviations (over all CVs) for each ML modeling algorithm"""
    with open(full_path + '/model_evaluation/Summary_performance_std.csv', mode='w', newline="") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        e = ['']
        e.extend(metrics)
        writer.writerow(e)  # Write headers (balanced accuracy, etc.)
        for algorithm in metric_dict:
            astats = []
            for l in list(metric_dict[algorithm].values()):
                l = [float(i) for i in l]
                std = stdev(l)
                astats.append(str(std))
            toAdd = [algorithm]
            toAdd.extend(astats)
            writer.writerow(toAdd)
    file.close()

def metricBoxplots(full_path,metrics,algorithms,metric_dict,jupyterRun):
    """ Export boxplots comparing algorithm performance for each standard metric"""
    if not os.path.exists(full_path + '/model_evaluation/metricBoxplots'):
        os.mkdir(full_path + '/model_evaluation/metricBoxplots')
    for metric in metrics:
        tempList = []
        for algorithm in algorithms:
            tempList.append(metric_dict[algorithm][metric])
        td = pd.DataFrame(tempList)
        td = td.transpose()
        td.columns = algorithms
        #Generate boxplot
        boxplot = td.boxplot(column=algorithms,rot=90)
        #Specify plot labels
        plt.ylabel(str(metric))
        plt.xlabel('ML Algorithm')
        #Export and/or show plot
        plt.savefig(full_path + '/model_evaluation/metricBoxplots/Compare_'+metric+'.png', bbox_inches="tight")
        if eval(jupyterRun):
            plt.show()
        else:
            plt.close('all')

def kruskalWallis(full_path,metrics,algorithms,metric_dict,sig_cutoff):
    """ Apply non-parametric Kruskal Wallis one-way ANOVA on ranks. Determines if there is a statistically significant difference in algorithm performance across CV runs.
    Completed for each standard metric separately."""
    # Create directory to store significance testing results (used for both Kruskal Wallis and MannWhitney U-test)
    if not os.path.exists(full_path + '/model_evaluation/statistical_comparisons'):
        os.mkdir(full_path + '/model_evaluation/statistical_comparisons')
    #Create dataframe to store analysis results for each metric
    label = ['Statistic', 'P-Value', 'Sig(*)']
    kruskal_summary = pd.DataFrame(index=metrics, columns=label)
    #Apply Kruskal Wallis test for each metric
    for metric in metrics:
        tempArray = []
        for algorithm in algorithms:
            tempArray.append(metric_dict[algorithm][metric])
        try:
            result = stats.kruskal(*tempArray)
        except:
            result = [tempArray[0],1]
        kruskal_summary.at[metric, 'Statistic'] = str(round(result[0], 6))
        kruskal_summary.at[metric, 'P-Value'] = str(round(result[1], 6))
        if result[1] < sig_cutoff:
            kruskal_summary.at[metric, 'Sig(*)'] = str('*')
        else:
            kruskal_summary.at[metric, 'Sig(*)'] = str('')
    #Export analysis summary to .csv file
    kruskal_summary.to_csv(full_path + '/model_evaluation/statistical_comparisons/KruskalWallis.csv')
    return kruskal_summary

def wilcoxonRank(full_path,metrics,algorithms,metric_dict,kruskal_summary,sig_cutoff):
    """ Apply non-parametric Wilcoxon signed-rank test (pairwise comparisons). If a significant Kruskal Wallis algorithm difference was found for a given metric, Wilcoxon tests individual algorithm pairs
    to determine if there is a statistically significant difference in algorithm performance across CV runs. Test statistic will be zero if all scores from one set are
    larger than the other."""
    for metric in metrics:
        if kruskal_summary['Sig(*)'][metric] == '*':
            wilcoxon_stats = []
            done = []
            for algorithm1 in algorithms:
                for algorithm2 in algorithms:
                    if not [algorithm1,algorithm2] in done and not [algorithm2,algorithm1] in done and algorithm1 != algorithm2:
                        set1 = metric_dict[algorithm1][metric]
                        set2 = metric_dict[algorithm2][metric]
                        #handle error when metric values are equal for both algorithms
                        if set1 == set2:  # Check if all nums are equal in sets
                            report = ['NA',1]
                        else: # Apply Wilcoxon Rank Sum test
                            report = stats.wilcoxon(set1,set2)
                        #Summarize test information in list
                        tempstats = [algorithm1,algorithm2,report[0],report[1],'']
                        if report[1] < sig_cutoff:
                            tempstats[4] = '*'
                        wilcoxon_stats.append(tempstats)
                        done.append([algorithm1,algorithm2])
            #Export test results
            wilcoxon_stats_df = pd.DataFrame(wilcoxon_stats)
            wilcoxon_stats_df.columns = ['Algorithm 1', 'Algorithm 2', 'Statistic', 'P-Value', 'Sig(*)']
            wilcoxon_stats_df.to_csv(full_path + '/model_evaluation/statistical_comparisons/WilcoxonRank_'+metric+'.csv', index=False)

def mannWhitneyU(full_path,metrics,algorithms,metric_dict,kruskal_summary,sig_cutoff):
    """ Apply non-parametric Mann Whitney U-test (pairwise comparisons). If a significant Kruskal Wallis algorithm difference was found for a given metric, Mann Whitney tests individual algorithm pairs
    to determine if there is a statistically significant difference in algorithm performance across CV runs. Test statistic will be zero if all scores from one set are
    larger than the other."""
    for metric in metrics:
        if kruskal_summary['Sig(*)'][metric] == '*':
            mann_stats = []
            done = []
            for algorithm1 in algorithms:
                for algorithm2 in algorithms:
                    if not [algorithm1,algorithm2] in done and not [algorithm2,algorithm1] in done and algorithm1 != algorithm2:
                        set1 = metric_dict[algorithm1][metric]
                        set2 = metric_dict[algorithm2][metric]
                        if set1 == set2:  # Check if all nums are equal in sets
                            report = ['NA',1]
                        else: #Apply Mann Whitney U test
                            report = stats.mannwhitneyu(set1,set2)
                        #Summarize test information in list
                        tempstats = [algorithm1,algorithm2,report[0],report[1],'']
                        if report[1] < sig_cutoff:
                            tempstats[4] = '*'
                        mann_stats.append(tempstats)
                        done.append([algorithm1,algorithm2])
            #Export test results
            mann_stats_df = pd.DataFrame(mann_stats)
            mann_stats_df.columns = ['Algorithm 1', 'Algorithm 2', 'Statistic', 'P-Value', 'Sig(*)']
            mann_stats_df.to_csv(full_path + '/model_evaluation/statistical_comparisons/MannWhitneyU_'+metric+'.csv', index=False)


def prepFI(algorithms,full_path,abbrev,metric_dict,primary_metric):
    """ Organizes and prepares model feature importance data for boxplot and composite feature importance figure generation."""
    #Initialize required lists
    fi_df_list = []         # algorithm feature importance dataframe list (used to generate FI boxplots for each algorithm)
    fi_ave_list = []        # algorithm feature importance averages list (used to generate composite FI barplots)
    ave_metric_list = []    # algorithm focus metric averages list (used in weighted FI viz)
    all_feature_list = []   # list of pre-feature selection feature names as they appear in FI reports for each algorithm
    #Get necessary feature importance data and primary metric data (currenly only 'balanced accuracy' can be used for this)
    for algorithm in algorithms:
        # Get relevant feature importance info
        temp_df = pd.read_csv(full_path+'/model_evaluation/feature_importance/'+abbrev[algorithm]+"_FI.csv") #CV FI scores for all original features in dataset.
        if algorithm == algorithms[0]:  # Should be same for all algorithm files (i.e. all original features in standard CV dataset order)
            all_feature_list = temp_df.columns.tolist()
        fi_df_list.append(temp_df)
        fi_ave_list.append(temp_df.mean().tolist()) #Saves average FI scores over CV runs
        # Get relevant metric info
        avgBA = mean(metric_dict[algorithm][primary_metric])
        ave_metric_list.append(avgBA)
    #Normalize Average Feature importance scores so they fall between (0 - 1)
    fi_ave_norm_list = []
    for each in fi_ave_list:  # each algorithm
        normList = []
        for i in range(len(each)): #each feature (score) in original data order
            if each[i] <= 0: #Feature importance scores assumed to be uninformative if at or below 0
                normList.append(0)
            else:
                normList.append((each[i]) / (max(each)))
        fi_ave_norm_list.append(normList)
    #Identify features with non-zero averages (step towards excluding features that had zero feature importance for all algorithms)
    alg_non_zero_FI_list = [] #stores list of feature name lists that are non-zero for each algorithm
    for each in fi_ave_list:  # each algorithm
        temp_non_zero_list = []
        for i in range(len(each)):  # each feature
            if each[i] > 0.0:
                temp_non_zero_list.append(all_feature_list[i]) #add feature names with positive values (doesn't need to be normalized for this)
        alg_non_zero_FI_list.append(temp_non_zero_list)
    non_zero_union_features = alg_non_zero_FI_list[0]  # grab first algorithm's list
    #Identify union of features with non-zero averages over all algorithms (i.e. if any algorithm found a non-zero score it will be considered for inclusion in top feature visualizations)
    for j in range(1, len(algorithms)):
        non_zero_union_features = list(set(non_zero_union_features) | set(alg_non_zero_FI_list[j]))
    non_zero_union_indexes = []
    for i in non_zero_union_features:
        non_zero_union_indexes.append(all_feature_list.index(i))
    return fi_df_list,fi_ave_list,fi_ave_norm_list,ave_metric_list,all_feature_list,non_zero_union_features,non_zero_union_indexes

def selectForCompositeViz(top_model_features,non_zero_union_features,non_zero_union_indexes,algorithms,ave_metric_list,fi_ave_norm_list):
    """ Identify list of top features over all algorithms to visualize (note that best features to vizualize are chosen using algorithm performance weighting and normalization:
    frac plays no useful role here only for viz). All features included if there are fewer than 'top_model_features'. Top features are determined by the sum of performance
    (i.e. balanced accuracy) weighted feature importances over all algorithms."""
    featuresToViz = None
    #Create performance weighted score sum dictionary for all features
    scoreSumDict = {}
    i = 0
    for each in non_zero_union_features:  # for each non-zero feature
        for j in range(len(algorithms)):  # for each algorithm
            # grab target score from each algorithm
            score = fi_ave_norm_list[j][non_zero_union_indexes[i]]
            # multiply score by algorithm performance weight
            weight = ave_metric_list[j]
            if weight <= .5:
                weight = 0
            if not weight == 0:
                weight = (weight - 0.5) / 0.5
            score = score * weight
            #score = score * ave_metric_list[j]
            if not each in scoreSumDict:
                scoreSumDict[each] = score
            else:
                scoreSumDict[each] += score
        i += 1
    # Sort features by decreasing score
    scoreSumDict_features = sorted(scoreSumDict, key=lambda x: scoreSumDict[x], reverse=True)
    if len(non_zero_union_features) > top_model_features: #Keep all features if there are fewer than specified top results
        featuresToViz = scoreSumDict_features[0:top_model_features]
    else:
        featuresToViz = scoreSumDict_features
    return featuresToViz #list of feature names to vizualize in composite FI plots.

def doFIBoxplots(full_path,fi_df_list,fi_ave_list,algorithms,original_headers,top_model_features, jupyterRun):
    """ Generate individual feature importance boxplots for each algorithm """
    algorithmCounter = 0
    for algorithm in algorithms: #each algorithms
        #Make average feature importance score dicitonary
        scoreDict = {}
        counter = 0
        for ave_score in fi_ave_list[algorithmCounter]: #each feature
            scoreDict[original_headers[counter]] = ave_score
            counter += 1
        # Sort features by decreasing score
        scoreDict_features = sorted(scoreDict, key=lambda x: scoreDict[x], reverse=True)
        #Make list of feature names to vizualize
        if len(original_headers) > top_model_features:
            featuresToViz = scoreDict_features[0:top_model_features]
        else:
            featuresToViz = scoreDict_features
        # FI score dataframe for current algorithm
        df = fi_df_list[algorithmCounter]
        # Subset of dataframe (in ranked order) to vizualize
        viz_df = df[featuresToViz]
        #Generate Boxplot
        fig = plt.figure(figsize=(15, 4))
        boxplot = viz_df.boxplot(rot=90)
        plt.title(algorithm)
        plt.ylabel('Feature Importance Score')
        plt.xlabel('Features')
        plt.xticks(np.arange(1, len(featuresToViz) + 1), featuresToViz, rotation='vertical')
        plt.savefig(full_path+'/model_evaluation/feature_importance/' + algorithm + '_boxplot',bbox_inches="tight")
        if eval(jupyterRun):
            plt.show()
        else:
            plt.close('all')    #Identify and sort (decreaseing) features with top average FI
        algorithmCounter += 1

def doFI_Histogram(full_path, fi_ave_list, algorithms, jupyterRun):
    """ Generate histogram showing distribution of average feature importances scores for each algorithm. """
    algorithmCounter = 0
    for algorithm in algorithms: #each algorithms
        aveScores = fi_ave_list[algorithmCounter]
        #Plot a histogram of average feature importance
        plt.hist(aveScores,bins=100)
        plt.xlabel("Average Feature Importance")
        plt.ylabel("Frequency")
        plt.title("Histogram of Average Feature Importance for "+str(algorithm))
        plt.xticks(rotation = 'vertical')
        plt.savefig(full_path+'/model_evaluation/feature_importance/' + algorithm + '_histogram',bbox_inches="tight")
        if eval(jupyterRun):
            plt.show()
        else:
            plt.close('all')

def getFI_To_Viz_Sorted(featuresToViz,all_feature_list,algorithms,fi_ave_norm_list):
    """ Takes a list of top features names for vizualization, gets their indexes. In every composite FI plot features are ordered the same way
    they are selected for vizualization (i.e. normalized and performance weighted). Because of this feature bars are only perfectly ordered in
    descending order for the normalized + performance weighted composite plot. """
    #Get original feature indexs for selected feature names
    feature_indexToViz = [] #indexes of top features
    for i in featuresToViz:
        feature_indexToViz.append(all_feature_list.index(i))
    # Create list of top feature importance values in original dataset feature order
    top_fi_ave_norm_list = [] #feature importance values of top features for each algorithm (list of lists)
    for i in range(len(algorithms)):
        tempList = []
        for j in feature_indexToViz: #each top feature index
            tempList.append(fi_ave_norm_list[i][j]) #add corresponding FI value
        top_fi_ave_norm_list.append(tempList)
    all_feature_listToViz = featuresToViz
    return top_fi_ave_norm_list,all_feature_listToViz

def composite_FI_plot(fi_list, algorithms, algColors, all_feature_listToViz, figName,full_path,jupyterRun,yLabelText):
    """ Generate composite feature importance plot given list of feature names and associated feature importance scores for each algorithm.
    This is run for different transformations of the normalized feature importance scores. """
    # Set basic plot properites
    rc('font', weight='bold', size=16)
    # The position of the bars on the x-axis
    r = all_feature_listToViz #feature names
    #Set width of bars
    barWidth = 0.75
    #Set figure dimensions
    plt.figure(figsize=(24, 12))
    #Plot first algorithm FI scores (lowest) bar
    p1 = plt.bar(r, fi_list[0], color=algColors[0], edgecolor='white', width=barWidth)
    #Automatically calculate space needed to plot next bar on top of the one before it
    bottoms = [] #list of space used by previous algorithms for each feature (so next bar can be placed directly above it)
    for i in range(len(algorithms) - 1):
        for j in range(i + 1):
            if j == 0:
                bottom = np.array(fi_list[0])
            else:
                bottom += np.array(fi_list[j])
        bottoms.append(bottom)
    if not isinstance(bottoms, list):
        bottoms = bottoms.tolist()
    #Plot subsequent feature bars for each subsequent algorithm
    ps = [p1[0]]
    for i in range(len(algorithms) - 1):
        p = plt.bar(r, fi_list[i + 1], bottom=bottoms[i], color=algColors[i + 1], edgecolor='white', width=barWidth)
        ps.append(p[0])
    lines = tuple(ps)
    # Specify axes info and legend
    plt.xticks(np.arange(len(all_feature_listToViz)), all_feature_listToViz, rotation='vertical')
    plt.xlabel("Feature", fontsize=20)
    plt.ylabel(yLabelText, fontsize=20)
    #plt.legend(lines[::-1], algorithms[::-1],loc="upper left", bbox_to_anchor=(1.01,1)) #legend outside plot
    plt.legend(lines[::-1], algorithms[::-1],loc="upper right")
    #Export and/or show plot
    plt.savefig(full_path+'/model_evaluation/feature_importance/Compare_FI_' + figName + '.png', bbox_inches='tight')
    if eval(jupyterRun):
        plt.show()
    else:
        plt.close('all')

def fracFI(top_fi_ave_norm_list):
    """ Transforms feature scores so that they sum to 1 over all features for a given algorithm.  This way the normalized and fracionated composit bar plot
    offers equal total bar area for every algorithm. The intuition here is that if an algorithm gives the same FI scores for all top features it won't be
    overly represented in the resulting plot (i.e. all features can have the same maximum feature importance which might lead to the impression that an
    algorithm is working better than it is.) Instead, that maximum 'bar-real-estate' has to be divided by the total number of features. Notably, this
    transformation has the potential to alter total algorithm FI bar height ranking of features. """
    fracLists = []
    for each in top_fi_ave_norm_list: #each algorithm
        fracList = []
        for i in range(len(each)): #each feature
            if sum(each) == 0: #check that all feature scores are not zero to avoid zero division error
                fracList.append(0)
            else:
                fracList.append((each[i] / (sum(each))))
        fracLists.append(fracList)
    return fracLists

def weightFI(ave_metric_list,top_fi_ave_norm_list):
    """ Weights the feature importance scores by algorithm performance (intuitive because when interpreting feature importances we want to place more weight on better performing algorithms) """
    # Prepare weights
    weights = []
    # replace all balanced accuraces <=.5 with 0 (i.e. these are no better than random chance)
    for i in range(len(ave_metric_list)):
        if ave_metric_list[i] <= .5:
            ave_metric_list[i] = 0
    # normalize balanced accuracies
    for i in range(len(ave_metric_list)):
        if ave_metric_list[i] == 0:
            weights.append(0)
        else:
            weights.append((ave_metric_list[i] - 0.5) / 0.5)
    # Weight normalized feature importances
    weightedLists = []
    for i in range(len(top_fi_ave_norm_list)): #each algorithm
        weightList = np.multiply(weights[i], top_fi_ave_norm_list[i]).tolist()
        weightedLists.append(weightList)
    return weightedLists,weights

def weightFracFI(fracLists,weights):
    """ Weight normalized and fractionated feature importances. """
    weightedFracLists = []
    for i in range(len(fracLists)):
        weightList = np.multiply(weights[i], fracLists[i]).tolist()
        weightedFracLists.append(weightList)
    return weightedFracLists

def saveRuntime(full_path,job_start_time):
    """ Save phase runtime """
    runtime_file = open(full_path + '/runtime/runtime_Stats.txt', 'w')
    runtime_file.write(str(time.time() - job_start_time))
    runtime_file.close()

def parseRuntime(full_path,algorithms,abbrev):
    """ Loads runtime summaries from entire pipeline and parses them into a single summary file."""
    dict = {}
    for file_path in glob.glob(full_path+'/runtime/*.txt'):
        f = open(file_path,'r')
        val = float(f.readline())
        ref = file_path.split('/')[-1].split('_')[1].split('.')[0]
        if ref in abbrev:
            ref = abbrev[ref]
        if not ref in dict:
            dict[ref] = val
        else:
            dict[ref] += val
    with open(full_path+'/runtimes.csv',mode='w', newline="") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Pipeline Component","Time (sec)"])
        writer.writerow(["Exploratory Analysis",dict['exploratory']])
        writer.writerow(["Preprocessing",dict['preprocessing']])
        try:
            writer.writerow(["Mutual Information",dict['mutualinformation']])
        except:
            pass
        try:
            writer.writerow(["MultiSURF",dict['multisurf']])
        except:
            pass
        writer.writerow(["Feature Selection",dict['featureselection']])
        for algorithm in algorithms: #Repoert runtimes for each algorithm
            writer.writerow(([algorithm,dict[abbrev[algorithm]]]))
        writer.writerow(["Stats Summary",dict['Stats']])

if __name__ == '__main__':
    job(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],int(sys.argv[7]),sys.argv[8],sys.argv[9],sys.argv[10],int(sys.argv[11]),float(sys.argv[12]),sys.argv[13])
