"""
File: StatsMain.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 6 of STREAMLINE - This 'Main' script manages Phase 6 run parameters, and submits job to run locally (to run serially) or on a linux computing cluster (parallelized).
             This script runs StatsJob.py which (for a single orginal target dataset) creates summaries of ML classification evaluation statistics (means and standard deviations),
             ROC and PRC plots (comparing CV performance in the same ML algorithm and comparing average performance between ML algorithms), model feature importance averages over CV runs,
             boxplots comparing ML algorithms for each metric, Kruskal Wallis and Mann Whitney statistical comparsions between ML algorithms, model feature importance boxplots for each
             algorithm, and composite feature importance plots summarizing model feature importance across all ML algorithms. This script is run on all cv results for a given original
             target dataset from Phase 1. All 'Main' scripts in this pipeline have the potential to be extended by users to submit jobs to other parallel computing frameworks (e.g. cloud computing).
Warnings: Designed to be run following the completion of STREAMLINE Phase 5 (ModelMain.py).
Sample Run Command (Linux cluster parallelized with all default run parameters):
    python StatsMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1
Sample Run Command (Local/serial with with all default run parameters):
    python StatsMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1 --run-parallel False
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import argparse
import os
import sys
import time
import pandas as pd
import StatsJob
import glob
import pickle

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description='')
    #No defaults
    parser.add_argument('--out-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--exp-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    #Defaults available
    parser.add_argument('--plot-ROC', dest='plot_ROC', type=str,help='Plot ROC curves individually for each algorithm including all CV results and averages', default='True')
    parser.add_argument('--plot-PRC', dest='plot_PRC', type=str,help='Plot PRC curves individually for each algorithm including all CV results and averages', default='True')
    parser.add_argument('--plot-box', dest='plot_metric_boxplots', type=str,help='Plot box plot summaries comparing algorithms for each metric', default='True')
    parser.add_argument('--plot-FI_box', dest='plot_FI_box', type=str,help='Plot feature importance boxplots and histograms for each algorithm', default='True')
    parser.add_argument('--metric-weight', dest='metric_weight',type=str,help='ML model metric used as weight in composite FI plots (only supports balanced_accuracy or roc_auc as options) Recommend setting the same as primary_metric if possible.',default='balanced_accuracy')
    parser.add_argument('--top-features', dest='top_model_features', type=int,help='number of top features to illustrate in figures', default=40)
    #Lostistical arguments
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='if run parallel on LSF compatible computing cluster',default="True")
    parser.add_argument('--queue',dest='queue',type=str,help='specify name of parallel computing queue (uses our research groups queue by default)',default="i2c2_normal")
    parser.add_argument('--res-mem', dest='reserved_memory', type=int, help='reserved memory for the job (in Gigabytes)',default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int, help='maximum memory before the job is automatically terminated',default=15)
    parser.add_argument('-c','--do-check',dest='do_check', help='Boolean: Specify whether to check for existence of all output files.', action='store_true')

    options = parser.parse_args(argv[1:])
    job_counter = 0

    # Argument checks
    if not os.path.exists(options.output_path):
        raise Exception("Output path must exist (from phase 1) before phase 6 can begin")
    if not os.path.exists(options.output_path + '/' + options.experiment_name):
        raise Exception("Experiment must exist (from phase 1) before phase 6 can begin")

    #Unpickle metadata from previous phase
    file = open(options.output_path+'/'+options.experiment_name+'/'+"metadata.pickle", 'rb')
    metadata = pickle.load(file)
    file.close()
    #Load variables specified earlier in the pipeline from metadata
    class_label = metadata['Class Label']
    instance_label = metadata['Instance Label']
    sig_cutoff = metadata['Statistical Significance Cutoff']
    cv_partitions = int(metadata['CV Partitions'])
    scale_data = metadata['Use Data Scaling']
    do_DT = metadata['Decision Tree']
    do_GP = metadata['Genetic Programming']
    jupyterRun = metadata['Run From Jupyter Notebook']
    primary_metric = metadata['Primary Metric'] # 0 list position is the True/False for algorithm use

    if not options.do_check: #Run job submission
        # Iterate through datasets
        dataset_paths = os.listdir(options.output_path + "/" + options.experiment_name)
        removeList = ['metadata.pickle','metadata.csv','algInfo.pickle','jobsCompleted','logs','jobs','DatasetComparisons',options.experiment_name+'_ML_Pipeline_Report.pdf']
        for text in removeList:
            if text in dataset_paths:
                dataset_paths.remove(text)

        for dataset_directory_path in dataset_paths:
            full_path = options.output_path + "/" + options.experiment_name + "/" + dataset_directory_path

            #Create folders for DT and GP vizualizations
            if eval(do_DT) and not os.path.exists(full_path+'/model_evaluation/DT_Viz'):
                os.mkdir(full_path+'/model_evaluation/DT_Viz')
            if eval(do_GP) and not os.path.exists(full_path+'/model_evaluation/GP_Viz'):
                os.mkdir(full_path+'/model_evaluation/GP_Viz')
            job_counter += 1
            if eval(options.run_parallel):
                submitClusterJob(full_path,options.plot_ROC,options.plot_PRC,options.plot_FI_box,class_label,instance_label,options.output_path+'/'+options.experiment_name,cv_partitions,scale_data,options.reserved_memory,options.maximum_memory,options.queue,options.plot_metric_boxplots,primary_metric,options.top_model_features,sig_cutoff,options.metric_weight,jupyterRun)
            else:
                submitLocalJob(full_path,options.plot_ROC,options.plot_PRC,options.plot_FI_box,class_label,instance_label,cv_partitions,scale_data,options.plot_metric_boxplots,primary_metric,options.top_model_features,sig_cutoff,options.metric_weight,jupyterRun)

        metadata['Export ROC Plot'] = options.plot_ROC
        metadata['Export PRC Plot'] = options.plot_PRC
        metadata['Export Metric Boxplots'] = options.plot_metric_boxplots
        metadata['Export Feature Importance Boxplots'] = options.plot_FI_box
        metadata['Metric Weighting Composite FI Plots'] = options.metric_weight
        metadata['Top Model Features To Display'] = options.top_model_features

        #Pickle the metadata for future use
        pickle_out = open(options.output_path+'/'+options.experiment_name+'/'+"metadata.pickle", 'wb')
        pickle.dump(metadata,pickle_out)
        pickle_out.close()

        #Now that primary pipeline phases are complete generate a human readable version of metadata
        df = pd.DataFrame.from_dict(metadata, orient ='index')
        df.to_csv(options.output_path+'/'+options.experiment_name+'/'+'metadata.csv',index=True)

    else: #run job completion checks
        datasets = os.listdir(options.output_path + "/" + options.experiment_name)
        datasets.remove('logs')
        datasets.remove('jobs')
        datasets.remove('jobsCompleted')
        if 'metadata.pickle' in datasets:
            datasets.remove('metadata.pickle')
        if 'algInfo.pickle' in datasets:
            datasets.remove('algInfo.pickle')
        if 'DatasetComparisons' in datasets:
            datasets.remove('DatasetComparisons')
        if 'metadata.csv' in datasets:
            datasets.remove('metadata.csv')

        phase6Jobs = []
        for dataset in datasets:
            phase6Jobs.append('job_stats_'+dataset+'.txt')

        for filename in glob.glob(options.output_path + "/" + options.experiment_name+'/jobsCompleted/job_stats*'):
            ref = filename.split('/')[-1]
            phase6Jobs.remove(ref)
        for job in phase6Jobs:
            print(job)
        if len(phase6Jobs) == 0:
            print("All Phase 6 Jobs Completed")
        else:
            print("Above Phase 6 Jobs Not Completed")

    if not options.do_check:
        print(str(job_counter)+ " jobs submitted in Phase 6")

def submitLocalJob(full_path,plot_ROC,plot_PRC,plot_FI_box,class_label,instance_label,cv_partitions,scale_data,plot_metric_boxplots,primary_metric,top_model_features,sig_cutoff,metric_weight,jupyterRun):
    """ Runs StatsJob.py locally, once for each of the original target datasets (all CV datasets analyzed at once). These runs will be completed serially rather than in parallel. """
    StatsJob.job(full_path,plot_ROC,plot_PRC,plot_FI_box,class_label,instance_label,cv_partitions,scale_data,plot_metric_boxplots,primary_metric,top_model_features,sig_cutoff,metric_weight,jupyterRun)

def submitClusterJob(full_path,plot_ROC,plot_PRC,plot_FI_box,class_label,instance_label,experiment_path,cv_partitions,scale_data,reserved_memory,maximum_memory,queue,plot_metric_boxplots,primary_metric,top_model_features,sig_cutoff,metric_weight,jupyterRun):
    """ Runs StatsJob.py once for each of the original target datasets (all CV datasets analyzed at once). Runs in parallel on a linux-based computing cluster that uses an IBM Spectrum LSF for job scheduling."""
    job_ref = str(time.time())
    job_name = experiment_path + '/jobs/P6_' + job_ref + '_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q '+queue+'\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('#BSUB -R "rusage[mem='+str(reserved_memory)+'G]"'+'\n')
    sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/P6_'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/P6_'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python '+this_file_path+'/StatsJob.py '+full_path+" "+plot_ROC+" "+plot_PRC+" "+plot_FI_box+" "+class_label+" "+instance_label+" "+str(cv_partitions)+" "+scale_data+" "+str(plot_metric_boxplots)+" "+str(primary_metric)+" "+str(top_model_features)+" "+str(sig_cutoff)+" "+str(metric_weight)+" "+str(jupyterRun)+'\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))
