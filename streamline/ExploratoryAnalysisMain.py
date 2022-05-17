"""
File: ExploratoryAnalysisMain.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 1 of STREAMLINE - This 'Main' script manages Phase 1 run parameters, updates the metadata file (with user specified run parameters across pipeline run)
             and submits job to run locally (to run serially) or on a linux computing cluster (parallelized).  This script runs ExploratoryAnalysisJob.py which conducts initial
             exploratory analysis of data and cross validation (CV) partitioning. Note that this entire pipeline may also be run within Jupyter Notebook (see STREAMLINE-Notebook.ipynb).
             All 'Main' scripts in this pipeline have the potential to be extended by users to submit jobs to other parallel computing frameworks (e.g. cloud computing).
Warnings:
    - Before running, be sure to check that all run parameters have relevant/desired values including those with default values available.
    - 'Target' datasets for analysis should be in comma-separated format (.txt or .csv)
    - Missing data values should be empty or indicated with an 'NA'.
    - Dataset(s) includes a header giving column labels.
    - Data columns include features, class label, and optionally instance (i.e. row) labels, or match labels (if matched cross validation will be used)
    - Binary class values are encoded as 0 (e.g. negative), and 1 (positive) with respect to true positive, true negative, false positive, false negative metrics. PRC plots focus on classification of 'positives'.
    - All feature values (both categorical and quantitative) are numerically encoded. Scikit-learn does not accept text-based values. However both instance_label and match_label values may be either numeric or text.
    - One or more target datasets for analysis should be included in the same data_path folder. The path to this folder is a critical pipeline run parameter. No spaces are allowed in filenames (this will lead to
      'invalid literal' by export_exploratory_analysis. If multiple datasets are being analyzed they must have the same class_label, and (if present) the same instance_label and match_label.

Sample Run Command (Linux cluster parallelized with all default run parameters):
    python ExploratoryAnalysisMain.py --data-path /Users/robert/Desktop/Datasets --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1

Sample Run Command (Local/serial with with all default run parameters):
    python ExploratoryAnalysisMain.py --data-path /Users/robert/Desktop/Datasets --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1 --run-parallel False
"""
#Import required packages ---------------------------------------------------------------------------------------------------------------------------
import sys
import os
import argparse
import glob
import ExploratoryAnalysisJob
import time
import csv
import pickle

def main(argv):
    #Parse arguments ---------------------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="")
    #Arguments with no defaults
    parser.add_argument('--data-path',dest='data_path',type=str,help='path to directory containing datasets')
    parser.add_argument('--out-path',dest='output_path',type=str,help='path to output directory')
    parser.add_argument('--exp-name', dest='experiment_name',type=str, help='name of experiment output folder (no spaces)')
    #Arguments with defaults available (but critical to check)
    parser.add_argument('--class-label', dest='class_label', type=str, help='outcome label of all datasets', default="Class")
    parser.add_argument('--inst-label', dest='instance_label', type=str, help='instance label of all datasets (if present)', default="None")
    parser.add_argument('--fi', dest='ignore_features_path',type=str, help='path to .csv file with feature labels to be ignored in analysis (e.g. /home/ryanurb/code/STREAMLINE/droppedFeatures.csv))', default="None")
    parser.add_argument('--cf', dest='categorical_feature_path',type=str, help='path to .csv file with feature labels specified to be treated as categorical where possible', default="None")
    #Arguments with defaults available (but less critical to check)
    parser.add_argument('--cv',dest='cv_partitions',type=int,help='number of CV partitions',default=10)
    parser.add_argument('--part',dest='partition_method',type=str,help="'S', or 'R', or 'M', for stratified, random, or matched, respectively",default="S")
    parser.add_argument('--match-label', dest='match_label', type=str, help='only applies when M selected for partition-method; indicates column with matched instance ids', default="None")
    parser.add_argument('--cat-cutoff', dest='categorical_cutoff', type=int,help='number of unique values after which a variable is considered to be quantitative vs categorical', default=10)
    parser.add_argument('--sig', dest='sig_cutoff', type=float, help='significance cutoff used throughout pipeline',default=0.05)
    parser.add_argument('--export-fc', dest='export_feature_correlations', type=str, help='run and export feature correlation analysis (yields correlation heatmap)',default="True")
    parser.add_argument('--export-up', dest='export_univariate_plots', type=str, help='export univariate analysis plots (note: univariate analysis still output by default)',default="False")
    parser.add_argument('--rand-state', dest='random_state', type=int, help='"Dont Panic" - sets a specific random seed for reproducible results',default=42)
    #Lostistical arguments
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='if run parallel on LSF compatible computing cluster',default="True")
    parser.add_argument('--queue',dest='queue',type=str,help='specify name of parallel computing queue (uses our research groups queue by default)',default="i2c2_normal")
    parser.add_argument('--res-mem', dest='reserved_memory', type=int, help='reserved memory for the job (in Gigabytes)',default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int, help='maximum memory before the job is automatically terminated',default=15)
    parser.add_argument('-c','--do-check',dest='do_check', help='Boolean: Specify whether to check for existence of all output files.', action='store_true')

    options = parser.parse_args(argv[1:])
    jupyterRun = 'False' #controls whether plots are shown or closed depending on whether jupyter notebook is used to run code or not
    job_counter = 0

    # Job submission ----------------------------------------------------------------------------------------------------------------------------------
    if not options.do_check: #Run job file
        makeDirTree(options.data_path,options.output_path,options.experiment_name,jupyterRun) #check file/path names and create directory tree for output
        #Determine file extension of datasets in target folder
        file_count = 0
        unique_datanames = []
        for dataset_path in glob.glob(options.data_path+'/*'):
            file_extension = dataset_path.split('/')[-1].split('.')[-1]
            data_name = dataset_path.split('/')[-1].split('.')[0] #Save unique dataset names so that analysis is run only once if there is both a .txt and .csv version of dataset with same name.
            if file_extension == 'txt' or file_extension == 'csv':
                if data_name not in unique_datanames:
                    unique_datanames.append(data_name)
                    job_counter += 1
                    if eval(options.run_parallel): #Run as job in parallel on linux computing cluster
                        submitClusterJob(dataset_path,options.output_path +'/'+options.experiment_name,options.cv_partitions,options.partition_method,options.categorical_cutoff,options.export_feature_correlations,options.export_univariate_plots,options.class_label,options.instance_label,options.match_label,options.random_state,options.reserved_memory,options.maximum_memory,options.queue,options.ignore_features_path,options.categorical_feature_path,options.sig_cutoff,jupyterRun)
                    else: #Run job locally, serially
                        submitLocalJob(dataset_path,options.output_path+'/'+options.experiment_name,options.cv_partitions,options.partition_method,options.categorical_cutoff,options.export_feature_correlations,options.export_univariate_plots,options.class_label,options.instance_label,options.match_label,options.random_state,options.ignore_features_path,options.categorical_feature_path,options.sig_cutoff,jupyterRun)
                    file_count += 1
        if file_count == 0: #Check that there was at least 1 dataset
            raise Exception("There must be at least one .txt or .csv dataset in data_path directory")

        #Create metadata dictionary object to keep track of pipeline run paramaters throughout phases
        metadata = {}
        metadata['Data Path'] = options.data_path
        metadata['Output Path'] = options.output_path
        metadata['Experiment Name'] = options.experiment_name
        metadata['Class Label'] = options.class_label
        metadata['Instance Label'] = options.instance_label
        metadata['Ignored Features'] = options.ignore_features_path
        metadata['Specified Categorical Features'] = options.categorical_feature_path
        metadata['CV Partitions'] = options.cv_partitions
        metadata['Partition Method'] = options.partition_method
        metadata['Match Label'] = options.match_label
        metadata['Categorical Cutoff'] = options.categorical_cutoff
        metadata['Statistical Significance Cutoff'] = options.sig_cutoff
        metadata['Export Feature Correlations'] = options.export_feature_correlations
        metadata['Export Univariate Plots'] = options.export_univariate_plots
        metadata['Random Seed'] = options.random_state
        metadata['Run From Jupyter Notebook'] = jupyterRun
        #Pickle the metadata for future use
        pickle_out = open(options.output_path+'/'+options.experiment_name+'/'+"metadata.pickle", 'wb')
        pickle.dump(metadata,pickle_out)
        pickle_out.close()

    else: #Instead of running job, checks whether previously run jobs were successfully completed
        datasets = os.listdir(options.output_path + "/" + options.experiment_name)
        datasets.remove('logs')
        datasets.remove('jobs')
        datasets.remove('jobsCompleted')
        if 'metadata.pickle' in datasets:
            datasets.remove('metadata.pickle')
        if 'DatasetComparisons' in datasets:
            datasets.remove('DatasetComparisons')
        phase1Jobs = []
        for dataset in datasets:
            phase1Jobs.append('job_exploratory_'+dataset+'.txt')
        for filename in glob.glob(options.output_path + "/" + options.experiment_name+'/jobsCompleted/job_exploratory*'):
            ref = filename.split('/')[-1]
            phase1Jobs.remove(ref)
        for job in phase1Jobs:
            print(job)
        if len(phase1Jobs) == 0:
            print("All Phase 1 Jobs Completed")
        else:
            print("Above Phase 1 Jobs Not Completed")

    if not options.do_check:
        print(str(job_counter)+ " jobs submitted in Phase 1")

def makeDirTree(data_path,output_path,experiment_name,jupyterRun):
    """ Checks existence of data folder path. Checks that experiment output folder does not already exist as well as validity of experiment_name parameter.
    Then generates initial output folder hierarchy. """
    #Check to make sure data_path exists and experiment name is valid & unique
    if not os.path.exists(data_path):
        raise Exception("Provided data_path does not exist")
    if os.path.exists(output_path+'/'+experiment_name):
        raise Exception("Error: A folder with the specified experiment name already exists at "+output_path+'/'+experiment_name+'. This path/folder name must be unique.')
    for char in experiment_name:
        if not char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_':
            raise Exception('Experiment Name must be alphanumeric')
    #Create output folder if it doesn't already exist
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    #Create Experiment folder, with log and job folders
    os.mkdir(output_path+'/'+experiment_name)
    os.mkdir(output_path+'/'+ experiment_name+'/jobsCompleted')
    if not eval(jupyterRun):
        os.mkdir(output_path+'/'+experiment_name+'/jobs')
        os.mkdir(output_path+'/'+experiment_name+'/logs')

def submitLocalJob(dataset_path,experiment_path,cv_partitions,partition_method,categorical_cutoff,export_feature_correlations,export_univariate_plots,class_label,instance_label,match_label,random_state,ignore_features_path,categorical_feature_path,sig_cutoff,jupyterRun):
    """ Runs ExploratoryAnalysisJob.py on each dataset in dataset_path locally. These runs will be completed serially rather than in parallel. """
    ExploratoryAnalysisJob.job(dataset_path,experiment_path,cv_partitions,partition_method,categorical_cutoff,export_feature_correlations,export_univariate_plots,class_label,instance_label,match_label,random_state,ignore_features_path,categorical_feature_path,sig_cutoff,jupyterRun)

def submitClusterJob(dataset_path,experiment_path,cv_partitions,partition_method,categorical_cutoff,export_feature_correlations,export_univariate_plots,class_label,instance_label,match_label,random_state,reserved_memory,maximum_memory,queue,ignore_features_path,categorical_feature_path,sig_cutoff,jupyterRun):
    """ Runs ExploratoryAnalysisJob.py on each dataset in dataset_path in parallel on a linux-based computing cluster that uses an IBM Spectrum LSF for job scheduling."""
    job_ref = str(time.time())
    job_name = experiment_path+'/jobs/P1_'+job_ref+'_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q '+queue+'\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('#BSUB -R "rusage[mem='+str(reserved_memory)+'G]"'+'\n')
    sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/P1_'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/P1_'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python '+this_file_path+'/ExploratoryAnalysisJob.py '+dataset_path+" "+experiment_path+" "+str(cv_partitions)+" "+partition_method+" "+str(categorical_cutoff)+
                  " "+export_feature_correlations+" "+export_univariate_plots+" "+class_label+" "+instance_label+" "+match_label+" "+str(random_state)+" "+str(ignore_features_path)+" "+str(categorical_feature_path)+" "+str(sig_cutoff)+" "+str(jupyterRun)+'\n')
    sh_file.close()
    os.system('bsub < '+job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))
