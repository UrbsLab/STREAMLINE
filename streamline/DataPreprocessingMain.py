"""
File: DataPreprocessingMain.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 2 of STREAMLINE - This 'Main' script manages Phase 2 run parameters, updates the metadata file (with user specified run parameters across pipeline run)
             and submits job to run locally (to run serially) or on a linux computing cluster (parallelized).  This script runs DataPreprocessingJob.py which conducts the
             data preproccesing (i.e. scaling and imputation). All 'Main' scripts in this pipeline have the potential to be extended by users to submit jobs to other parallel
             computing frameworks (e.g. cloud computing).
Warnings: Designed to be run following the completion of STREAMLINE Phase 1 (ExploratoryAnalysisMain.py). This script should be run as part of the pipeline regardless
              of whether either scaling or imputation is applied so that the metadata file is updated appropriately for downstream phases.
Sample Run Command (Linux cluster parallelized with all default run parameters):
    python DataPreprocessingMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1
Sample Run Command (Local/serial with with all default run parameters):
    python DataPreprocessingMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1 --run-parallel False
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import sys
import os
import argparse
import glob
import pandas as pd
import DataPreprocessingJob
import time
import csv
import pickle

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description="")
    #No defaults
    parser.add_argument('--out-path',dest='output_path',type=str,help='path to output directory')
    parser.add_argument('--exp-name', dest='experiment_name',type=str, help='name of experiment output folder (no spaces)')
    #Defaults available
    parser.add_argument('--scale',dest='scale_data',type=str,help='perform data scaling (required for SVM, and to use Logistic regression with non-uniform feature importance estimation)',default="True")
    parser.add_argument('--impute', dest='impute_data',type=str,help='perform missing value data imputation (required for most ML algorithms if missing data is present)',default="True")
    parser.add_argument('--multi-impute', dest='multi_impute',type=str,help='applies multivariate imputation to quantitative features, otherwise uses median imputation',default="True")
    parser.add_argument('--over-cv', dest='overwrite_cv',type=str,help='overwrites earlier cv datasets with new scaled/imputed ones',default="True")
    #Lostistical arguments
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='if run parallel on LSF compatible computing cluster',default="True")
    parser.add_argument('--queue',dest='queue',type=str,help='specify name of parallel computing queue (uses our research groups queue by default)',default="i2c2_normal")
    parser.add_argument('--res-mem', dest='reserved_memory', type=int, help='reserved memory for the job (in Gigabytes)',default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int, help='maximum memory before the job is automatically terminated',default=15)
    parser.add_argument('-c','--do-check',dest='do_check', help='Boolean: Specify whether to check for existence of all output files.', action='store_true')

    options = parser.parse_args(argv[1:])
    job_counter = 0

    # Argument checks-------------------------------------------------------------
    if not os.path.exists(options.output_path):
        raise Exception("Output path must exist (from phase 1) before phase 2 can begin")
    if not os.path.exists(options.output_path + '/' + options.experiment_name):
        raise Exception("Experiment must exist (from phase 1) before phase 2 can begin")

    #Unpickle metadata from previous phase
    file = open(options.output_path+'/'+options.experiment_name+'/'+"metadata.pickle", 'rb')
    metadata = pickle.load(file)
    file.close()
    #Load variables specified earlier in the pipeline from metadata
    class_label = metadata['Class Label']
    instance_label = metadata['Instance Label']
    random_state = int(metadata['Random Seed'])
    categorical_cutoff = int(metadata['Categorical Cutoff'])
    cv_partitions = int(metadata['CV Partitions'])
    jupyterRun = metadata['Run From Jupyter Notebook']

    if not options.do_check: #Run job file
        #Iterate through datasets, ignoring common folders
        dataset_paths = os.listdir(options.output_path+"/"+options.experiment_name)
        removeList = removeList = ['metadata.pickle','metadata.csv','algInfo.pickle','jobsCompleted','logs','jobs','DatasetComparisons','UsefulNotebooks']
        for text in removeList:
            if text in dataset_paths:
                dataset_paths.remove(text)

        for dataset_directory_path in dataset_paths:
            full_path = options.output_path+"/"+options.experiment_name+"/"+dataset_directory_path
            for cv_train_path in glob.glob(full_path+"/CVDatasets/*Train.csv"):
                job_counter += 1
                cv_test_path = cv_train_path.replace("Train.csv","Test.csv")
                if eval(options.run_parallel):
                    submitClusterJob(cv_train_path,cv_test_path,options.output_path+'/'+options.experiment_name,options.scale_data,options.impute_data,options.overwrite_cv,categorical_cutoff,class_label,instance_label,random_state,options.reserved_memory,options.maximum_memory,options.queue,options.multi_impute,jupyterRun)
                else:
                    submitLocalJob(cv_train_path,cv_test_path,options.output_path+'/'+options.experiment_name,options.scale_data,options.impute_data,options.overwrite_cv,categorical_cutoff,class_label,instance_label,random_state,options.multi_impute,jupyterRun)

        #Update metadata
        metadata['Use Data Scaling'] = options.scale_data
        metadata['Use Data Imputation'] = options.impute_data
        metadata['Use Multivariate Imputation'] = options.multi_impute
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

        phase2Jobs = []
        for dataset in datasets:
            for cv in range(cv_partitions):
                phase2Jobs.append('job_preprocessing_' + dataset + '_' + str(cv) + '.txt')

        for filename in glob.glob(options.output_path + "/" + options.experiment_name + '/jobsCompleted/job_preprocessing*'):
            ref = filename.split('/')[-1]
            phase2Jobs.remove(ref)
        for job in phase2Jobs:
            print(job)
        if len(phase2Jobs) == 0:
            print("All Phase 2 Jobs Completed")
        else:
            print("Above Phase 2 Jobs Not Completed")

    if not options.do_check:
        print(str(job_counter)+ " jobs submitted in Phase 2")

def submitLocalJob(cv_train_path,cv_test_path,experiment_path,scale_data,impute_data,overwrite_cv,categorical_cutoff,class_label,instance_label,random_state,multi_impute,jupyterRun):
    """ Runs DataPreprocessingJob.py locally on a single CV dataset. These runs will be completed serially rather than in parallel. """
    DataPreprocessingJob.job(cv_train_path,cv_test_path,experiment_path,scale_data,impute_data,overwrite_cv,categorical_cutoff,class_label,instance_label,random_state,multi_impute,jupyterRun)

def submitClusterJob(cv_train_path,cv_test_path,experiment_path,scale_data,impute_data,overwrite_cv,categorical_cutoff,class_label,instance_label,random_state,reserved_memory,maximum_memory,queue,multi_impute,jupyterRun):
    """ Runs DataPreprocessingJob.py on a single CV dataset based on each dataset in phase 1 target data folder. Runs in parallel on a linux-based computing cluster that uses an IBM Spectrum LSF for job scheduling."""
    job_ref = str(time.time())
    job_name = experiment_path+'/jobs/P2_'+job_ref+'_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q '+queue+'\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('#BSUB -R "rusage[mem='+str(reserved_memory)+'G]"'+'\n')
    sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/P2_'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/P2_'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python '+this_file_path+'/DataPreprocessingJob.py '+cv_train_path+" "+cv_test_path+" "+experiment_path+" "+scale_data+
                  " "+impute_data+" "+overwrite_cv+" "+str(categorical_cutoff)+" "+class_label+" "+instance_label+" "+str(random_state)+" "+str(multi_impute)+" "+str(jupyterRun)+'\n')
    sh_file.close()
    os.system('bsub < '+job_name)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
