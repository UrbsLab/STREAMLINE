"""
File: FeatureSelectionMain.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 4 of STREAMLINE - This 'Main' script manages Phase 4 run parameters, updates the metadata file (with user specified run parameters across pipeline run)
             and submits job to run locally (to run serially) or on a linux computing cluster (parallelized).  This script runs FeatureSelectionJob.py which creates an average feature importance
             summary across all CV datasets from Phase 3 and applies collective feature selection (i.e. takes the union of features identified as 'important' by either of the implemented
             feature importance estimation algorithms). Allows user to keep all features determined to be potentially informative, as well as to specify a max_features_to_keep in the
             case of large feature spaces to reduce computational time. This script runs quickly so one job per original target dataset is run rather than one for each cv dataset. All
             'Main' scripts in this pipeline have the potential to be extended by users to submit jobs to other parallel computing frameworks (e.g. cloud computing).
Warnings: Designed to be run following the completion of STREAMLINE Phase 3 (FeatureImportanceMain.py).
Sample Run Command (Linux cluster parallelized with all default run parameters):
    python FeatureSelectionMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1
Sample Run Command (Local/serial with with all default run parameters):
    python FeatureSelectionMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1 --run-parallel False
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import argparse
import os
import sys
import pandas as pd
import FeatureSelectionJob
import time
import csv
import glob
import pickle

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description='')
    #No defaults
    parser.add_argument('--out-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--exp-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    #Defaults available
    parser.add_argument('--max-feat', dest='max_features_to_keep', type=int,help='max features to keep (only applies if filter_poor_features is True)', default=2000)
    parser.add_argument('--filter-feat', dest='filter_poor_features', type=str, help='filter out the worst performing features prior to modeling',default='True')
    parser.add_argument('--top-features', dest='top_features', type=int,help='number of top features to illustrate in figures', default=40)
    parser.add_argument('--export-scores', dest='export_scores', type=str,help='export figure summarizing average feature importance scores over cv partitions', default='True')
    parser.add_argument('--over-cv', dest='overwrite_cv',type=str,help='overwrites working cv datasets with new feature subset datasets',default="True")
    #Lostistical arguments
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='if run parallel on LSF compatible computing cluster',default="True")
    parser.add_argument('--queue',dest='queue',type=str,help='specify name of parallel computing queue (uses our research groups queue by default)',default="i2c2_normal")
    parser.add_argument('--res-mem', dest='reserved_memory', type=int, help='reserved memory for the job (in Gigabytes)',default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int, help='maximum memory before the job is automatically terminated',default=15)
    parser.add_argument('-c','--do-check',dest='do_check', help='Boolean: Specify whether to check for existence of all output files.', action='store_true')

    options = parser.parse_args(argv[1:])
    job_counter = 0

    #Unpickle metadata from previous phase
    file = open(options.output_path+'/'+options.experiment_name+'/'+"metadata.pickle", 'rb')
    metadata = pickle.load(file)
    file.close()
    #Load variables specified earlier in the pipeline from metadata
    class_label = metadata['Class Label']
    instance_label = metadata['Instance Label']
    cv_partitions = int(metadata['CV Partitions'])
    do_mutual_info = metadata['Use Mutual Information']
    do_multisurf = metadata['Use MultiSURF']
    jupyterRun = metadata['Run From Jupyter Notebook']

    # Argument checks
    if not os.path.exists(options.output_path):
        raise Exception("Output path must exist (from phase 1) before phase 4 can begin")
    if not os.path.exists(options.output_path + '/' + options.experiment_name):
        raise Exception("Experiment must exist (from phase 1) before phase 4 can begin")

    if not options.do_check: #Run job file
        dataset_paths = os.listdir(options.output_path + "/" + options.experiment_name)
        removeList = removeList = ['metadata.pickle','metadata.csv','algInfo.pickle','jobsCompleted','logs','jobs','DatasetComparisons','UsefulNotebooks']
        for text in removeList:
            if text in dataset_paths:
                dataset_paths.remove(text)

        for dataset_directory_path in dataset_paths:
            full_path = options.output_path + "/" + options.experiment_name + "/" + dataset_directory_path
            job_counter += 1
            if eval(options.run_parallel):
                submitClusterJob(full_path,options.output_path+'/'+options.experiment_name,do_mutual_info,do_multisurf,options.max_features_to_keep,options.filter_poor_features,options.top_features,options.export_scores,class_label,instance_label,cv_partitions,options.overwrite_cv,options.reserved_memory,options.maximum_memory,options.queue,jupyterRun)
            else:
                submitLocalJob(full_path,do_mutual_info,do_multisurf,options.max_features_to_keep,options.filter_poor_features,options.top_features,options.export_scores,class_label,instance_label,cv_partitions,options.overwrite_cv,jupyterRun)

        #Update metadata
        metadata['Max Features to Keep'] = options.max_features_to_keep
        metadata['Filter Poor Features'] = options.filter_poor_features
        metadata['Top Features to Display'] = options.top_features
        metadata['Export Feature Importance Plot'] = options.export_scores
        metadata['Overwrite CV Datasets'] = options.overwrite_cv
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

        phase4Jobs = []
        for dataset in datasets:
            phase4Jobs.append('job_featureselection_' + dataset + '.txt')

        for filename in glob.glob(options.output_path + "/" + options.experiment_name + '/jobsCompleted/job_featureselection*'):
            ref = filename.split('/')[-1]
            phase4Jobs.remove(ref)
        for job in phase4Jobs:
            print(job)
        if len(phase4Jobs) == 0:
            print("All Phase 4 Jobs Completed")
        else:
            print("Above Phase 4 Jobs Not Completed")

    if not options.do_check:
        print(str(job_counter)+ " jobs submitted in Phase 4")

def submitLocalJob(full_path,do_mutual_info,do_multisurf,max_features_to_keep,filter_poor_features,top_features,export_scores,class_label,instance_label,cv_partitions,overwrite_cv,jupyterRun):
    """ Runs FeatureSelectionJob.py locally, once for each of the original target datasets (all CV datasets analyzed at once). These runs will be completed serially rather than in parallel. """
    FeatureSelectionJob.job(full_path,do_mutual_info,do_multisurf,max_features_to_keep,filter_poor_features,top_features,export_scores,class_label,instance_label,cv_partitions,overwrite_cv,jupyterRun)

def submitClusterJob(full_path,experiment_path,do_mutual_info,do_multisurf,max_features_to_keep,filter_poor_features,top_features,export_scores,class_label,instance_label,cv_partitions,overwrite_cv,reserved_memory,maximum_memory,queue,jupyterRun):
    """ Runs FeatureSelectionJob.py once for each of the original target datasets (all CV datasets analyzed at once). Runs in parallel on a linux-based computing cluster that uses an IBM Spectrum LSF for job scheduling."""
    job_ref = str(time.time())
    job_name = experiment_path+'/jobs/P4_'+job_ref+'_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q '+queue+'\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('#BSUB -R "rusage[mem='+str(reserved_memory)+'G]"'+'\n')
    sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/P4_'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/P4_'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python '+this_file_path+'/FeatureSelectionJob.py '+full_path+" "+do_mutual_info+" "+do_multisurf+" "+
                  str(max_features_to_keep)+" "+filter_poor_features+" "+str(top_features)+" "+export_scores+" "+class_label+" "+instance_label+" "+str(cv_partitions)+" "+overwrite_cv+" "+jupyterRun+'\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))
