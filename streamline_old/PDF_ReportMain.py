"""
File: PDF_ReportMain.py
Authors: Ryan J. Urbanowicz, Richard Zhang, Wilson Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 8 and 10 of STREAMLINE (Optional)- This 'Main' script manages Phase 8 and 10 run parameters, and submits job to run locally (to run serially) or on a linux computing
             cluster (parallelized). This script runs PDF_ReportJob.py which generates a formatted PDF summary report of key pipeline results.
             All 'Main' scripts in this pipeline have the potential to be extended by users to submit jobs to other parallel computing frameworks (e.g. cloud computing).
Warnings: Designed to be run following the completion of either STREAMLINE Phase 6 (StatsMain.py), and or Phase 7 (DataCompareMain.py).
Sample Run Command (Linux cluster parallelized with all default run parameters):
    python PDF_ReportMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1
Sample Run Command (Local/serial with with all default run parameters):
    python PDF_ReportMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1 --run-parallel False
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import os
import re
import sys
import argparse
import time
import glob
import PDF_ReportJob

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description="")
    #No defaults

    parser.add_argument('--out-path',dest='output_path',type=str,help='path to output directory')
    parser.add_argument('--exp-name', dest='experiment_name',type=str, help='name of experiment output folder (no spaces)')

    #Defaults available (Note default settings will generate pdf summary for Training analysis, but arguments must be specified for summary of models applied to new/replication data)
    parser.add_argument('--training',dest='training',type=str,help='Indicate True or False for whether to generate pdf summary for pipeline training or followup application analysis to new dataset',default='True')
    parser.add_argument('--rep-path',dest='rep_data_path',type=str,help='path to directory containing replication or hold-out testing datasets (must have at least all features with same labels as in original training dataset)',default="None")
    parser.add_argument('--dataset',dest='dataset_for_rep',type=str,help='path to target original training dataset',default="None")
    #Lostistical arguments
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='if run parallel on LSF compatible computing cluster',default="True")
    parser.add_argument('--queue',dest='queue',type=str,help='specify name of parallel computing queue (uses our research groups queue by default)',default="i2c2_normal")
    parser.add_argument('--res-mem', dest='reserved_memory', type=int, help='reserved memory for the job (in Gigabytes)',default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int, help='maximum memory before the job is automatically terminated',default=15)
    parser.add_argument('-c','--do-check',dest='do_check', help='Boolean: Specify whether to check for existence of all output files.', action='store_true')

    options = parser.parse_args(argv[1:])
    job_counter = 0
    experiment_path = options.output_path+'/'+options.experiment_name

    #Check that user has specified additional run parameters if intending to generate a replication analysis summary (when applying previously trained models)
    if not options.training == 'True':
        if options.rep_data_path == 'None' or options.dataset_for_rep == 'None':
            raise Exception('Replication and Dataset paths must be specified as arguments to generate PDF summary on new data analysis!')

    if not options.do_check: #Run job submission
        job_counter += 1
        if eval(options.run_parallel):
            submitClusterJob(experiment_path,options.training,options.rep_data_path,options.dataset_for_rep,options.reserved_memory,options.maximum_memory,options.queue)
        else:
            submitLocalJob(experiment_path,options.training,options.rep_data_path,options.dataset_for_rep)

    else: #run job completion checks
        if options.training == 'True': # Make pdf summary for training analysis
            for filename in glob.glob(options.output_path + "/" + options.experiment_name+'/jobsCompleted/job_data_pdf_training*'):
                if filename.split('/')[-1] == 'job_data_pdf_training.txt':
                    print("Phase 8 Job Completed")
                else:
                    print("Phase 8 Job Not Completed")
        else: #Make pdf summary for application analysis
            train_name = options.dataset_for_rep.split('/')[-1].split('.')[0]
            for filename in glob.glob(options.output_path + "/" + options.experiment_name+'/jobsCompleted/job_data_pdf_apply_'+str(train_name)+'*'):
                if filename.split('/')[-1] == 'job_data_pdf_apply_'+str(train_name)+'.txt':
                    print("Phase 10 Job Completed")
                else:
                    print("Phase 10 Job Not Completed")

    if not options.do_check:
        if options.training == 'True':
            print(str(job_counter)+ " job submitted in Phase 8")
        else:
            print(str(job_counter)+ " job submitted in Phase 10")

def submitLocalJob(experiment_path,training,rep_data_path,dataset_for_rep):
    """ Runs PDF_ReportJob.py locally, once. """
    PDF_ReportJob.job(experiment_path,training,rep_data_path,dataset_for_rep)

def submitClusterJob(experiment_path,training,rep_data_path,dataset_for_rep,reserved_memory,maximum_memory,queue):
    """ Runs PDF_ReportJob.py once. Runs on a linux-based computing cluster that uses an IBM Spectrum LSF for job scheduling."""
    job_ref = str(time.time())
    job_name = experiment_path + '/jobs/PDF_' + job_ref + '_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q '+queue+'\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('#BSUB -R "rusage[mem='+str(reserved_memory)+'G]"'+'\n')
    sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/PDF_'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/PDF_'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python ' + this_file_path + '/PDF_ReportJob.py ' + experiment_path+' '+training+' '+rep_data_path+' '+dataset_for_rep + '\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))
