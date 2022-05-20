"""
File: ModelMain.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 5 of STREAMLINE - This 'Main' script manages Phase 5 run parameters, updates the metadata file (with user specified run parameters across pipeline run)
             and submits job to run locally (to run serially) or on a linux computing cluster (parallelized).  This script runs ModelJob.py which conducts machine learning
             modeling using respective training datasets. This pipeline currently includes the following 13 ML modeling algorithms for binary classification:
             * Naive Bayes, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LGBoost, Support Vector Machine (SVM), Artificial Neural Network (ANN),
             * k Nearest Neighbors (k-NN), Educational Learning Classifier System (eLCS), X Classifier System (XCS), and the Extended Supervised Tracking and Classifying System (ExSTraCS)
             This phase includes hyperparameter optimization of all algorithms (other than naive bayes), model training, model feature importance estimation (using internal algorithm
             estimations, if available, or via permutation feature importance), and performance evaluation on hold out testing data. This script creates a single job for each
             combination of cv dataset (for each original target dataset) and ML modeling algorithm. In addition to an option to check the completion of all jobs, this script also has a
             'resubmit' option that will run any jobs that may have failed from a previous run. All 'Main' scripts in this pipeline have the potential to be extended by users to
             submit jobs to other parallel computing frameworks (e.g. cloud computing).
Warnings: Designed to be run following the completion of STREAMLINE Phase 4 (FeatureSelectionMain.py). SVM modeling should only be applied when data scaling is applied by the pipeline
            Logistic Regression' baseline model feature importance estimation is determined by the exponential of the feature's coefficient. This should only be used if data scaling is
            applied by the pipeline. Otherwise 'use_uniform_FI' should be True.
Sample Run Command (Linux cluster parallelized with all default run parameters):
    python ModelMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1
Sample Run Command (Local/serial with with all default run parameters):
    python ModelMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1 --run-parallel False
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import argparse
import os
import sys
import pandas as pd
import glob
import ModelJob
import time
import csv
import random
import pickle

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description='')
    #No defaults
    parser.add_argument('--out-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--exp-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    #Sets default run all or none to make algorithm selection from command line simpler
    parser.add_argument('--do-all', dest='do_all', type=str, help='run all modeling algorithms by default (when set False, individual algorithms are activated individually)',default='True')
    #ML modeling algorithms: Defaults available
    parser.add_argument('--do-NB', dest='do_NB', type=str, help='run naive bayes modeling',default='None')
    parser.add_argument('--do-LR', dest='do_LR', type=str, help='run logistic regression modeling',default='None')
    parser.add_argument('--do-DT', dest='do_DT', type=str, help='run decision tree modeling',default='None')
    parser.add_argument('--do-RF', dest='do_RF', type=str, help='run random forest modeling',default='None')
    parser.add_argument('--do-GB', dest='do_GB', type=str, help='run gradient boosting modeling',default='None')
    parser.add_argument('--do-XGB', dest='do_XGB', type=str, help='run XGBoost modeling',default='None')
    parser.add_argument('--do-LGB', dest='do_LGB', type=str, help='run LGBoost modeling',default='None')
    parser.add_argument('--do-CGB', dest='do_CGB', type=str,help='run CatBoost modeling',default='None')
    parser.add_argument('--do-SVM', dest='do_SVM', type=str, help='run support vector machine modeling',default='None')
    parser.add_argument('--do-ANN', dest='do_ANN', type=str, help='run artificial neural network modeling',default='None')
    parser.add_argument('--do-KNN', dest='do_KNN', type=str, help='run k-nearest neighbors classifier modeling',default='None')
    parser.add_argument('--do-GP', dest='do_GP', type=str, help='run genetic programming symbolic classifier modeling',default='None')
    #Experimental ML modeling algorithms (rule-based ML algorithms that are in develompent by our research group)
    parser.add_argument('--do-eLCS', dest='do_eLCS', type=str, help='run eLCS modeling (a basic supervised-learning learning classifier system)',default='False')
    parser.add_argument('--do-XCS', dest='do_XCS', type=str, help='run XCS modeling (a supervised-learning-only implementation of the best studied learning classifier system)',default='False')
    parser.add_argument('--do-ExSTraCS', dest='do_ExSTraCS', type=str, help='run ExSTraCS modeling (a learning classifier system designed for biomedical data mining)',default='None')
    ### Add new algorithms here...
    #Other Analysis Parameters - Defaults available
    parser.add_argument('--metric', dest='primary_metric', type=str,help='primary scikit-learn specified scoring metric used for hyperparameter optimization and permutation-based model feature importance evaluation', default='balanced_accuracy')
    parser.add_argument('--subsample', dest='training_subsample', type=int, help='for long running algos (XGB,SVM,ANN,KN), option to subsample training set (0 for no subsample)', default=0)
    parser.add_argument('--use-uniformFI', dest='use_uniform_FI', type=str, help='overrides use of any available feature importance estimate methods from models, instead using permutation_importance uniformly',default='True')
    #Hyperparameter sweep options - Defaults available
    parser.add_argument('--n-trials', dest='n_trials', type=str,help='# of bayesian hyperparameter optimization trials using optuna (specify an integer or None)', default=200)
    parser.add_argument('--timeout', dest='timeout', type=str,help='seconds until hyperparameter sweep stops running new trials (Note: it may run longer to finish last trial started) If set to None, STREAMLINE is completely replicable, but will take longer to run', default=900) #900 sec = 15 minutes default
    parser.add_argument('--export-hyper-sweep', dest='export_hyper_sweep_plots', type=str, help='export optuna-generated hyperparameter sweep plots', default='False')
    #LCS specific parameters - Defaults available
    parser.add_argument('--do-LCS-sweep', dest='do_lcs_sweep', type=str, help='do LCS hyperparam tuning or use below params',default='False')
    parser.add_argument('--nu', dest='nu', type=int, help='fixed LCS nu param (recommended range 1-10), set to larger value for data with less or no noise', default=1)
    parser.add_argument('--iter', dest='iterations', type=int, help='fixed LCS # learning iterations param', default=200000)
    parser.add_argument('--N', dest='N', type=int, help='fixed LCS rule population maximum size param', default=2000)
    parser.add_argument('--lcs-timeout', dest='lcs_timeout', type=int, help='seconds until hyperparameter sweep stops for LCS algorithms', default=1200)
    #Lostistical arguments - Defaults available
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='if run parallel on LSF compatible computing cluster',default="True")
    parser.add_argument('--queue',dest='queue',type=str,help='specify name of parallel computing queue (uses our research groups queue by default)',default="i2c2_normal")
    parser.add_argument('--res-mem', dest='reserved_memory', type=int, help='reserved memory for the job (in Gigabytes)',default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int, help='maximum memory before the job is automatically terminated',default=15)
    parser.add_argument('-c','--do-check',dest='do_check', help='Boolean: Specify whether to check for existence of all output files.', action='store_true')
    parser.add_argument('-r','--do-resubmit',dest='do_resubmit', help='Boolean: Rerun any jobs that did not complete (or failed) in an earlier run.', action='store_true')

    options = parser.parse_args(argv[1:])
    job_counter = 0

    # Argument checks
    if not os.path.exists(options.output_path):
        raise Exception("Output path must exist (from phase 1) before phase 5 can begin")
    if not os.path.exists(options.output_path + '/' + options.experiment_name):
        raise Exception("Experiment must exist (from phase 1) before phase 5 can begin")

    #Unpickle metadata from previous phase
    file = open(options.output_path+'/'+options.experiment_name+'/'+"metadata.pickle", 'rb')
    metadata = pickle.load(file)
    file.close()
    #Load variables specified earlier in the pipeline from metadata
    class_label = metadata['Class Label']
    instance_label = metadata['Instance Label']
    random_state = int(metadata['Random Seed'])
    cv_partitions = int(metadata['CV Partitions'])
    filter_poor_features = metadata['Filter Poor Features']
    jupyterRun = metadata['Run From Jupyter Notebook']

    if options.do_resubmit: #Attempts to resolve optuna hyperparameter optimization hangup (i.e. when it runs indefinitely for a given random seed attempt)
        random_state = random.randint(1,1000)

    #Create ML modeling algorithm information dictionary, given as ['algorithm used (set to true initially by default)','algorithm abreviation', 'color used for algorithm on figures']
    ### Note that other named colors used by matplotlib can be found here: https://matplotlib.org/3.5.0/_images/sphx_glr_named_colors_003.png
    ### Make sure new ML algorithm abbreviations and color designations are unique
    algInfo = {}
    algInfo['Naive Bayes'] = [True,'NB','silver']
    algInfo['Logistic Regression'] = [True,'LR','dimgrey']
    algInfo['Decision Tree'] = [True,'DT','yellow']
    algInfo['Random Forest'] = [True,'RF','blue']
    algInfo['Gradient Boosting'] = [True,'GB','cornflowerblue']
    algInfo['Extreme Gradient Boosting'] = [True,'XGB','cyan']
    algInfo['Light Gradient Boosting'] = [True,'LGB','pink']
    algInfo['Category Gradient Boosting'] = [True,'CGB','magenta']
    algInfo['Support Vector Machine'] = [True,'SVM','orange']
    algInfo['Artificial Neural Network'] = [True,'ANN','red']
    algInfo['K-Nearest Neightbors'] = [True,'KNN','chocolate']
    algInfo['Genetic Programming'] = [True,'GP','purple']
    algInfo['eLCS'] = [True,'eLCS','green']
    algInfo['XCS'] = [True,'XCS','olive']
    algInfo['ExSTraCS'] = [True,'ExSTraCS','lawngreen']
    ### Add new algorithms here...

    #Set up ML algorithm True/False use
    if not eval(options.do_all): #If do all algorithms is false
        for key in algInfo:
            algInfo[key][0] = False #Set algorithm use to False

    #Set algorithm use truth for each algorithm specified by user (i.e. if user specified True/False for a specific algorithm)
    if not options.do_NB == 'None':
        algInfo['Naive Bayes'][0] = eval(options.do_NB)
    if not options.do_LR == 'None':
        algInfo['Logistic Regression'][0] = eval(options.do_LR)
    if not options.do_DT == 'None':
        algInfo['Decision Tree'][0] = eval(options.do_DT)
    if not options.do_RF == 'None':
        algInfo['Random Forest'][0] = eval(options.do_RF)
    if not options.do_GB == 'None':
        algInfo['Gradient Boosting'][0] = eval(options.do_GB)
    if not options.do_XGB == 'None':
        algInfo['Extreme Gradient Boosting'][0] = eval(options.do_XGB)
    if not options.do_LGB == 'None':
        algInfo['Light Gradient Boosting'][0] = eval(options.do_LGB)
    if not options.do_CGB == 'None':
        algInfo['Category Gradient Boosting'][0] = eval(options.do_CGB)
    if not options.do_SVM == 'None':
        algInfo['Support Vector Machine'][0] = eval(options.do_SVM)
    if not options.do_ANN == 'None':
        algInfo['Artificial Neural Network'][0] = eval(options.do_ANN)
    if not options.do_KNN == 'None':
        algInfo['K-Nearest Neightbors'][0] = eval(options.do_KNN)
    if not options.do_GP == 'None':
        algInfo['Genetic Programming'][0] = eval(options.do_GP)
    if not options.do_eLCS == 'None':
        algInfo['eLCS'][0] = eval(options.do_eLCS)
    if not options.do_XCS == 'None':
        algInfo['XCS'][0] = eval(options.do_XCS)
    if not options.do_ExSTraCS == 'None':
        algInfo['ExSTraCS'][0] = eval(options.do_ExSTraCS)
    ### Add new algorithms here...

    #Pickle the algorithm information dictionary for future use
    pickle_out = open(options.output_path+'/'+options.experiment_name+'/'+"algInfo.pickle", 'wb')
    pickle.dump(algInfo,pickle_out)
    pickle_out.close()

    #Make list of algorithms to be run (full names)
    algorithms = []
    for key in algInfo:
        if algInfo[key][0]: #Algorithm is true
            algorithms.append(key)

    if not options.do_check and not options.do_resubmit: #Run job submission
        dataset_paths = os.listdir(options.output_path + "/" + options.experiment_name)
        removeList = ['metadata.pickle','metadata.csv','algInfo.pickle','jobsCompleted','logs','jobs','DatasetComparisons',options.experiment_name+'_ML_Pipeline_Report.pdf']
        for text in removeList:
            if text in dataset_paths:
                dataset_paths.remove(text)

        for dataset_directory_path in dataset_paths:
            full_path = options.output_path + "/" + options.experiment_name + "/" + dataset_directory_path
            if not os.path.exists(full_path+'/models'):
                os.mkdir(full_path+'/models')
            if not os.path.exists(full_path+'/model_evaluation'):
                os.mkdir(full_path+'/model_evaluation')
            if not os.path.exists(full_path+'/models/pickledModels'):
                os.mkdir(full_path+'/models/pickledModels')
            for cvCount in range(cv_partitions):
                train_file_path = full_path+'/CVDatasets/'+dataset_directory_path+"_CV_"+str(cvCount)+"_Train.csv"
                test_file_path = full_path + '/CVDatasets/' + dataset_directory_path + "_CV_" + str(cvCount) + "_Test.csv"
                for algorithm in algorithms:
                    algAbrev = algInfo[algorithm][1]
                    algNoSpace = algorithm.replace(" ", "_")
                    job_counter += 1
                    if eval(options.run_parallel):
                        submitClusterJob(algNoSpace,train_file_path,test_file_path,full_path,options.n_trials,options.timeout,options.lcs_timeout,options.export_hyper_sweep_plots,instance_label,class_label,random_state,options.output_path+'/'+options.experiment_name,cvCount,filter_poor_features,options.reserved_memory,options.maximum_memory,options.do_lcs_sweep,options.nu,options.iterations,options.N,options.training_subsample,options.queue,options.use_uniform_FI,options.primary_metric,algAbrev,jupyterRun)
                    else:
                        submitLocalJob(algNoSpace,train_file_path,test_file_path,full_path,options.n_trials,options.timeout,options.lcs_timeout,options.export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,options.do_lcs_sweep,options.nu,options.iterations,options.N,options.training_subsample,options.use_uniform_FI,options.primary_metric,algAbrev,jupyterRun)

        #Update metadata
        metadata['Naive Bayes'] = str(algInfo['Naive Bayes'][0])
        metadata['Logistic Regression'] = str(algInfo['Logistic Regression'][0])
        metadata['Decision Tree'] = str(algInfo['Decision Tree'][0])
        metadata['Random Forest'] = str(algInfo['Random Forest'][0])
        metadata['Gradient Boosting'] = str(algInfo['Gradient Boosting'][0])
        metadata['Extreme Gradient Boosting'] = str(algInfo['Extreme Gradient Boosting'][0])
        metadata['Light Gradient Boosting'] = str(algInfo['Light Gradient Boosting'][0])
        metadata['Category Gradient Boosting'] = str(algInfo['Category Gradient Boosting'][0])
        metadata['Support Vector Machine'] = str(algInfo['Support Vector Machine'][0])
        metadata['Artificial Neural Network'] = str(algInfo['Artificial Neural Network'][0])
        metadata['K-Nearest Neightbors'] = str(algInfo['K-Nearest Neightbors'][0])
        metadata['Genetic Programming'] = str(algInfo['Genetic Programming'][0])
        metadata['eLCS'] = str(algInfo['eLCS'][0])
        metadata['XCS'] = str(algInfo['XCS'][0])
        metadata['ExSTraCS'] = str(algInfo['ExSTraCS'][0])
        ### Add new algorithms here...
        metadata['Primary Metric'] = options.primary_metric
        metadata['Training Subsample for KNN,ANN,SVM,and XGB'] = options.training_subsample
        metadata['Uniform Feature Importance Estimation (Models)'] = options.use_uniform_FI
        metadata['Hyperparameter Sweep Number of Trials'] = options.n_trials
        metadata['Hyperparameter Timeout'] = options.timeout
        metadata['Export Hyperparameter Sweep Plots'] = options.export_hyper_sweep_plots
        metadata['Do LCS Hyperparameter Sweep'] = options.do_lcs_sweep
        metadata['nu'] = options.nu
        metadata['Training Iterations'] = options.iterations
        metadata['N (Rule Population Size)'] = options.N
        metadata['LCS Hyperparameter Sweep Timeout'] = options.lcs_timeout
        #Pickle the metadata for future use
        pickle_out = open(options.output_path+'/'+options.experiment_name+'/'+"metadata.pickle", 'wb')
        pickle.dump(metadata,pickle_out)
        pickle_out.close()

    elif options.do_check and not options.do_resubmit: #run job completion checks
        datasets = os.listdir(options.output_path + "/" + options.experiment_name)
        removeList = removeList = ['metadata.pickle','metadata.csv','algInfo.pickle','jobsCompleted','logs','jobs','DatasetComparisons','UsefulNotebooks']
        for text in removeList:
            if text in datasets:
                datasets.remove(text)

        phase5Jobs = []
        for dataset in datasets:
            for cv in range(cv_partitions):
                for algorithm in algorithms:
                    phase5Jobs.append('job_model_' + dataset + '_' + str(cv) +'_' +algInfo[algorithm][1]+'.txt') #use algorithm abreviation for filenames

        for filename in glob.glob(options.output_path + "/" + options.experiment_name + '/jobsCompleted/job_model*'):
            ref = filename.split('/')[-1]
            phase5Jobs.remove(ref)
        for job in phase5Jobs:
            print(job)
        if len(phase5Jobs) == 0:
            print("All Phase 5 Jobs Completed")
        else:
            print("Above Phase 5 Jobs Not Completed")
        print()

    elif options.do_resubmit and not options.do_check: #resubmit any jobs that didn't finish in previous run (mix of job check and job submit)
        datasets = os.listdir(options.output_path + "/" + options.experiment_name)
        removeList = removeList = ['metadata.pickle','metadata.csv','algInfo.pickle','jobsCompleted','logs','jobs','DatasetComparisons','UsefulNotebooks']
        for text in removeList:
            if text in datasets:
                datasets.remove(text)

        #start by making list of finished jobs instead of all jobs then step through loop
        phase5completed = []
        for filename in glob.glob(options.output_path + "/" + options.experiment_name + '/jobsCompleted/job_model*'):
            ref = filename.split('/')[-1]
            phase5completed.append(ref)

        for dataset in datasets:
            for cv in range(cv_partitions):
                for algorithm in algorithms:
                    algAbrev = algInfo[algorithm][1]
                    algNoSpace = algorithm.replace(" ", "_")
                    targetFile = 'job_model_' + dataset + '_' + str(cv) +'_' +algInfo[algorithm][1]+'.txt'
                    if targetFile not in phase5completed: #target for a re-submit
                        full_path = options.output_path + "/" + options.experiment_name + "/" + dataset
                        train_file_path = full_path+'/CVDatasets/'+dataset+"_CV_"+str(cv)+"_Train.csv"
                        test_file_path = full_path + '/CVDatasets/' + dataset + "_CV_" + str(cv) + "_Test.csv"
                        if eval(options.run_parallel):
                            job_counter += 1
                            submitClusterJob(algNoSpace,train_file_path,test_file_path,full_path,options.n_trials,options.timeout,options.lcs_timeout,options.export_hyper_sweep_plots,instance_label,class_label,random_state,options.output_path+'/'+options.experiment_name,cv,filter_poor_features,options.reserved_memory,options.maximum_memory,options.do_lcs_sweep,options.nu,options.iterations,options.N,options.training_subsample,options.queue,options.use_uniform_FI,options.primary_metric,algAbrev,jupyterRun)
                        else:
                            submitLocalJob(algNoSpace,train_file_path,test_file_path,full_path,options.n_trials,options.timeout,options.lcs_timeout,options.export_hyper_sweep_plots,instance_label,class_label,random_state,cv,filter_poor_features,options.do_lcs_sweep,options.nu,options.iterations,options.N,options.training_subsample,options.use_uniform_FI,options.primary_metric,algAbrev,jupyterRun)
    else:
        print("Run options in conflict. Do not request to run check and resubmit at the same time.")

    if not options.do_check:
        print(str(job_counter)+ " jobs submitted in Phase 5")

def submitLocalJob(algNoSpace,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,do_lcs_sweep,nu,iterations,N,training_subsample,use_uniform_FI,primary_metric,algAbrev,jupyterRun):
    """ Runs ModelJob.py locally, once for each combination of cv dataset (for each original target dataset) and ML modeling algorithm. These runs will be completed serially rather than in parallel. """
    ModelJob.job(algNoSpace,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,do_lcs_sweep,nu,iterations,N,training_subsample,use_uniform_FI,primary_metric,algAbrev,jupyterRun)

def submitClusterJob(algNoSpace,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,experiment_path,cvCount,filter_poor_features,reserved_memory,maximum_memory,do_lcs_sweep,nu,iterations,N,training_subsample,queue,use_uniform_FI,primary_metric,algAbrev,jupyterRun):
    """ Runs ModelJob.py once for each combination of cv dataset (for each original target dataset) and ML modeling algorithm. Runs in parallel on a linux-based computing cluster that uses an IBM Spectrum LSF for job scheduling."""
    job_ref = str(time.time())
    job_name = experiment_path+'/jobs/P5_'+str(algAbrev)+'_'+str(cvCount)+'_'+job_ref+'_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q '+queue+'\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('#BSUB -R "rusage[mem='+str(reserved_memory)+'G]"'+'\n')
    sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/P5_'+str(algAbrev)+'_'+str(cvCount)+'_'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/P5_'+str(algAbrev)+'_'+str(cvCount)+'_'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python '+this_file_path+'/ModelJob.py '+algNoSpace+" "+train_file_path+" "+test_file_path+" "+full_path+" "+
                  str(n_trials)+" "+str(timeout)+" "+str(lcs_timeout)+" "+export_hyper_sweep_plots+" "+instance_label+" "+class_label+" "+
                  str(random_state)+" "+str(cvCount)+" "+str(filter_poor_features)+" "+str(do_lcs_sweep)+" "+str(nu)+" "+str(iterations)+" "+str(N)+" "+str(training_subsample)+" "+str(use_uniform_FI)+" "+str(primary_metric)+" "+str(algAbrev)+" "+str(jupyterRun)+'\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))
