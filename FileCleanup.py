"""
File: FileCleanupMain.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 11 of STREAMLINE (Optional)- This 'Main' script runs Phase 11 which deletes all temporary files in pipeline output folder.
            This script is not necessary to run, but serves as a convenience to reduce space and clutter following a pipeline run.
Sample Run Command (runs locally only):
    python FileCleanupMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import argparse
import sys
import os
import time
import shutil
import glob

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description="")
    #No defaults
    parser.add_argument('--out-path',dest='output_path',type=str,help='path to output directory')
    parser.add_argument('--exp-name', dest='experiment_name',type=str, help='name of experiment output folder (no spaces)')
    parser.add_argument('--del-time', dest='del_time',type=str, help='delete individual run-time files (but save summary)',default="True")
    parser.add_argument('--del-oldCV', dest='del_oldCV',type=str, help='delete any of the older versions of CV training and testing datasets not overwritten (preserves final training and testing datasets)',default="True")

    options = parser.parse_args(argv[1:])
    experiment_path = options.output_path+'/'+options.experiment_name

    if not os.path.exists(options.output_path):
        raise Exception("Provided output_path does not exist")
    if not os.path.exists(experiment_path):
        raise Exception("Provided experiment name in given output_path does not exist")

    # Get dataset paths for all completed dataset analyses in experiment folder
    datasets = os.listdir(experiment_path)
    experiment_name = experiment_path.split('/')[-1] #Name of experiment folder
    removeList = removeList = ['metadata.pickle','metadata.csv','algInfo.pickle','jobsCompleted','logs','jobs','DatasetComparisons','UsefulNotebooks',experiment_name+'_ML_Pipeline_Report.pdf']
    for text in removeList:
        if text in datasets:
            datasets.remove(text)

    #Delete log folder/files
    try:
        shutil.rmtree(experiment_path+'/'+'logs')
    except:
        pass

    #Delete job folder/files
    try:
        shutil.rmtree(experiment_path+'/'+'jobs')
    except:
        pass

    #Delete jobscompleted folder/files
    try:
        shutil.rmtree(experiment_path+'/'+'jobsCompleted')
    except:
        pass
    #Remake folders (empty) incase user wants to rerun scripts like pdf report from command line
    os.mkdir(experiment_path+'/jobsCompleted')
    os.mkdir(experiment_path+'/jobs')
    os.mkdir(experiment_path+'/logs')

    #Delete target files within each dataset subfolder
    for dataset in datasets:
        #Delete individual runtime files (save runtime summary generated in phase 6)
        if eval(options.del_time):
            try:
                shutil.rmtree(experiment_path+'/'+dataset+'/'+'runtime')
            except:
                pass
        #Delete temporary feature importance pickle files (only needed for phase 4 and then saved as summary files in phase 6)
        try:
            shutil.rmtree(experiment_path+'/'+dataset+'/feature_selection/mutualinformation/pickledForPhase4')
        except:
            pass
        try:
            shutil.rmtree(experiment_path+'/'+dataset+'/feature_selection/multisurf/pickledForPhase4')
        except:
            pass
        #Delete older training and testing CV datasets (does not delete any final versions used for training). Older cv datasets might have been kept to see what they look like prior to preprocessing and feature selection.
        if eval(options.del_oldCV):
            #Delete CV files generated after preprocessing but before feature selection
            files = glob.glob(experiment_path+'/'+dataset+'/CVDatasets/*CVOnly*')
            for f in files:
                try:
                    os.remove(f)
                except:
                    pass
            #Delete CV files generated after CV partitioning but before preprocessing
            files = glob.glob(experiment_path+'/'+dataset+'/CVDatasets/*CVPre*')
            for f in files:
                try:
                    os.remove(f)
                except:
                    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))
