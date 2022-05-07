import argparse
import os
import sys
import pandas as pd
from LCSDIVE.AnalysisPhase1_pretrained import main as AnalysisPhase1_main
from LCSDIVE import *

'''
Sample Run Command:
python VisualizationMain.py --output-path /Users/robert/Desktop/outputs --experiment-name test1

Local Command:
python VisualizationMain.py --output-path /Users/robert/Desktop/outputs --experiment-name randomtest2 --run-parallel False
'''

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    parser.add_argument('--run-parallel', dest='run_parallel', type=str, help='path to directory containing datasets', default="True")
    parser.add_argument('--res-mem', dest='reserved_memory', type=int, help='reserved memory for the job (in Gigabytes)', default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int, help='maximum memory before the job is automatically terminated', default=15)

    options = parser.parse_args(argv[1:])
    output_path = options.output_path
    experiment_name = options.experiment_name
    if options.run_parallel == 'True':
        run_parallel = 1
    else:
        run_parallel = 0
    reserved_memory = options.reserved_memory
    maximum_memory = options.maximum_memory

    # Argument checks
    if not os.path.exists(output_path):
        raise Exception("Output path must exist (from phase 1) before phase 7 can begin")

    if not os.path.exists(output_path + '/' + experiment_name):
        raise Exception("Experiment must exist (from phase 1) before phase 7 can begin")

    metadata = pd.read_csv(output_path + '/' + experiment_name + '/' + 'metadata.csv').values
    class_label = metadata[0, 1]
    instance_label = metadata[1, 1]
    random_state = int(metadata[2, 1])
    cv_partitions = int(metadata[4, 1])

    dataset_paths = os.listdir(output_path + "/" + experiment_name)
    dataset_paths.remove('logs')
    dataset_paths.remove('jobs')
    dataset_paths.remove('jobsCompleted')
    dataset_paths.remove('metadata.csv')
    for dataset_directory_path in dataset_paths:
        full_path = output_path + "/" + experiment_name + "/" + dataset_directory_path
        cv_datasets_path = full_path+'/CVDatasets'
        model_path = full_path+'/training/pickledModels'
        if not os.path.exists(full_path + '/viz-outputs'):
            os.mkdir(full_path + '/viz-outputs')
        vizoutput_path = full_path + '/viz-outputs'
        vizexperiment_name = 'root'
        AnalysisPhase1_main(['','--d', cv_datasets_path, '--m', model_path, '--o', vizoutput_path, '--e', vizexperiment_name,
                             '--class', class_label, '--inst', instance_label, '--random-state', str(random_state),
                             '--cv', str(cv_partitions), '--cluster', str(run_parallel), '--m1', str(reserved_memory),
                             '--m2', str(maximum_memory)])

if __name__ == '__main__':
    sys.exit(main(sys.argv))


