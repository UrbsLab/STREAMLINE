import os
import glob
import pickle
from pathlib import Path


def check_phase_1(output_path, experiment_name, datasets):
    phase1_jobs = []
    for dataset in datasets:
        phase1_jobs.append('job_exploratory_' + dataset + '.txt')
    for filename in glob.glob(output_path + "/" + experiment_name + '/jobsCompleted/job_exploratory*'):
        filename = str(Path(filename).as_posix())
        ref = filename.split('/')[-1]
        phase1_jobs.remove(ref)
    return phase1_jobs


def check_phase_2(output_path, experiment_name, datasets):
    file = open(output_path + '/' + experiment_name + '/' + "metadata.pickle", 'rb')
    cv_partitions = pickle.load(file)['CV Partitions']
    file.close()

    phase2_jobs = []
    for dataset in datasets:
        for cv in range(cv_partitions):
            phase2_jobs.append('job_preprocessing_' + dataset + '_' + str(cv) + '.txt')

    for filename in glob.glob(output_path + "/" + experiment_name + '/jobsCompleted/job_preprocessing*'):
        filename = str(Path(filename).as_posix())
        ref = filename.split('/')[-1]
        phase2_jobs.remove(ref)
    return phase2_jobs


def check_phase_3(output_path, experiment_name, datasets):
    file = open(output_path + '/' + experiment_name + '/' + "metadata.pickle", 'rb')
    metadata = pickle.load(file)
    file.close()
    cv_partitions = metadata['CV Partitions']
    do_multisurf = metadata['Use Mutual Information']
    do_mutual_info = metadata['Use MultiSURF']

    phase3_jobs = []
    for dataset in datasets:
        for cv in range(cv_partitions):
            if do_multisurf:
                phase3_jobs.append('job_multisurf_' + dataset + '_' + str(cv) + '.txt')
            if do_mutual_info:
                phase3_jobs.append('job_mutual_information_' + dataset + '_' + str(cv) + '.txt')

    for filename in glob.glob(output_path + "/" + experiment_name + '/jobsCompleted/job_mu*'):
        filename = str(Path(filename).as_posix())
        ref = filename.split('/')[-1]
        phase3_jobs.remove(ref)
    return phase3_jobs


def check_phase_4(output_path, experiment_name, datasets):
    phase4_jobs = []
    for dataset in datasets:
        phase4_jobs.append('job_featureselection_' + dataset + '.txt')

    for filename in glob.glob(output_path + "/" + experiment_name + '/jobsCompleted/job_featureselection*'):
        filename = str(Path(filename).as_posix())
        ref = filename.split('/')[-1]
        phase4_jobs.remove(ref)
    return phase4_jobs


def check_phase_5(output_path, experiment_name, datasets):
    try:
        file = open(output_path + '/' + experiment_name + '/' + "metadata.pickle", 'rb')
        cv_partitions = pickle.load(file)['CV Partitions']
        file.close()

        pickle_in = open(output_path + '/' + experiment_name + '/' + "algInfo.pickle", 'rb')
        alg_info = pickle.load(pickle_in)
        algorithms = list()
        ABBREVIATION = dict()
        for algorithm in alg_info.keys():
            ABBREVIATION[algorithm] = alg_info[algorithm][1]
            if alg_info[algorithm][0]:
                algorithms.append(algorithm)
        pickle_in.close()
        phase5_jobs = []
        for dataset in datasets:
            for cv in range(cv_partitions):
                for algorithm in algorithms:
                    phase5_jobs.append('job_model_' + dataset + '_' + str(cv) + '_' + ABBREVIATION[algorithm] + '.txt')

        for filename in glob.glob(output_path + "/" + experiment_name + '/jobsCompleted/job_model*'):
            filename = str(Path(filename).as_posix())
            ref = filename.split('/')[-1]
            phase5_jobs.remove(ref)
        return phase5_jobs
    except Exception:
        return ['NOT REACHED YET']


def check_phase_6(output_path, experiment_name, datasets):
    phase6_jobs = []
    for dataset in datasets:
        phase6_jobs.append('job_stats_' + dataset + '.txt')

    for filename in glob.glob(output_path + "/" + experiment_name + '/jobsCompleted/job_stats*'):
        filename = str(Path(filename).as_posix())
        ref = filename.split('/')[-1]
        phase6_jobs.remove(ref)
    return phase6_jobs


def check_phase_7(output_path, experiment_name, datasets=None):
    for filename in glob.glob(output_path + "/" + experiment_name + '/jobsCompleted/job_data_compare*'):
        filename = str(Path(filename).as_posix())
        if filename.split('/')[-1] == 'job_data_compare.txt':
            return []
        else:
            return ['job_data_compare.txt']
    return ['job_data_compare.txt']


def check_phase_8(output_path, experiment_name, datasets=None):
    # Make pdf summary for training analysis
    for filename in glob.glob(output_path + "/" + experiment_name
                              + '/jobsCompleted/job_data_pdf_training*'):
        filename = str(Path(filename).as_posix())
        if filename.split('/')[-1] == 'job_data_pdf_training.txt':
            return []
        else:
            return ['job_data_pdf_training.txt']
    return ['job_data_pdf_training.txt']


def check_phase_9(output_path, experiment_name, rep_data_path):
    phase9_jobs = []
    for dataset_filename in glob.glob(rep_data_path + '/*'):
        dataset_filename = str(Path(dataset_filename).as_posix())
        apply_name = dataset_filename.split('/')[-1].split('.')[0]
        phase9_jobs.append('job_apply_' + str(apply_name))
    for filename in glob.glob(output_path + "/" + experiment_name + '/jobsCompleted/job_apply*'):
        filename = str(Path(filename).as_posix())
        ref = filename.split('/')[-1].split('.')[0]
        phase9_jobs.remove(ref)
    return phase9_jobs


def check_phase_10(output_path, experiment_name, dataset_for_rep):
    # Make pdf summary for application analysis
    train_name = dataset_for_rep.split('/')[-1].split('.')[0]
    for filename in glob.glob(output_path + "/" + experiment_name
                              + '/jobsCompleted/job_data_pdf_apply_' + str(train_name) + '*'):
        filename = str(Path(filename).as_posix())
        if filename.split('/')[-1] == 'job_data_pdf_apply_' + str(train_name) + '.txt':
            return []
        else:
            return ['job_data_pdf_apply_' + str(train_name) + '.txt']
    return ['job_data_pdf_apply_' + str(train_name) + '.txt']


def check_phase_11(output_path, experiment_name):
    # Check if clean job is done
    not_deleted = list(glob.glob(output_path + "/" + experiment_name + '/jobsCompleted/*')) + \
                    list(glob.glob(output_path + "/" + experiment_name + '/jobs/*'))
    not_deleted = [str(Path(path)) for path in not_deleted]
    return not_deleted


FN_LIST = [check_phase_1, check_phase_2, check_phase_3, check_phase_4,
           check_phase_5, check_phase_6, check_phase_7, check_phase_8,
           check_phase_9, check_phase_10, check_phase_11]


def check_phase(output_path, experiment_name, phase=5, len_only=True,
                rep_data_path=None, dataset_for_rep=None, output=True):
    datasets = os.listdir(output_path + "/" + experiment_name)
    remove_list = ['.DS_Store', 'metadata.pickle', 'metadata.csv', 'algInfo.pickle',
                   'jobsCompleted', 'dask_logs', 'logs', 'jobs',
                   'DatasetComparisons', 'UsefulNotebooks',
                   experiment_name + '_STREAMLINE_Report.pdf']
    for text in remove_list:
        if text in datasets:
            datasets.remove(text)

    if phase < 9:
        phase_jobs = FN_LIST[phase - 1](output_path, experiment_name, datasets)
    elif phase == 9:
        phase_jobs = FN_LIST[phase - 1](output_path, experiment_name, rep_data_path)
    elif phase == 10:
        phase_jobs = FN_LIST[phase - 1](output_path, experiment_name, dataset_for_rep)
    elif phase == 11:
        phase_jobs = FN_LIST[phase - 1](output_path, experiment_name)
    else:
        raise Exception("Unknown Phase")

    if output:
        if len(phase_jobs) == 0:
            print("All Phase " + str(phase) + " Jobs Completed")
        elif len_only:
            print(str(len(phase_jobs)) + " Phase " + str(phase) + " Jobs Left")
        else:
            print("Below Phase " + str(phase) + " Jobs Not Completed:")
            for job in phase_jobs:
                print(job)
    return phase_jobs
