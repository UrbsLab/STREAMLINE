import os
import multiprocessing

from joblib import Parallel, delayed

num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', -1))
if num_cores == -1:
    num_cores = multiprocessing.cpu_count()


def check_if_single_phase(params):
    phase_list = [params['do_eda'], params['do_dataprep'], params['do_feat_imp'], 
                    params['do_feat_sel'], params['do_model'], params['do_stats'], 
                    params['do_compare_dataset'], params['do_report'], params['do_replicate'], 
                    params['do_rep_report'], params['do_cleanup']]
    phase_count = 0
    for phase in phase_list:
        if phase:
            phase_count += 1
    if phase_count == 1:
        return True
    else:
        return False
        

def parallel_eda_call(eda_job, params):
    """
    Runner function for running eda job objects
    """
    if params and 'top_features' in params:
        eda_job.run(params['top_features'])
    else:
        eda_job.run()


def model_runner_fn(job, model):
    """
    Runner function for running model job objects
    """
    job.run(model)


def runner_fn(job):
    """
    Runner function for running job objects
    """
    job.run()


def run_jobs(job_list):
    """
    Function to start and join a list of job objects
    """
    for i in range(0, len(job_list), num_cores):
        sub_jobs(job_list[i:i + num_cores])


def sub_jobs(job_list):
    Parallel(n_jobs=num_cores)(
        delayed(model_runner_fn)(job_obj, model
                                 ) for job_obj, model in job_list)
