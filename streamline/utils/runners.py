def parallel_eda_call(eda_job, params):
    """
    Runner function for running eda job objects
    """
    if params and 'top_features' in params:
        eda_job.run(params['top_features'])
    else:
        eda_job.run()


def parallel_kfold_call(kfold_job):
    """
    Runner function for running cv job objects
    """
    kfold_job.run()


def runner_fn(job):
    """
    Runner function for running job objects
    """
    job.run()


def run_jobs(job_list):
    """
    Function to start and join a list of job objects
    """
    for j in job_list:
        j.start()
    for j in job_list:
        j.join()
