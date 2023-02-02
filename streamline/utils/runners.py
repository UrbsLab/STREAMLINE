def parallel_eda_call(eda_job, params):
    if params and 'top_features' in params:
        eda_job.run(params['top_features'])
    else:
        eda_job.run()


def parallel_kfold_call(kfold_job):
    kfold_job.run()


def run_jobs(job_list):
    for j in job_list:
        j.start()
    for j in job_list:
        j.join()