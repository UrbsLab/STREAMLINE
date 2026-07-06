import os
import multiprocessing

from joblib import Parallel, delayed

num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', -1))
if num_cores == -1:
    num_cores = multiprocessing.cpu_count()
        

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


def progress_is_enabled():
    return str(os.environ.get("STREAMLINE_PROGRESS", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def progress_bar(iterable, total=None, label=None):
    if not progress_is_enabled():
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, total=total, desc=label, unit="job")


def run_dask_tasks(tasks, client, label=None):
    task_list = list(tasks)
    if not task_list:
        return []
    futures = client.compute(task_list)
    if progress_is_enabled():
        try:
            from dask.distributed import progress
            if label:
                print(label)
            progress(futures, notebook=False)
        except Exception:
            pass
    return client.gather(futures)


def parallel_worker_count(total_jobs=None):
    workers_raw = os.environ.get("STREAMLINE_PARALLEL_WORKERS")
    try:
        workers = int(workers_raw) if workers_raw else int(num_cores)
    except (TypeError, ValueError):
        workers = int(num_cores)
    workers = max(1, workers)
    if total_jobs is not None:
        workers = min(workers, max(1, int(total_jobs)))
    return workers


def run_parallel_jobs(function, jobs, workers=None, label="STREAMLINE Parallel"):
    """
    Run tuple-argument jobs with local joblib parallelism.
    """
    job_list = list(jobs)
    if not job_list:
        return []
    worker_count = workers if workers is not None else parallel_worker_count(len(job_list))
    worker_count = max(1, min(int(worker_count), len(job_list)))
    if worker_count == 1:
        return [function(*job) for job in progress_bar(job_list, total=len(job_list), label=label)]
    results = Parallel(n_jobs=worker_count, backend="loky", return_as="generator")(
        delayed(function)(*job) for job in job_list
    )
    return list(progress_bar(results, total=len(job_list), label=label))


def run_parallel_items(function, items, workers=None, label="STREAMLINE Parallel"):
    """
    Run one-argument jobs with local joblib parallelism.
    """
    item_list = list(items)
    if not item_list:
        return []
    worker_count = workers if workers is not None else parallel_worker_count(len(item_list))
    worker_count = max(1, min(int(worker_count), len(item_list)))
    if worker_count == 1:
        return [function(item) for item in progress_bar(item_list, total=len(item_list), label=label)]
    results = Parallel(n_jobs=worker_count, backend="loky", return_as="generator")(
        delayed(function)(item) for item in item_list
    )
    return list(progress_bar(results, total=len(item_list), label=label))


def run_callable(callable_obj):
    return callable_obj()


def run_parallel_functions(functions, workers=None, label="STREAMLINE Parallel"):
    """
    Run zero-argument callables with local joblib parallelism.
    """
    return run_parallel_items(run_callable, functions, workers=workers, label=label)


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
