from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster, LSFCluster, SGECluster, PBSCluster

cluster_dict = {
    'SLURM': SLURMCluster,
    'LSF': LSFCluster,
    'SGE': SGECluster,
    'PBS': PBSCluster,
    'Local': LocalCluster
}


def get_cluster(cluster_type, output_path="."):
    client = None
    try:
        cluster = cluster_dict[cluster_type](queue='defq',
                                             cores=1,
                                             memory="10 GB",
                                             walltime="24:00:00",
                                             log_directory=output_path + "/dask_logs/")
        cluster.adapt(maximum_jobs=400)
        client = Client(cluster)
    except Exception:
        raise Exception("Exception: Unknown Type")
    return client
