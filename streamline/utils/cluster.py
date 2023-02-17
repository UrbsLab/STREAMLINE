from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster, LSFCluster, SGECluster, PBSCluster

cluster_dict = {
    'SLURM': SLURMCluster,
    'LSF': LSFCluster,
    'SGE': SGECluster,
    'PBS': PBSCluster,
    'Local': LocalCluster
}


def get_cluster(cluster_type='SLURM', output_path=".", queue='defq', memory=4):
    client = None
    try:
        if cluster_type == 'SLURM':
            cluster = SLURMCluster(queue=queue,
                                   cores=1,
                                   memory=str(memory) + "G",
                                   walltime="12:00:00",
                                   log_directory=output_path + "/dask_logs/")
            cluster.adapt(maximum_jobs=400)
        elif cluster_type == "LSF":
            cluster = LSFCluster(queue='i2c2_normal',
                                 cores=1,
                                 memory="4G",
                                 walltime="12:00:00",
                                 log_directory=output_path + "/dask_logs/")
            cluster.adapt(maximum_jobs=400)
        else:
            try:
                cluster = cluster_dict[cluster_type](queue=queue,
                                                     cores=1,
                                                     memory=str(memory) + "G",
                                                     walltime="12:00:00",
                                                     log_directory=output_path + "/dask_logs/")
                cluster.adapt(maximum_jobs=400)
            except KeyError:
                raise Exception("Unknown or Unsupported Cluster Type")

        client = Client(cluster)
    except Exception:
        raise Exception("Exception: Unknown Exception")
    return client
