from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster, LSFCluster, SGECluster
from dask_jobqueue import HTCondorCluster, MoabCluster, OARCluster, PBSCluster

cluster_dict = {
    'HTCondor': HTCondorCluster,
    'LSF': LSFCluster,
    'Moab': MoabCluster,
    'OAR': OARCluster,
    'PBS': PBSCluster,
    'SGE': SGECluster,
    'SLURM': SLURMCluster,
    'Local': LocalCluster
}


def get_cluster(cluster_type='SLURM', output_path=".", queue='defq', memory=4):
    client = None
    try:
        if cluster_type == 'SLURM':
            cluster = SLURMCluster(queue=queue,
                                   cores=1,
                                   memory=str(memory) + "G",
                                   walltime="24:00:00",
                                   log_directory=output_path + "/dask_logs/")
            cluster.adapt(maximum_jobs=400)
        elif cluster_type == "LSF":
            cluster = LSFCluster(queue=queue,
                                 cores=1,
                                 memory=str(memory) + "G",
                                 walltime="24:00",
                                 log_directory=output_path + "/dask_logs/")
            cluster.adapt(maximum_jobs=400)
        elif cluster_type == 'UGE':
            cluster = SGECluster(queue=queue,
                                 cores=1,
                                 memory=str(memory) + "G",
                                 resource_spec="mem_free=" + str(memory) + "G",
                                 walltime="24:00:00",
                                 log_directory=output_path + "/dask_logs/")
            cluster.adapt(maximum_jobs=400)
        elif cluster_type == 'HTCondor':
            cluster = HTCondorCluster(cores=1,
                                      disk=str(memory) + "G",
                                      memory=str(memory) + "G",
                                      log_directory=output_path + "/dask_logs/")
            cluster.adapt(maximum_jobs=400)
        else:
            try:
                cluster = cluster_dict[cluster_type](queue=queue,
                                                     cores=1,
                                                     memory=str(memory) + "G",
                                                     walltime="24:00:00",
                                                     log_directory=output_path + "/dask_logs/")
                cluster.adapt(maximum_jobs=400)
            except KeyError:
                raise Exception("Unknown or Unsupported Cluster Type")
        client = Client(cluster)
    except Exception as e:
        print(e)
        raise Exception("Exception: Unknown Exception")
    # print("Running dask-cluster")
    # print(client.scheduler_info())
    return client
