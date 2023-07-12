import os
import pickle
import time
import dask
from pathlib import Path
from joblib import Parallel, delayed
from streamline.postanalysis.gererate_report import ReportJob
from streamline.utils.runners import runner_fn
from streamline.utils.cluster import get_cluster


class ReportRunner:
    """
    Runner Class for collating dataset compare job
    """

    def __init__(self, output_path=None, experiment_name=None, experiment_path=None,
                 training=True, rep_data_path=None, dataset_for_rep=None,
                 run_cluster=False, queue='defq', reserved_memory=4):
        """
        Args:
            output_path: path to output directory
            experiment_name: name of experiment (no spaces)
            training: Indicate True or False for whether to generate pdf summary for pipeline \
                      training or followup application analysis to new dataset,default=True
            rep_data_path: path to directory containing replication or hold-out testing datasets \
                           (must have at least all features with same labels as in
                           original training dataset),default=None
            dataset_for_rep: path to target original training dataset

        """
        assert (output_path is not None and experiment_name is not None) or (experiment_path is not None)
        if output_path is not None and experiment_name is not None:
            self.output_path = output_path
            self.experiment_name = experiment_name
            self.experiment_path = self.output_path + '/' + self.experiment_name
        else:
            self.experiment_path = experiment_path
            self.experiment_name = self.experiment_path.split('/')[-1]
            self.output_path = self.experiment_path.split('/')[-2]

        self.training = training
        self.rep_data_path = rep_data_path
        self.train_data_path = dataset_for_rep

        self.run_cluster = run_cluster
        self.queue = queue
        self.reserved_memory = reserved_memory
        self.algorithms = None
        self.get_algorithms()

        # Argument checks
        if not os.path.exists(self.output_path):
            raise Exception("Output path must exist (from phase 1) before phase 6 can begin")
        if not os.path.exists(self.output_path + '/' + self.experiment_name):
            raise Exception("Experiment must exist (from phase 1) before phase 6 can begin")

    def run(self, run_parallel=False):

        if self.run_cluster in ["SLURMOld", "LSFOld"]:
            if self.run_cluster == "SLURMOld":
                self.submit_slurm_cluster_job()

            if self.run_cluster == "LSFOld":
                self.submit_lsf_cluster_job()
        else:
            job_obj = ReportJob(self.output_path, self.experiment_name, None, self.algorithms, None,
                                self.training, self.train_data_path, self.rep_data_path)
            # running direct because it's faster
            HACK = not run_parallel
            if not HACK:
                if run_parallel and run_parallel != "False" and not self.run_cluster:
                    # p = multiprocessing.Process(target=runner_fn, args=(job_obj, ))
                    # p.start()
                    # p.join()
                    Parallel()(delayed(runner_fn)(job_obj) for job_obj in [job_obj, ])
                elif self.run_cluster and "Old" not in self.run_cluster:
                    get_cluster(self.run_cluster, self.output_path + '/' + self.experiment_name,
                                self.queue, self.reserved_memory)
                    dask.compute([dask.delayed(runner_fn)(job_obj) for job_obj in [job_obj, ]])
                else:
                    job_obj.run()

            else:
                job_obj.run()

    def get_algorithms(self):
        pickle_in = open(self.output_path + '/' + self.experiment_name + '/' + "algInfo.pickle", 'rb')
        alg_info = pickle.load(pickle_in)
        algorithms = list()
        for algorithm in alg_info.keys():
            if alg_info[algorithm][0]:
                algorithms.append(algorithm)
        self.algorithms = algorithms
        pickle_in.close()

    def get_cluster_params(self):
        cluster_params = [self.output_path, self.experiment_name, None, None, None,
                          self.training, self.train_data_path, self.rep_data_path]
        cluster_params = [str(i) for i in cluster_params]
        return cluster_params

    def submit_slurm_cluster_job(self):
        job_ref = str(time.time())
        job_name = self.output_path + '/' + self.experiment_name + '/jobs/PDF_' + job_ref + '_run.sh'
        sh_file = open(job_name, 'w')
        sh_file.write('#!/bin/bash\n')
        sh_file.write('#SBATCH -p ' + self.queue + '\n')
        sh_file.write('#SBATCH --job-name=' + job_ref + '\n')
        sh_file.write('#SBATCH --mem=' + str(self.reserved_memory) + 'G' + '\n')
        # sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
        sh_file.write(
            '#SBATCH -o ' + self.output_path + '/' + self.experiment_name +
            '/logs/PDF_' + job_ref + '.o\n')
        sh_file.write(
            '#SBATCH -e ' + self.output_path + '/' + self.experiment_name +
            '/logs/PDF_' + job_ref + '.e\n')

        file_path = str(Path(__file__).parent.parent.parent) + "/streamline/legacy" + '/ReportJobSubmit.py'
        cluster_params = self.get_cluster_params()
        command = ' '.join(['srun', 'python', file_path] + cluster_params)
        sh_file.write(command + '\n')
        sh_file.close()
        os.system('sbatch ' + job_name)

    def submit_lsf_cluster_job(self):
        job_ref = str(time.time())
        job_name = self.output_path + '/' + self.experiment_name + '/jobs/PDF_' + job_ref + '_run.sh'
        sh_file = open(job_name, 'w')
        sh_file.write('#!/bin/bash\n')
        sh_file.write('#BSUB -q ' + self.queue + '\n')
        sh_file.write('#BSUB -J ' + job_ref + '\n')
        sh_file.write('#BSUB -R "rusage[mem=' + str(self.reserved_memory) + 'G]"' + '\n')
        sh_file.write('#BSUB -M ' + str(self.reserved_memory) + 'GB' + '\n')
        sh_file.write(
            '#BSUB -o ' + self.output_path + '/' + self.experiment_name +
            '/logs/PDF_' + job_ref + '.o\n')
        sh_file.write(
            '#BSUB -e ' + self.output_path + '/' + self.experiment_name +
            '/logs/PDF_' + job_ref + '.e\n')

        file_path = str(Path(__file__).parent.parent.parent) + "/streamline/legacy" + '/ReportJobSubmit.py'
        cluster_params = self.get_cluster_params()
        command = ' '.join(['python', file_path] + cluster_params)
        sh_file.write(command + '\n')
        sh_file.close()
        os.system('bsub < ' + job_name)
