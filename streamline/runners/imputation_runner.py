import os
import glob
import pickle
import time
import dask
from pathlib import Path
from joblib import Parallel, delayed
from streamline.dataprep.scale_and_impute import ScaleAndImpute
from streamline.utils.runners import runner_fn, num_cores
from streamline.utils.cluster import get_cluster


class ImputationRunner:
    """
    Runner class for Data Processing Jobs of CV Splits
    """

    def __init__(self, output_path, experiment_name, scale_data=True, impute_data=True,
                 multi_impute=True, overwrite_cv=True, outcome_label="Class", instance_label=None, random_state=None,
                 run_cluster=False, queue='defq', reserved_memory=4):
        """

        Args:
            output_path:
            experiment_name:
            scale_data:
            impute_data:
            multi_impute:
            overwrite_cv:
        """
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.scale_data = scale_data
        self.impute_data = impute_data
        self.multi_impute = multi_impute
        self.overwrite_cv = overwrite_cv
        self.outcome_label = outcome_label
        self.instance_label = instance_label
        self.random_state = random_state

        self.run_cluster = run_cluster
        self.queue = queue
        self.reserved_memory = reserved_memory

        # Argument checks-------------------------------------------------------------
        if not os.path.exists(self.output_path):
            raise Exception("Output path must exist (from phase 1) before phase 2 can begin")
        if not os.path.exists(self.output_path + '/' + self.experiment_name):
            raise Exception("Experiment must exist (from phase 1) before phase 2 can begin")

        self.save_metadata()

    def run(self, run_parallel=False):
        job_counter = 0
        job_list = []
        dataset_paths = os.listdir(self.output_path + "/" + self.experiment_name)
        remove_list = ['.DS_Store', 'metadata.pickle', 'metadata.csv', 'algInfo.pickle', 'jobsCompleted', 'dask_logs',
                       'logs', 'jobs', 'DatasetComparisons']
        for text in remove_list:
            if text in dataset_paths:
                dataset_paths.remove(text)

        for dataset_directory_path in dataset_paths:
            full_path = self.output_path + "/" + self.experiment_name + "/" + dataset_directory_path

            # Create folder to store scaling and imputing files
            if not os.path.exists(full_path + '/scale_impute/'):
                os.makedirs(full_path + '/scale_impute/')

            for cv_train_path in glob.glob(full_path + "/CVDatasets/*Train.csv"):
                cv_train_path = str(Path(cv_train_path).as_posix())
                job_counter += 1
                cv_test_path = cv_train_path.replace("Train.csv", "Test.csv")

                if self.run_cluster == "SLURMOld":
                    self.submit_slurm_cluster_job(cv_train_path, cv_test_path)
                    continue

                if self.run_cluster == "LSFOld":
                    self.submit_lsf_cluster_job(cv_train_path, cv_test_path)
                    continue

                if run_parallel and run_parallel != "False":
                    job_obj = ScaleAndImpute(cv_train_path, cv_test_path,
                                             self.output_path + "/" + self.experiment_name,
                                             self.scale_data, self.impute_data, self.multi_impute, self.overwrite_cv,
                                             self.outcome_label, self.instance_label, self.random_state)
                    # p = multiprocessing.Process(target=runner_fn, args=(job_obj, ))
                    job_list.append(job_obj)
                else:
                    job_obj = ScaleAndImpute(cv_train_path, cv_test_path,
                                             self.output_path + "/" + self.experiment_name,
                                             self.scale_data, self.impute_data, self.multi_impute, self.overwrite_cv,
                                             self.outcome_label, self.instance_label, self.random_state)
                    job_obj.run()
        if run_parallel and run_parallel != "False" and not self.run_cluster:
            Parallel(n_jobs=num_cores)(delayed(runner_fn)(job_obj) for job_obj in job_list)
        if self.run_cluster and "Old" not in self.run_cluster:
            get_cluster(self.run_cluster,
                        self.output_path + '/' + self.experiment_name, self.queue, self.reserved_memory)
            dask.compute([dask.delayed(runner_fn)(job_obj) for job_obj in job_list])

    def save_metadata(self):
        file = open(self.output_path + '/' + self.experiment_name + '/' + "metadata.pickle", 'rb')
        metadata = pickle.load(file)
        file.close()
        metadata['Use Data Scaling'] = self.scale_data
        metadata['Use Data Imputation'] = self.impute_data
        metadata['Use Multivariate Imputation'] = self.multi_impute
        # Pickle the metadata for future use
        pickle_out = open(self.output_path + '/' + self.experiment_name + '/' + "metadata.pickle", 'wb')
        pickle.dump(metadata, pickle_out)
        pickle_out.close()

    def get_cluster_params(self, cv_train_path, cv_test_path):
        cluster_params = [cv_train_path, cv_test_path,
                          self.output_path + "/" + self.experiment_name,
                          self.scale_data, self.impute_data, self.multi_impute, self.overwrite_cv,
                          self.outcome_label, self.instance_label, self.random_state]
        cluster_params = [str(i) for i in cluster_params]
        return cluster_params

    def submit_slurm_cluster_job(self, cv_train_path, cv_test_path):
        job_ref = str(time.time())
        job_name = self.output_path + '/' + self.experiment_name + '/jobs/P1_' + job_ref + '_run.sh'
        sh_file = open(job_name, 'w')
        sh_file.write('#!/bin/bash\n')
        sh_file.write('#SBATCH -p ' + self.queue + '\n')
        sh_file.write('#SBATCH --job-name=' + job_ref + '\n')
        sh_file.write('#SBATCH --mem=' + str(self.reserved_memory) + 'G' + '\n')
        # sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
        sh_file.write(
            '#SBATCH -o ' + self.output_path + '/' + self.experiment_name +
            '/logs/P1_' + job_ref + '.o\n')
        sh_file.write(
            '#SBATCH -e ' + self.output_path + '/' + self.experiment_name +
            '/logs/P1_' + job_ref + '.e\n')

        file_path = str(Path(__file__).parent.parent.parent) + "/streamline/legacy" + '/DataJobSubmit.py'
        cluster_params = self.get_cluster_params(cv_train_path, cv_test_path)
        command = ' '.join(['srun', 'python', file_path] + cluster_params)
        sh_file.write(command + '\n')
        sh_file.close()
        os.system('sbatch ' + job_name)

    def submit_lsf_cluster_job(self, cv_train_path, cv_test_path):
        job_ref = str(time.time())
        job_name = self.output_path + '/' + self.experiment_name + '/jobs/P2_' + job_ref + '_run.sh'
        sh_file = open(job_name, 'w')
        sh_file.write('#!/bin/bash\n')
        sh_file.write('#BSUB -q ' + self.queue + '\n')
        sh_file.write('#BSUB -J ' + job_ref + '\n')
        sh_file.write('#BSUB -R "rusage[mem=' + str(self.reserved_memory) + 'G]"' + '\n')
        sh_file.write('#BSUB -M ' + str(self.reserved_memory) + 'GB' + '\n')
        sh_file.write(
            '#BSUB -o ' + self.output_path + '/' + self.experiment_name +
            '/logs/P2_' + job_ref + '.o\n')
        sh_file.write(
            '#BSUB -e ' + self.output_path + '/' + self.experiment_name +
            '/logs/P2_' + job_ref + '.e\n')

        file_path = str(Path(__file__).parent.parent.parent) + "/streamline/legacy" + '/DataJobSubmit.py'
        cluster_params = self.get_cluster_params(cv_train_path, cv_test_path)
        command = ' '.join(['python', file_path] + cluster_params)
        sh_file.write(command + '\n')
        sh_file.close()
        os.system('bsub < ' + job_name)
