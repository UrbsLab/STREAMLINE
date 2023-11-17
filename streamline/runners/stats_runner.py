import logging
import os
import glob
import time
import dask
import pickle
from pathlib import Path
from joblib import Parallel, delayed
from streamline.postanalysis.statistics import StatsJob
from streamline.utils.runners import runner_fn, num_cores
from streamline.utils.cluster import get_cluster


class StatsRunner:
    """
    Runner Class for collating statistics of all the models
    """

    def __init__(self, output_path, experiment_name,
                 outcome_label="Class", outcome_type=None, instance_label=None,
                 scoring_metric='balanced_accuracy',
                 top_features=40, sig_cutoff=0.05, metric_weight='balanced_accuracy', scale_data=True,
                 exclude_plots=None, show_plots=False,
                 run_cluster=False, queue='defq', reserved_memory=4):
        """
        Args:
            output_path: path to output directory
            experiment_name: name of experiment (no spaces)
            scoring_metric='balanced_accuracy'
            sig_cutoff: significance cutoff, default=0.05
            metric_weight='balanced_accuracy'
            scale_data=True
            exclude_plots:
            metric_weight: ML model metric used as weight in composite FI plots \
                           (only supports balanced_accuracy or roc_auc as options). \
                           Recommend setting the same as primary_metric if possible, \
                           default='balanced_accuracy'
            top_features: number of top features to illustrate in figures, default=40
            show_plots: flag to show plots

        """
        self.dataset = None
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.outcome_label = outcome_label
        self.instance_label = instance_label

        self.outcome_type = outcome_type
        if self.outcome_type is None:
            file = open(self.output_path + '/' + self.experiment_name + '/' + "metadata.pickle", 'rb')
            metadata = pickle.load(file)
            self.outcome_type = metadata['Outcome Type']
            file.close()

        self.algorithms = None
        self.get_algorithms()

        self.algorithms = sorted(self.algorithms)

        self.scale_data = scale_data
        self.sig_cutoff = sig_cutoff
        self.show_plots = show_plots
        self.scoring_metric = scoring_metric
        self.exclude_plots = exclude_plots

        known_exclude_options = ['plot_ROC', 'plot_PRC', 'plot_FI_box', 'plot_metric_boxplots']
        if exclude_plots is not None:
            for x in exclude_plots:
                if x not in known_exclude_options:
                    logging.warning("Unknown exclusion option " + str(x))
        else:
            exclude_plots = list()

        self.plot_roc = 'plot_ROC' not in exclude_plots
        self.plot_prc = 'plot_PRC' not in exclude_plots
        self.plot_metric_boxplots = 'plot_metric_boxplots' not in exclude_plots
        self.plot_fi_box = 'plot_FI_box' not in exclude_plots
        self.metric_weight = metric_weight
        self.top_features = top_features

        self.run_cluster = run_cluster
        self.queue = queue
        self.reserved_memory = reserved_memory

        # Argument checks
        if not os.path.exists(self.output_path):
            raise Exception("Output path must exist (from phase 1) before phase 6 can begin")
        if not os.path.exists(self.output_path + '/' + self.experiment_name):
            raise Exception("Experiment must exist (from phase 1) before phase 6 can begin")

        self.save_metadata()

    def run(self, run_parallel=False):

        # Iterate through datasets, ignoring common folders
        dataset_paths = os.listdir(self.output_path + "/" + self.experiment_name)
        remove_list = ['.DS_Store', 'metadata.pickle', 'metadata.csv', 'algInfo.pickle', 'jobsCompleted', 'dask_logs',
                       'logs', 'jobs', 'DatasetComparisons',
                       self.experiment_name + '_STREAMLINE_Report.pdf']

        for text in remove_list:
            if text in dataset_paths:
                dataset_paths.remove(text)

        job_list = list()
        for dataset_directory_path in dataset_paths:
            full_path = self.output_path + "/" + self.experiment_name + "/" + dataset_directory_path

            # Create folders for DT and GP visualizations
            if "DT" in self.algorithms and not os.path.exists(full_path + '/model_evaluation/DT_Viz'):
                os.mkdir(full_path + '/model_evaluation/DT_Viz')
            if "GP" in self.algorithms and not os.path.exists(full_path + '/model_evaluation/GP_Viz'):
                os.mkdir(full_path + '/model_evaluation/GP_Viz')

            cv_dataset_paths = list(glob.glob(full_path + "/CVDatasets/*_CV_*Train.csv"))
            cv_dataset_paths = [str(Path(cv_dataset_path)) for cv_dataset_path in cv_dataset_paths]
            cv_partitions = len(cv_dataset_paths)

            if self.run_cluster == "SLURMOld":
                self.submit_slurm_cluster_job(full_path, cv_partitions)
                continue

            if self.run_cluster == "LSFOld":
                self.submit_lsf_cluster_job(full_path, cv_partitions)
                continue

            job_obj = StatsJob(full_path, self.outcome_label, self.outcome_type, self.instance_label,
                               self.scoring_metric,
                               cv_partitions, self.top_features, self.sig_cutoff, self.metric_weight, self.scale_data,
                               self.exclude_plots,
                               self.show_plots)
            if run_parallel and run_parallel != "False":
                # p = multiprocessing.Process(target=runner_fn, args=(job_obj, ))
                job_list.append(job_obj)
            else:
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
        metadata['Export ROC Plot'] = self.plot_roc
        metadata['Export PRC Plot'] = self.plot_prc
        metadata['Export Metric Boxplots'] = self.plot_metric_boxplots
        metadata['Export Feature Importance Boxplots'] = self.plot_fi_box
        metadata['Metric Weighting Composite FI Plots'] = self.metric_weight
        metadata['Top Model Features To Display'] = self.top_features
        # Pickle the metadata for future use
        pickle_out = open(self.output_path + '/' + self.experiment_name + '/' + "metadata.pickle", 'wb')
        pickle.dump(metadata, pickle_out)
        pickle_out.close()

    def get_algorithms(self):
        pickle_in = open(self.output_path + '/' + self.experiment_name + '/' + "algInfo.pickle", 'rb')
        alg_info = pickle.load(pickle_in)
        algorithms = list()
        for algorithm in alg_info.keys():
            if alg_info[algorithm][0]:
                algorithms.append(algorithm)
        self.algorithms = algorithms
        pickle_in.close()

    def get_cluster_params(self, full_path, len_cv):
        exclude_param = ','.join(self.exclude_plots) if self.exclude_plots else None
        cluster_params = [full_path, self.outcome_label, self.outcome_type, self.instance_label,
                          self.scoring_metric,
                          len_cv, self.top_features, self.sig_cutoff, self.metric_weight, self.scale_data,
                          exclude_param,
                          self.show_plots]
        cluster_params = [str(i) for i in cluster_params]
        return cluster_params

    def submit_slurm_cluster_job(self, dataset_path, len_cv):
        job_ref = str(time.time())
        job_name = self.output_path + '/' + self.experiment_name + '/jobs/P6_' + job_ref + '_run.sh'
        sh_file = open(job_name, 'w')
        sh_file.write('#!/bin/bash\n')
        sh_file.write('#SBATCH -p ' + self.queue + '\n')
        sh_file.write('#SBATCH --job-name=' + job_ref + '\n')
        sh_file.write('#SBATCH --mem=' + str(self.reserved_memory) + 'G' + '\n')
        # sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
        sh_file.write(
            '#SBATCH -o ' + self.output_path + '/' + self.experiment_name +
            '/logs/P6_' + job_ref + '.o\n')
        sh_file.write(
            '#SBATCH -e ' + self.output_path + '/' + self.experiment_name +
            '/logs/P6_' + job_ref + '.e\n')

        file_path = str(Path(__file__).parent.parent.parent) + "/streamline/legacy" + '/StatsJobSubmit.py'
        cluster_params = self.get_cluster_params(dataset_path, len_cv)
        command = ' '.join(['srun', 'python', file_path] + cluster_params)
        sh_file.write(command + '\n')
        sh_file.close()
        os.system('sbatch ' + job_name)

    def submit_lsf_cluster_job(self, dataset_path, len_cv):
        job_ref = str(time.time())
        job_name = self.output_path + '/' + self.experiment_name + '/jobs/P6_' + job_ref + '_run.sh'
        sh_file = open(job_name, 'w')
        sh_file.write('#!/bin/bash\n')
        sh_file.write('#BSUB -q ' + self.queue + '\n')
        sh_file.write('#BSUB -J ' + job_ref + '\n')
        sh_file.write('#BSUB -R "rusage[mem=' + str(self.reserved_memory) + 'G]"' + '\n')
        sh_file.write('#BSUB -M ' + str(self.reserved_memory) + 'GB' + '\n')
        sh_file.write(
            '#BSUB -o ' + self.output_path + '/' + self.experiment_name +
            '/logs/P6_' + job_ref + '.o\n')
        sh_file.write(
            '#BSUB -e ' + self.output_path + '/' + self.experiment_name +
            '/logs/P6_' + job_ref + '.e\n')

        file_path = str(Path(__file__).parent.parent.parent) + "/streamline/legacy" + '/StatsJobSubmit.py'
        cluster_params = self.get_cluster_params(dataset_path, len_cv)
        command = ' '.join(['python', file_path] + cluster_params)
        sh_file.write(command + '\n')
        sh_file.close()
        os.system('bsub < ' + job_name)
