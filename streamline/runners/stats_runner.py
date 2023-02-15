import os
import glob
import dask
import pickle
from joblib import Parallel, delayed
from streamline.modeling.utils import SUPPORTED_MODELS
from streamline.modeling.utils import is_supported_model
from streamline.postanalysis.statistics import StatsJob
from streamline.utils.runners import runner_fn, num_cores
from streamline.utils.cluster import get_cluster


class StatsRunner:
    """
    Runner Class for collating statistics of all the models
    """

    def __init__(self, output_path, experiment_name, algorithms=None, exclude=("XCS", "eLCS"),
                 class_label="Class", instance_label=None, scoring_metric='balanced_accuracy',
                 top_features=40, sig_cutoff=0.05, metric_weight='balanced_accuracy', scale_data=True,
                 plot_roc=True, plot_prc=True, plot_fi_box=True, plot_metric_boxplots=True, show_plots=False):
        """
        Args:
            output_path: path to output directory
            experiment_name: name of experiment (no spaces)
            algorithms: list of str of ML models to run
            scoring_metric='balanced_accuracy'
            sig_cutoff: significance cutoff, default=0.05
            metric_weight='balanced_accuracy'
            scale_data=True
            plot_roc: Plot ROC curves individually for each algorithm including all CV results and averages,
                                default=True
            plot_prc: Plot PRC curves individually for each algorithm including all CV results and averages,
                                default=True
            plot_metric_boxplots: Plot box plot summaries comparing algorithms for each metric, default=True
            plot_fi_box: Plot feature importance boxplots and histograms for each algorithm, default=True
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
        self.class_label = class_label
        self.instance_label = instance_label

        if algorithms is None:
            self.algorithms = SUPPORTED_MODELS
            if exclude is not None:
                for algorithm in exclude:
                    try:
                        self.algorithms.remove(algorithm)
                    except Exception:
                        Exception("Unknown algorithm in exclude: " + str(algorithm))
        else:
            self.algorithms = list()
            for algorithm in algorithms:
                self.algorithms.append(is_supported_model(algorithm))

        self.scale_data = scale_data
        self.sig_cutoff = sig_cutoff
        self.show_plots = show_plots
        self.scoring_metric = scoring_metric

        self.plot_roc = plot_roc
        self.plot_prc = plot_prc
        self.plot_metric_boxplots = plot_metric_boxplots
        self.plot_fi_box = plot_fi_box
        self.metric_weight = metric_weight
        self.top_features = top_features

        # Argument checks
        if not os.path.exists(self.output_path):
            raise Exception("Output path must exist (from phase 1) before phase 6 can begin")
        if not os.path.exists(self.output_path + '/' + self.experiment_name):
            raise Exception("Experiment must exist (from phase 1) before phase 6 can begin")

        self.save_metadata()

    def run(self, run_parallel):

        # Iterate through datasets, ignoring common folders
        dataset_paths = os.listdir(self.output_path + "/" + self.experiment_name)
        remove_list = ['metadata.pickle', 'metadata.csv', 'algInfo.pickle', 'jobsCompleted',
                       'logs', 'jobs', 'DatasetComparisons',
                       self.experiment_name + '_ML_Pipeline_Report.pdf']

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
            cv_partitions = len(cv_dataset_paths)
            job_obj = StatsJob(full_path, self.algorithms, self.class_label, self.instance_label, self.scoring_metric,
                               cv_partitions, self.top_features, self.sig_cutoff, self.metric_weight, self.scale_data,
                               self.plot_roc, self.plot_prc, self.plot_fi_box, self.plot_metric_boxplots,
                               self.show_plots)
            if run_parallel and run_parallel != "False":
                # p = multiprocessing.Process(target=runner_fn, args=(job_obj, ))
                job_list.append(job_obj)
            else:
                job_obj.run()
        if run_parallel and (run_parallel in ["multiprocessing", "True", True]):
            Parallel(n_jobs=num_cores)(delayed(runner_fn)(job_obj) for job_obj in job_list)
        if run_parallel and (run_parallel not in ["multiprocessing", "True", True, "False"]):
            get_cluster(run_parallel)
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
