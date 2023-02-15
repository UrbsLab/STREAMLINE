import os
import glob
import pickle
import dask
from joblib import Parallel, delayed
from streamline.modeling.utils import SUPPORTED_MODELS, is_supported_model
from streamline.postanalysis.model_replicate import ReplicateJob
from streamline.utils.runners import num_cores, runner_fn
from streamline.utils.cluster import get_cluster


class ReplicationRunner:
    """
    Phase 9 of STREAMLINE (Optional)- This 'Main' script manages Phase 9 run parameters,
    and submits job to run locally (to run serially) or on
    cluster (parallelized). This script runs ApplyModelJob.py which applies and
    evaluates all trained models on one or more previously unseen hold-out or replication study dataset(s).
    """

    def __init__(self, rep_data_path, dataset_for_rep, output_path, experiment_name,
                 class_label=None, instance_label=None, match_label=None, algorithms=None, load_algo=True,
                 exclude=("XCS", "eLCS"),
                 export_feature_correlations=True, plot_roc=True, plot_prc=True, plot_metric_boxplots=True):
        """

        Args:
            rep_data_path: path to directory containing replication or \
                           hold-out testing datasets (must have at least all \
                           features with same labels as in original training dataset)
            dataset_for_rep: path to target original training dataset
            output_path: path to output directory
            experiment_name: name of experiment (no spaces)
            match_label: applies if original training data included column with matched instance ids, default=None
            export_feature_correlations: run and export feature correlation analysis (yields correlation heatmap), \
                                         default=True
            plot_roc: Plot ROC curves individually for each algorithm including all CV results and averages, \
                      default=True
            plot_prc: Plot PRC curves individually for each algorithm including all CV results and averages, \
                      default=True
            plot_metric_boxplots: Plot box plot summaries comparing algorithms for each metric, default=True
        """

        self.rep_data_path = rep_data_path
        self.dataset_for_rep = dataset_for_rep
        self.output_path = output_path
        self.experiment_name = experiment_name
        # Param for future expansion
        self.plot_lists = None
        self.match_label = match_label

        self.export_feature_correlations = export_feature_correlations
        self.plot_roc = plot_roc
        self.plot_prc = plot_prc
        self.plot_metric_boxplots = plot_metric_boxplots

        self.experiment_path = self.output_path + '/' + self.experiment_name

        # Save unique dataset names so that analysis is run only once if there is
        # both a .txt and .csv version of dataset with same name.
        self.data_name = self.dataset_for_rep.split('/')[-1].split('.')[0]

        # Unpickle metadata from previous phase
        file = open(self.output_path + '/' + self.experiment_name + '/' + "metadata.pickle", 'rb')
        metadata = pickle.load(file)
        file.close()
        # Load variables specified earlier in the pipeline from metadata
        self.class_label = class_label
        if not class_label:
            self.class_label = metadata['Class Label']
        self.instance_label = instance_label
        if not instance_label:
            self.instance_label = metadata['Instance Label']
        self.categorical_cutoff = metadata['Categorical Cutoff']
        self.sig_cutoff = metadata['Statistical Significance Cutoff']
        self.cv_partitions = metadata['CV Partitions']
        self.scale_data = metadata['Use Data Scaling']
        self.impute_data = metadata['Use Data Imputation']
        self.multi_impute = metadata['Use Multivariate Imputation']
        self.show_plots = False
        self.scoring_metric = metadata['Primary Metric']

        # Argument checks
        if not os.path.exists(self.output_path):
            raise Exception("Output path must exist (from phase 1-8) before model application can begin")
        if not os.path.exists(self.output_path + '/' + self.experiment_name):
            raise Exception("Experiment must exist (from phase 1-8) before model application can begin")

        # location of folder containing models respective training dataset
        self.full_path = self.output_path + "/" + self.experiment_name + "/" + self.data_name

        if not os.path.exists(self.full_path + "/applymodel"):
            os.mkdir(self.full_path + "/applymodel")

        if not self.show_plots:
            if not os.path.exists(self.output_path + "/" + self.experiment_name + '/jobs'):
                os.mkdir(self.output_path + "/" + self.experiment_name + '/jobs')
            if not os.path.exists(self.output_path + "/" + self.experiment_name + '/logs'):
                os.mkdir(self.output_path + "/" + self.experiment_name + '/logs')

        if not load_algo:
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
        else:
            self.get_algorithms()

        self.save_metadata()

    def run(self, run_parallel):
        # Determine file extension of datasets in target folder:
        file_count = 0
        unique_datanames = list()
        job_list = list()
        for dataset_filename in glob.glob(self.rep_data_path + '/*'):
            # Save unique dataset names so that analysis is run only once if
            # there is both a .txt and .csv version of dataset with same name.
            file_extension = dataset_filename.split('/')[-1].split('.')[-1]
            apply_name = dataset_filename.split('/')[-1].split('.')[0]

            if not os.path.exists(self.full_path + "/applymodel/" + apply_name):
                os.mkdir(self.full_path + "/applymodel/" + apply_name)

            if file_extension == 'txt' or file_extension == 'csv':
                if apply_name not in unique_datanames:
                    file_count += 1
                    unique_datanames.append(apply_name)
                    job_obj = ReplicateJob(dataset_filename,
                                           self.dataset_for_rep, self.full_path, self.class_label, self.instance_label,
                                           self.match_label, algorithms=self.algorithms, exclude=("XCS", "eLCS"),
                                           cv_partitions=self.cv_partitions,
                                           export_feature_correlations=self.export_feature_correlations,
                                           plot_roc=self.plot_roc, plot_prc=self.plot_prc,
                                           plot_metric_boxplots=self.plot_metric_boxplots,
                                           categorical_cutoff=self.categorical_cutoff,
                                           sig_cutoff=self.sig_cutoff, scale_data=self.scale_data,
                                           impute_data=self.impute_data,
                                           multi_impute=self.multi_impute, show_plots=self.show_plots,
                                           scoring_metric=self.scoring_metric)
                    if run_parallel or run_parallel != "False":
                        # p = multiprocessing.Process(target=runner_fn, args=(job_obj,))
                        job_list.append(job_obj)
                    else:
                        job_obj.run()
                if run_parallel and (run_parallel in ["multiprocessing", "True", True]):
                    Parallel(n_jobs=num_cores)(delayed(runner_fn)(job_obj) for job_obj in job_list)
                if run_parallel and (run_parallel not in ["multiprocessing", "True", True, "False"]):
                    get_cluster(run_parallel)
                    dask.compute([dask.delayed(runner_fn)(job_obj) for job_obj in job_list])
        if file_count == 0:
            # Check that there was at least 1 dataset
            raise Exception("There must be at least one .txt or .csv dataset in rep_data_path directory")

    def save_metadata(self):
        # Update metadata this will alter the relevant
        # metadata so that it is specific to the 'apply' analysis being run.
        file = open(self.output_path + '/' + self.experiment_name + '/' + "metadata.pickle", 'rb')
        metadata = pickle.load(file)
        file.close()
        metadata['Export Feature Correlations'] = self.export_feature_correlations
        metadata['Export ROC Plot'] = self.plot_roc
        metadata['Export PRC Plot'] = self.plot_prc
        metadata['Export Metric Boxplots'] = self.plot_metric_boxplots
        metadata['Match Label'] = self.match_label
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
