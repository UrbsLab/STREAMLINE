import glob
import os
import logging
import multiprocessing
from streamline.modeling.utils import SUPPORTED_MODELS
from streamline.modeling.utils import is_supported_model
from streamline.postanalysis.gererate_report import ReportJob
from streamline.utils.runners import runner_fn


class ReportRunner:
    """
    Runner Class for collating dataset compare job
    """

    def __init__(self, output_path=None, experiment_name=None, experiment_path=None, algorithms=None, exclude=None,
                 training=True, rep_data_path=None, dataset_for_rep=None):
        """
        Args:
            output_path: path to output directory
            experiment_name: name of experiment (no spaces)
            algorithms: list of str of ML models to run
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

        if algorithms is None:
            self.algorithms = SUPPORTED_MODELS
            if exclude is not None:
                for algorithm in exclude:
                    try:
                        self.algorithms.remove(algorithm)
                    except Exception:
                        logging.error("Unknown algorithm in exclude: " + str(algorithm))
            self.exclude = None
        else:
            self.algorithms = list()
            for algorithm in algorithms:
                self.algorithms.append(is_supported_model(algorithm))
            self.exclude = exclude

        # Argument checks
        if not os.path.exists(self.output_path):
            raise Exception("Output path must exist (from phase 1) before phase 6 can begin")
        if not os.path.exists(self.output_path + '/' + self.experiment_name):
            raise Exception("Experiment must exist (from phase 1) before phase 6 can begin")

    def run(self, run_parallel=False):
        job_obj = ReportJob(self.output_path, self.experiment_name, None, self.algorithms, None,
                            self.training, self.train_data_path, self.rep_data_path)
        if run_parallel:
            p = multiprocessing.Process(target=runner_fn, args=(job_obj, ))
            p.start()
            p.join()
        else:
            job_obj.run()
