import os
import dask
import multiprocessing
from joblib import Parallel, delayed
from streamline.modeling.utils import SUPPORTED_MODELS
from streamline.modeling.utils import is_supported_model
from streamline.postanalysis.dataset_compare import CompareJob
from streamline.utils.runners import runner_fn
from streamline.utils.cluster import get_cluster


class CompareRunner:
    """
    Runner Class for collating dataset compare job
    """

    def __init__(self, output_path, experiment_name, experiment_path=None, algorithms=None, exclude=("XCS", "eLCS"),
                 class_label="Class", instance_label=None, sig_cutoff=0.05, show_plots=False):
        """
        Args:
            output_path: path to output directory
            experiment_name: name of experiment (no spaces)
            algorithms: list of str of ML models to run
            sig_cutoff: significance cutoff, default=0.05
            show_plots: flag to show plots
        """
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.class_label = class_label
        self.instance_label = instance_label
        self.experiment_path = experiment_path

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

        self.sig_cutoff = sig_cutoff
        self.show_plots = show_plots

        # Argument checks
        if not os.path.exists(self.output_path):
            raise Exception("Output path must exist (from phase 1) before phase 6 can begin")
        if not os.path.exists(self.output_path + '/' + self.experiment_name):
            raise Exception("Experiment must exist (from phase 1) before phase 6 can begin")

    def run(self, run_parallel=False):
        job_obj = CompareJob(self.output_path, self.experiment_name, None, self.algorithms, None,
                             self.class_label, self.instance_label, self.sig_cutoff, self.show_plots)
        if run_parallel in ["multiprocessing", "True"]:
            # p = multiprocessing.Process(target=runner_fn, args=(job_obj, ))
            # p.start()
            # p.join()
            Parallel()(delayed(runner_fn)(job_obj) for job_obj in [job_obj, ])
        elif run_parallel and (run_parallel in ["multiprocessing", "True"]):
            get_cluster(run_parallel) 
            dask.compute([dask.delayed(runner_fn)(job_obj) for job_obj in [job_obj, ]])
        else:
            job_obj.run()
