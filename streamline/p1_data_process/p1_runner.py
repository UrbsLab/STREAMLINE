import logging
import os
import pickle
import re
import glob
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

import dask
import logging
from dask.distributed import Client, LocalCluster
logger = logging.getLogger("distributed.worker")
logger.setLevel(logging.WARNING)

from streamline.p1_data_process.data_process import DataProcess
from streamline.utils.runners import parallel_eda_call, num_cores, run_dask_tasks, run_parallel_jobs
from streamline.utils.cluster import get_cluster  # must return a connected Dask Client


class P1Runner:
    """
    Phase 1 runner (dataset-free, flag-driven plotting, Dask-aware).

    Modes (set via run_cluster):
      • "Local"        → local Dask parallelization.
      • "Parallel"     → local joblib parallelization.
      • "BashSLURM"    → submit a bash script (sbatch) that runs p1_jobsubmit.py per dataset.
      • "BashLSF"      → submit a bash script (bsub) that runs p1_jobsubmit.py per dataset.
      • any other str  → modern Dask cluster name; get_cluster(name, ...) returns a connected Client (works in Jupyter).

    If no raw datasets are found and cv_provided=True, it runs in import-only mode by discovering
    <cv_input_root>/<dataset>/CVDatasets folders and seeding schema from the first *_Train.csv.
    """

    def __init__(
        self,
        data_path,
        output_path,
        experiment_name,
        exclude_eda_output=None,
        outcome_label="Class",
        outcome_type=None,
        instance_label=None,
        match_label=None,
        n_splits=10,
        partition_method="Stratified",
        ignore_features=None,
        categorical_features=None,
        quantitative_features=None,
        top_features=20,
        categorical_cutoff=10,
        sig_cutoff=0.05,
        featureeng_missingness=0.5,
        cleaning_missingness=0.5,
        correlation_removal_threshold=1.0,
        random_state=None,
        run_cluster=False,       # False | "Local" | "Parallel" | "BashSLURM" | "BashLSF" | "<dask-cluster-name>"
        queue='defq',
        reserved_memory=4,
        show_plots=False,

        # DataProcess controls
        one_hot_encoding=True,
        cv_provided=False,
        cv_input_root=None,

        # plotting flags (forwarded to DataProcess)
        enable_plots=False,
        plot_missingness=False,
        plot_class_counts=False,
        plot_correlation=False,
        correlation_plot_max_features=200,
        plot_univariate=False,
        univariate_top_k=20,
        plot_anomalies=False,

        # force flag
        force=False
    ):
        self.data_path = data_path
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.outcome_label = outcome_label
        self.outcome_type = DataProcess._normalize_outcome_type(outcome_type) if outcome_type else None
        self.instance_label = instance_label
        self.match_label = match_label
        self.ignore_features = ignore_features
        self.categorical_cutoff = categorical_cutoff
        self.categorical_features = categorical_features
        self.quantitative_features = quantitative_features
        self.featureeng_missingness = featureeng_missingness
        self.cleaning_missingness = cleaning_missingness
        self.correlation_removal_threshold = correlation_removal_threshold
        self.top_features = top_features
        self.exclude_eda_output = exclude_eda_output

        # DataProcess
        self.one_hot_encoding = bool(one_hot_encoding)
        self.cv_provided = bool(cv_provided)
        self.cv_input_root = cv_input_root

        # Analysis excludes only (plots are flag-gated)
        known_exclude_options = ['describe_csv', 'correlation']
        exploration_list = ["Describe", "Univariate Analysis", "Feature Correlation"]
        if exclude_eda_output is not None:
            for x in exclude_eda_output:
                if x not in known_exclude_options:
                    logging.warning("Unknown EDA exclusion option " + str(x))
            if 'describe_csv' in exclude_eda_output and "Describe" in exploration_list:
                exploration_list.remove("Describe")
            if 'correlation' in exclude_eda_output and "Feature Correlation" in exploration_list:
                exploration_list.remove("Feature Correlation")
        self.exploration_list = exploration_list

        self.n_splits = n_splits
        self.partition_method = partition_method
        if self.outcome_type == "Continuous":
            self.partition_method = "Random"
        self.run_cluster = run_cluster  # see modes above
        self.queue = queue
        self.reserved_memory = reserved_memory
        self.show_plots = show_plots
        self.random_state = random_state
        self.sig_cutoff = sig_cutoff

        # Plot flags
        self.enable_plots = bool(enable_plots)
        self.plot_missingness = bool(plot_missingness)
        self.plot_class_counts = bool(plot_class_counts)
        self.plot_correlation = bool(plot_correlation)
        self.correlation_plot_max_features = int(correlation_plot_max_features)
        self.plot_univariate = bool(plot_univariate)
        self.univariate_top_k = int(univariate_top_k)
        self.plot_anomalies = bool(plot_anomalies)

        self.force = bool(force)

        self.make_dir_tree()
        self.save_metadata()

    # ----------------------------
    # Main
    # ----------------------------
    def run(self):
        job_obj_list = []
        unique_datanames = []
        file_count = 0

        discovered_files = []
        if self.data_path and os.path.exists(self.data_path):
            discovered_files = glob.glob(self.data_path.rstrip('/') + '/*')

        # MODE 1: raw datasets
        for dataset_path in discovered_files:
            dataset_path = str(Path(dataset_path).as_posix())
            file_extension = dataset_path.split('/')[-1].split('.')[-1].lower()
            data_name = dataset_path.split('/')[-1].split('.')[0]
            if file_extension not in ('txt', 'csv', 'tsv'):
                continue
            if data_name in unique_datanames:
                continue
            unique_datanames.append(data_name)
            file_count += 1

            ds_out_dir = os.path.join(self.output_path, self.experiment_name, data_name)
            os.makedirs(ds_out_dir, exist_ok=True)

            # Load DataFrame
            if file_extension == 'csv':
                df = pd.read_csv(dataset_path, na_values='NA', sep=',')
            elif file_extension == 'tsv':
                df = pd.read_csv(dataset_path, na_values='NA', sep='\t')
            else:  # txt
                df = pd.read_csv(dataset_path, na_values='NA', delim_whitespace=True)
            df.columns = df.columns.str.strip()

            # Set outcome_type if needed
            if self.outcome_type is None and self.outcome_label in df.columns:
                nunique = df[self.outcome_label].nunique()
                self.outcome_type = "Binary" if nunique == 2 else ("Multiclass" if 2 < nunique <= self.categorical_cutoff else "Continuous")
                if self.outcome_type == "Continuous":
                    self.partition_method = "Random"
                self.save_metadata()

            if self.run_cluster in ("BashSLURM", "BashLSF"):
                self._submit_bash_job(dataset_path)
                continue

            dp = self._build_dataprocess(df, data_name, cv_path=self.cv_input_root)
            job_obj_list.append(dp)

        # MODE 2: import-only (no raw data)
        if self.cv_provided and (file_count == 0):
            if not self.cv_input_root or not os.path.isdir(self.cv_input_root):
                raise Exception("cv_input_root must point to <dataset>/CVDatasets when cv_provided=True and no raw datasets are found")

            candidate_dirs = []
            if os.path.isdir(os.path.join(self.cv_input_root, 'CVDatasets')):
                candidate_dirs = [self.cv_input_root]
            else:
                for name in os.listdir(self.cv_input_root):
                    ds_dir = os.path.join(self.cv_input_root, name)
                    if os.path.isdir(os.path.join(ds_dir, 'CVDatasets')):
                        candidate_dirs.append(ds_dir)
            if not candidate_dirs:
                raise Exception(f"No <dataset>/CVDatasets folders found under cv_input_root: {self.cv_input_root}")

            for ds_dir in candidate_dirs:
                ds_name = os.path.basename(ds_dir.rstrip('/'))
                ds_out_dir = os.path.join(self.output_path, self.experiment_name, ds_name)
                os.makedirs(ds_out_dir, exist_ok=True)

                # seed df from first Train split
                cv_glob = glob.glob(os.path.join(ds_dir, 'CVDatasets', f'{ds_name}_CV_*_Train.csv')) or \
                          glob.glob(os.path.join(ds_dir, 'CVDatasets', '*_Train.csv'))
                if not cv_glob:
                    raise Exception(f"No Train splits found in {os.path.join(ds_dir, 'CVDatasets')}")

                df = pd.read_csv(sorted(cv_glob)[0], na_values='NA')
                df.columns = df.columns.str.strip()

                # Set outcome_type if needed
                if self.outcome_type is None and self.outcome_label in df.columns:
                    nunique = df[self.outcome_label].nunique()
                    self.outcome_type = "Binary" if nunique == 2 else ("Multiclass" if 2 < nunique <= self.categorical_cutoff else "Continuous")
                    if self.outcome_type == "Continuous":
                        self.partition_method = "Random"
                    self.save_metadata()

                if self.run_cluster in ("BashSLURM", "BashLSF"):
                    self._submit_bash_job(
                        dataset_path=None,
                        dataset_name=ds_name,
                        cv_input_root=ds_dir,
                    )
                    continue

                dp = self._build_dataprocess(df, ds_name, cv_path=ds_dir, force_import_only=True)
                job_obj_list.append(dp)

        # error if expected raw data but none
        if not self.cv_provided and file_count == 0:
            raise Exception("There must be at least one .txt, .tsv, or .csv dataset in data_path directory")

        # ---- EXECUTION STRATEGY ----
        run_mode = str(self.run_cluster) if self.run_cluster else "Serial"
        if run_mode == "Local":
            # Local Dask parallelization
            n_workers = num_cores
            with LocalCluster(processes=True, n_workers=n_workers, threads_per_worker=1) as cluster:
                with Client(cluster) as client:
                    tasks = [dask.delayed(parallel_eda_call)(job_obj, {'top_features': self.top_features}) for job_obj in job_obj_list]
                    run_dask_tasks(tasks, client, label="Phase 1 Dask jobs")
        elif run_mode == "Parallel":
            run_parallel_jobs(
                parallel_eda_call,
                [(job_obj, {'top_features': self.top_features}) for job_obj in job_obj_list],
                label="Phase 1 Parallel jobs",
            )
        elif self.run_cluster and self.run_cluster != "Serial" and self.run_cluster not in ("BashSLURM", "BashLSF"):
            # Modern Dask cluster (works in Jupyter)
            client: Client = get_cluster(self.run_cluster,
                                         os.path.join(self.output_path, self.experiment_name),
                                         self.queue, self.reserved_memory)
            tasks = [dask.delayed(parallel_eda_call)(job_obj, {'top_features': self.top_features}) for job_obj in job_obj_list]
            run_dask_tasks(tasks, client, label="Phase 1 Dask jobs")
        else:
            # Serial
            for job_obj in job_obj_list:
                job_obj.run(self.top_features)

        self.save_run_params()

    # ----------------------------
    # Helpers
    # ----------------------------
    def _build_dataprocess(self, df: pd.DataFrame, dataset_name: str, cv_path: "str | None", force_import_only: bool = False):
        return DataProcess(
            data=df,
            experiment_path=os.path.join(self.output_path, self.experiment_name),
            outcome_label=self.outcome_label,
            outcome_type=self.outcome_type,
            match_label=self.match_label if (self.match_label in df.columns) else None,
            instance_label=self.instance_label if (self.instance_label in df.columns) else None,
            ignore_features=self.ignore_features,
            categorical_features=self.categorical_features,
            quantitative_features=self.quantitative_features,
            exclude_eda_output=self.exclude_eda_output,
            categorical_cutoff=self.categorical_cutoff,
            sig_cutoff=self.sig_cutoff,
            featureeng_missingness=self.featureeng_missingness,
            cleaning_missingness=self.cleaning_missingness,
            correlation_removal_threshold=self.correlation_removal_threshold,
            partition_method=self.partition_method,
            n_splits=self.n_splits,
            one_hot_encoding=self.one_hot_encoding,
            random_state=self.random_state,
            show_plots=self.show_plots,
            cv_provided=(self.cv_provided or force_import_only),
            cv_input_path=cv_path,
            dataset_name=dataset_name,
            # plot flags
            enable_plots=self.enable_plots,
            plot_missingness=self.plot_missingness,
            plot_class_counts=self.plot_class_counts,
            plot_correlation=self.plot_correlation,
            correlation_plot_max_features=self.correlation_plot_max_features,
            plot_univariate=self.plot_univariate,
            univariate_top_k=self.univariate_top_k,
            plot_anomalies=self.plot_anomalies,
        )

    def make_dir_tree(self):
        """
        Validates and creates experiment structure.
        data_path can be missing when cv_provided=True (import-only).
        """
        if not self.cv_provided:
            if not self.data_path or not os.path.exists(self.data_path):
                raise Exception("Provided data_path does not exist")

        exp_dir = os.path.join(self.output_path, self.experiment_name)
        if os.path.exists(exp_dir):
            if not self.force:
                raise Exception(
                    f"Error: Experiment folder already exists: {exp_dir} (use force=True to overwrite)."
                )
            else:
                import shutil
                logging.warning(f"Force flag set: removing existing experiment folder {exp_dir}")
                shutil.rmtree(exp_dir)

        if not re.match(r'^[A-Za-z0-9_]+$', self.experiment_name):
            raise Exception('Experiment Name must be alphanumeric')

        os.makedirs(self.output_path, exist_ok=True)
        os.mkdir(exp_dir)
        os.mkdir(exp_dir + '/jobsCompleted')
        os.mkdir(exp_dir + '/jobs')
        os.mkdir(exp_dir + '/logs')

    def save_metadata(self):
        metadata = dict()
        metadata['Data Path'] = self.data_path
        metadata['Output Path'] = self.output_path
        metadata['Experiment Name'] = self.experiment_name
        metadata['Outcome Label'] = self.outcome_label
        metadata['Outcome Type'] = self.outcome_type
        metadata['Instance Label'] = self.instance_label
        metadata['Match Label'] = self.match_label
        metadata['Ignored Features'] = self.ignore_features
        metadata['Specified Categorical Features'] = self.categorical_features
        metadata['Specified Quantitative Features'] = self.quantitative_features
        metadata['CV Partitions'] = self.n_splits
        metadata['Partition Method'] = self.partition_method
        metadata['Categorical Cutoff'] = self.categorical_cutoff
        metadata['Statistical Significance Cutoff'] = self.sig_cutoff
        metadata['Engineering Missingness Cutoff'] = self.featureeng_missingness
        metadata['Cleaning Missingness Cutoff'] = self.cleaning_missingness
        metadata['Correlation Removal Threshold'] = self.correlation_removal_threshold
        metadata['List of Exploratory Analysis Ran'] = self.exploration_list
        metadata['Random Seed'] = self.random_state
        metadata['Run From Notebook'] = self.show_plots
        with open(self.output_path + '/' + self.experiment_name + '/' + "metadata.pickle", 'wb') as f:
            pickle.dump(metadata, f)

    def save_run_params(self, run_parallel=False):
        """Save or update run parameters in a single pickle file (dict keyed by timestamp)."""
        run_params = {
            "run_parallel": run_parallel,
            "data_path": self.data_path,
            "output_path": self.output_path,
            "experiment_name": self.experiment_name,
            "outcome_label": self.outcome_label,
            "outcome_type": self.outcome_type,
            "instance_label": self.instance_label,
            "match_label": self.match_label,
            "n_splits": self.n_splits,
            "partition_method": self.partition_method,
            "ignore_features": self.ignore_features,
            "categorical_features": self.categorical_features,
            "quantitative_features": self.quantitative_features,
            "categorical_cutoff": self.categorical_cutoff,
            "sig_cutoff": self.sig_cutoff,
            "featureeng_missingness": self.featureeng_missingness,
            "cleaning_missingness": self.cleaning_missingness,
            "correlation_removal_threshold": self.correlation_removal_threshold,
            "random_state": self.random_state,
            "run_cluster": self.run_cluster,
            "queue": self.queue,
            "reserved_memory": self.reserved_memory,
            "show_plots": self.show_plots,
            "one_hot_encoding": self.one_hot_encoding,
            "cv_provided": self.cv_provided,
            "cv_input_root": self.cv_input_root,
            "exclude_eda_output": self.exclude_eda_output,
        }

        exp_root = os.path.join(self.output_path, self.experiment_name)
        os.makedirs(exp_root, exist_ok=True)
        params_file = os.path.join(exp_root, "run_params.pickle")

        # Load existing dictionary if file exists, else start new
        if os.path.exists(params_file):
            with open(params_file, "rb") as f:
                all_params = pickle.load(f)
        else:
            all_params = {}

        ts = datetime.now().isoformat()
        all_params[ts] = run_params

        with open(params_file, "wb") as f:
            pickle.dump(all_params, f)

        logging.info(f"Updated run parameters in {params_file}")

    # ----------------------------
    # Bash submission (uses p1_jobsubmit.py)
    # ----------------------------
    def _submit_bash_job(self, dataset_path, dataset_name=None, cv_input_root=None):
        job_ref = str(time.time())
        run_dir = self.output_path + '/' + self.experiment_name
        os.makedirs(run_dir + '/jobs', exist_ok=True)
        os.makedirs(run_dir + '/logs', exist_ok=True)
        job_name = run_dir + f'/jobs/P1_{job_ref}_run.sh'

        
        if self.run_cluster == "BashSLURM":
            launcher = 'sbatch' 
        elif self.run_cluster == "BashLSF":
            launcher = 'bsub <'
        else:
            raise Exception("Bash submission of HPC type unsupported")
        
        with open(job_name, 'w') as sh:
            if self.run_cluster == "BashSLURM":
                sh.write('#!/bin/bash\n')
                sh.write('#SBATCH -p ' + self.queue + '\n')
                sh.write('#SBATCH --job-name=' + job_ref + '\n')
                sh.write('#SBATCH --mem=' + str(self.reserved_memory) + 'G' + '\n')
                sh.write('#SBATCH -o ' + run_dir + f'/logs/P1_{job_ref}.o\n')
                sh.write('#SBATCH -e ' + run_dir + f'/logs/P1_{job_ref}.e\n')
                cmd = self._bash_submit_command(dataset_path, dataset_name=dataset_name, cv_input_root=cv_input_root)
                sh.write('srun ' + cmd + '\n')
            else:
                sh.write('#!/bin/bash\n')
                sh.write('#BSUB -q ' + self.queue + '\n')
                sh.write('#BSUB -J ' + job_ref + '\n')
                sh.write('#BSUB -R "rusage[mem=' + str(self.reserved_memory) + 'G]"' + '\n')
                sh.write('#BSUB -M ' + str(self.reserved_memory) + 'GB' + '\n')
                sh.write('#BSUB -o ' + run_dir + f'/logs/P1_{job_ref}.o\n')
                sh.write('#BSUB -e ' + run_dir + f'/logs/P1_{job_ref}.e\n')
                cmd = self._bash_submit_command(dataset_path, dataset_name=dataset_name, cv_input_root=cv_input_root)
                sh.write(cmd + '\n')

        os.system(f'{launcher} {job_name}')

    def _bash_submit_command(self, dataset_path, dataset_name=None, cv_input_root=None):
        """
        Build command to run a single-dataset job via p1_jobsubmit.py (bash path).
        p1_jobsubmit.py must parse args and run DataProcess once.
        """
        script_path = str(Path(__file__).parent / "p1_jobsubmit.py")
        args = [
            'python', script_path,
            '--dataset_path', dataset_path or '',
            '--dataset_name', dataset_name or '',
            '--output_path', self.output_path,
            '--experiment_name', self.experiment_name,
            '--exclude', ','.join(self.exclude_eda_output) if self.exclude_eda_output else '',
            '--outcome_label', self.outcome_label or '',
            '--outcome_type', self.outcome_type or '',
            '--instance_label', self.instance_label or '',
            '--match_label', self.match_label or '',
            '--n_splits', str(self.n_splits),
            '--partition_method', self.partition_method,
            '--ignore_features', ','.join(self.ignore_features) if isinstance(self.ignore_features, list) else (self.ignore_features or ''),
            '--categorical_features', ','.join(self.categorical_features) if isinstance(self.categorical_features, list) else (self.categorical_features or ''),
            '--quantitative_features', ','.join(self.quantitative_features) if isinstance(self.quantitative_features, list) else (self.quantitative_features or ''),
            '--top_features', str(self.top_features),
            '--categorical_cutoff', str(self.categorical_cutoff),
            '--sig_cutoff', str(self.sig_cutoff),
            '--featureeng_missingness', str(self.featureeng_missingness),
            '--cleaning_missingness', str(self.cleaning_missingness),
            '--correlation_removal_threshold', str(self.correlation_removal_threshold),
            '--random_state', str(self.random_state) if self.random_state is not None else '',
            '--one_hot_encoding', str(int(self.one_hot_encoding)),
            '--cv_provided', str(int(self.cv_provided)),
            '--cv_input_root', cv_input_root or self.cv_input_root or '',
            # plotting flags
            '--enable_plots', str(int(self.enable_plots)),
            '--plot_missingness', str(int(self.plot_missingness)),
            '--plot_class_counts', str(int(self.plot_class_counts)),
            '--plot_correlation', str(int(self.plot_correlation)),
            '--correlation_plot_max_features', str(int(self.correlation_plot_max_features)),
            '--plot_univariate', str(int(self.plot_univariate)),
            '--univariate_top_k', str(int(self.univariate_top_k)),
            '--plot_anomalies', str(int(self.plot_anomalies)),
        ]
        return ' '.join(args)
