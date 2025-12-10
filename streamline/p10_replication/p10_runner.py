# streamline/p10_replication/p10_runner.py

from __future__ import annotations
import glob
import logging
import os
import pickle
import time
from pathlib import Path
from typing import List, Optional

import dask
from dask.distributed import Client, LocalCluster
from joblib import Parallel, delayed

from streamline.utils.cluster import get_cluster
from streamline.utils.runners import num_cores, runner_fn
from streamline.p10_replication.replication import ReplicationJob


class P10Runner:
    """
    Phase 10 – Replication / External Validation

    Applies trained models from one training dataset to one or more replication datasets.
    """

    def __init__(
        self,
        rep_data_path: str,
        dataset_for_rep: str,
        output_path: str,
        experiment_name: str,
        outcome_label: Optional[str] = None,
        instance_label: Optional[str] = None,
        match_label: Optional[str] = None,
        exclude_plots: Optional[List[str]] = None,
        run_cluster: str = "Serial",   # Serial | Local | BashSLURM | BashLSF | <dask-cluster-name>
        queue: str = "defq",
        reserved_memory: int = 4,
        show_plots: bool = False,
    ):
        self.rep_data_path = rep_data_path
        self.dataset_for_rep = dataset_for_rep
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.run_cluster = run_cluster or "Serial"
        self.queue = queue
        self.reserved_memory = int(reserved_memory)
        self.show_plots = show_plots
        self.match_label = match_label

        known_exclude_options = [
            "plot_ROC",
            "plot_PRC",
            "plot_metric_boxplots",
            "feature_correlations",
        ]
        if exclude_plots is not None:
            for x in exclude_plots:
                if x not in known_exclude_options:
                    logging.warning("Unknown exclusion option %s", x)
        else:
            exclude_plots = []

        self.exclude_plots = exclude_plots
        self.plot_roc = "plot_ROC" not in exclude_plots
        self.plot_prc = "plot_PRC" not in exclude_plots
        self.plot_metric_boxplots = "plot_metric_boxplots" not in exclude_plots
        self.plot_fi_box = "plot_FI_box" not in exclude_plots
        self.export_feature_correlations = "feature_correlations" not in exclude_plots

        self.exp_root = Path(self.output_path) / self.experiment_name
        if not self.exp_root.is_dir():
            raise Exception("Experiment must exist (from phases 1–8) before replication can begin")

        # load metadata
        with open(self.exp_root / "metadata.pickle", "rb") as f:
            metadata = pickle.load(f)

        self.outcome_type = metadata["Outcome Type"]

        # respect explicit CLI overrides if provided
        self.outcome_label = outcome_label or metadata["Outcome Label"]
        self.instance_label = instance_label or metadata["Instance Label"]

        self.ignore_features = metadata.get("Ignored Features", [])
        self.categorical_cutoff = metadata["Categorical Cutoff"]
        self.sig_cutoff = metadata["Statistical Significance Cutoff"]
        self.featureeng_missingness = metadata["Engineering Missingness Cutoff"]
        self.cleaning_missingness = metadata["Cleaning Missingness Cutoff"]
        self.cv_partitions = metadata["CV Partitions"]
        self.scale_data = metadata["Use Data Scaling"]
        self.impute_data = metadata["Use Data Imputation"]
        self.multi_impute = metadata["Use Multivariate Imputation"]
        self.scoring_metric = metadata["Primary Metric"]
        self.random_state = metadata["Random Seed"]

        self._update_metadata_flags()

        # location of folder containing models for this training dataset
        self.data_name = Path(self.dataset_for_rep).stem
        self.full_path = str(self.exp_root / self.data_name)

        os.makedirs(self.full_path + "/replication", exist_ok=True)

        if not self.show_plots:
            os.makedirs(self.exp_root / "jobs", exist_ok=True)
            os.makedirs(self.exp_root / "logs", exist_ok=True)

        self._get_algorithms()

    def _update_metadata_flags(self):
        # Update metadata to reflect replication-specific export choices
        meta_path = self.exp_root / "metadata.pickle"
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        metadata["Export Feature Correlations"] = self.export_feature_correlations
        metadata["Export ROC Plot"] = self.plot_roc
        metadata["Export PRC Plot"] = self.plot_prc
        metadata["Export Metric Boxplots"] = self.plot_metric_boxplots
        metadata["Match Label"] = self.match_label

        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)

    def _get_algorithms(self):
        with open(self.exp_root / "algInfo.pickle", "rb") as f:
            alg_info = pickle.load(f)
        algorithms = [alg for alg, (flag, *_rest) in alg_info.items() if flag]
        self.algorithms = algorithms

    # ------------------------------------------------------------------ #
    # Main run
    # ------------------------------------------------------------------ #

    def run(self):
        # Discover replication files once
        files = sorted(glob.glob(os.path.join(self.rep_data_path, "*")))
        unique_datanames = set()
        jobs = []

        for dataset_filename in files:
            dataset_filename = str(Path(dataset_filename).as_posix())
            ext = Path(dataset_filename).suffix.lower()
            apply_name = Path(dataset_filename).stem

            if ext not in {".txt", ".csv", ".tsv"}:
                continue

            if apply_name in unique_datanames:
                continue
            unique_datanames.add(apply_name)

            rep_dir = Path(self.full_path) / "replication" / apply_name
            rep_dir.mkdir(parents=True, exist_ok=True)

            if self.run_cluster in {"BashSLURM", "BashLSF"}:
                self._submit_bash(dataset_filename)
                continue

            job_obj = ReplicationJob(
                dataset_filename=dataset_filename,
                dataset_for_rep=self.dataset_for_rep,
                full_path=self.full_path,
                outcome_label=self.outcome_label,
                outcome_type=self.outcome_type,
                instance_label=self.instance_label,
                match_label=self.match_label,
                ignore_features=self.ignore_features,
                cv_partitions=self.cv_partitions,
                exclude_plots=self.exclude_plots,
                categorical_cutoff=self.categorical_cutoff,
                sig_cutoff=self.sig_cutoff,
                scale_data=self.scale_data,
                impute_data=self.impute_data,
                multi_impute=self.multi_impute,
                show_plots=self.show_plots,
                scoring_metric=self.scoring_metric,
                random_state=self.random_state,
            )
            jobs.append(job_obj)

        if not jobs:
            raise Exception(
                "There must be at least one .txt, .csv, or .tsv dataset in rep_data_path directory"
            )

        # Dispatch according to run_cluster
        if self.run_cluster == "Serial":
            for job_obj in jobs:
                job_obj.run()
        elif self.run_cluster == "Local":
            # local multiprocessing via joblib + runner_fn
            Parallel(n_jobs=num_cores)(
                delayed(runner_fn)(job_obj) for job_obj in jobs
            )
        else:
            # Dask cluster
            if self.run_cluster in {"BashSLURM", "BashLSF"}:
                # already submitted individually
                return

            client: Client = get_cluster(
                self.run_cluster, str(self.exp_root), self.queue, self.reserved_memory
            )
            dask.compute(
                [dask.delayed(runner_fn)(job_obj) for job_obj in jobs],
                scheduler=client,
            )

    # ------------------------------------------------------------------ #
    # Bash submission helpers
    # ------------------------------------------------------------------ #

    def _submit_bash(self, dataset_filename: str):
        job_ref = str(time.time())
        jobs_dir = self.exp_root / "jobs"
        logs_dir = self.exp_root / "logs"
        os.makedirs(jobs_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        sh = jobs_dir / f"P10_{job_ref}_run.sh"
        script = Path(__file__).with_name("p10_jobsubmit.py")

        args = " ".join(
            [
                "python",
                str(script),
                "--dataset_filename",
                dataset_filename,
                "--dataset_for_rep",
                self.dataset_for_rep,
                "--output_path",
                str(self.output_path),
                "--experiment_name",
                self.experiment_name,
                "--outcome_label",
                self.outcome_label,
                "--instance_label",
                self.instance_label or "",
                "--match_label",
                self.match_label or "",
                "--exclude_plots",
                ",".join(self.exclude_plots) if self.exclude_plots else "",
                "--show_plots",
                str(int(self.show_plots)),
            ]
        )

        with open(sh, "w") as f:
            f.write("#!/bin/bash\n")
            if self.run_cluster == "BashSLURM":
                f.write(f"#SBATCH -p {self.queue}\n")
                f.write(f"#SBATCH --job-name={job_ref}\n")
                f.write(f"#SBATCH --mem={self.reserved_memory}G\n")
                f.write(f"#SBATCH -o {logs_dir}/P10_{job_ref}.o\n")
                f.write(f"#SBATCH -e {logs_dir}/P10_{job_ref}.e\n")
                f.write(f"srun {args}\n")
                launcher = "sbatch"
                cmd = f"{launcher} {sh}"
            else:
                f.write(f"#BSUB -q {self.queue}\n")
                f.write(f"#BSUB -J {job_ref}\n")
                f.write(f"#BSUB -R \"rusage[mem={self.reserved_memory}G]\"\n")
                f.write(f"#BSUB -M {self.reserved_memory}GB\n")
                f.write(f"#BSUB -o {logs_dir}/P10_{job_ref}.o\n")
                f.write(f"#BSUB -e {logs_dir}/P10_{job_ref}.e\n")
                f.write(f"{args}\n")
                launcher = "bsub <"
                cmd = f"{launcher} {sh}"

        os.system(cmd)
