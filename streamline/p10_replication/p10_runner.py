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

from streamline.p10_replication.replication import ReplicationJob
from streamline.utils.cluster import get_cluster
from streamline.utils.runners import num_cores, quote_command_parts, run_dask_tasks, run_parallel_items, runner_fn


class P10Runner:
    """
    Phase 10 runner (Replication / External Validation).

    Applies a trained dataset pipeline to one or more external replication datasets.
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
        run_cluster: str = "Serial",   # Serial | Local | Parallel | BashSLURM | BashLSF | <dask-cluster-name>
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
        self.show_plots = bool(show_plots)
        self.match_label = match_label

        known_exclude_options = [
            "plot_ROC",
            "plot_PRC",
            "plot_metric_boxplots",
            "plot_FI_box",
            "feature_correlations",
        ]
        if exclude_plots is None:
            exclude_plots = []
        for entry in exclude_plots:
            if entry not in known_exclude_options:
                logging.warning("Unknown plot exclusion option: %s", entry)
        self.exclude_plots = exclude_plots

        self.exp_root = Path(self.output_path) / self.experiment_name
        if not self.exp_root.is_dir():
            raise Exception("Experiment must exist (from phases 1-9) before replication can begin")

        with (self.exp_root / "metadata.pickle").open("rb") as f:
            metadata = pickle.load(f)

        self.outcome_type = metadata["Outcome Type"]
        self.outcome_label = outcome_label or metadata["Outcome Label"]
        self.instance_label = instance_label or metadata.get("Instance Label")
        self.ignore_features = metadata.get("Ignored Features", [])
        self.categorical_cutoff = metadata.get("Categorical Cutoff", 10)
        self.sig_cutoff = metadata.get("Statistical Significance Cutoff", 0.05)
        self.featureeng_missingness = metadata.get("Engineering Missingness Cutoff", 0.5)
        self.cleaning_missingness = metadata.get("Cleaning Missingness Cutoff", 0.5)
        self.cv_partitions = metadata.get("CV Partitions", 5)
        self.scale_data = metadata.get("Use Data Scaling", True)
        self.impute_data = metadata.get("Use Data Imputation", True)
        self.multi_impute = metadata.get("Use Multivariate Imputation", False)
        self.scoring_metric = metadata.get("Primary Metric", "balanced_accuracy")
        self.random_state = metadata.get("Random Seed")

        self.train_data_name = Path(self.dataset_for_rep).stem
        self.train_dataset_root = self.exp_root / self.train_data_name
        if not self.train_dataset_root.is_dir():
            raise Exception(
                f"Training dataset directory does not exist in experiment output: {self.train_dataset_root}"
            )

        (self.train_dataset_root / "replication").mkdir(parents=True, exist_ok=True)

        if self.run_cluster in {"BashSLURM", "BashLSF"}:
            (self.exp_root / "jobs").mkdir(parents=True, exist_ok=True)
            (self.exp_root / "logs").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> None:
        files = sorted(glob.glob(os.path.join(self.rep_data_path, "*")))
        jobs: List[ReplicationJob] = []
        seen_names = set()

        for dataset_filename in files:
            ext = Path(dataset_filename).suffix.lower()
            if ext not in {".csv", ".tsv", ".txt"}:
                continue

            apply_name = Path(dataset_filename).stem
            if apply_name in seen_names:
                continue
            seen_names.add(apply_name)

            if self.run_cluster in {"BashSLURM", "BashLSF"}:
                self._submit_bash(dataset_filename)
                continue

            jobs.append(self._build_job(dataset_filename))

        if not jobs and self.run_cluster not in {"BashSLURM", "BashLSF"}:
            raise Exception(
                "There must be at least one .txt, .csv, or .tsv dataset in rep_data_path"
            )

        if self.run_cluster == "Serial":
            for job in jobs:
                job.run()
        elif self.run_cluster == "Local":
            with LocalCluster(processes=True, n_workers=num_cores, threads_per_worker=1) as cluster:
                with Client(cluster) as client:
                    run_dask_tasks(
                        [dask.delayed(runner_fn)(job) for job in jobs],
                        client,
                        label="Phase 10 Dask jobs",
                    )
        elif self.run_cluster == "Parallel":
            run_parallel_items(runner_fn, jobs, label="Phase 10 Parallel jobs")
        elif self.run_cluster in {"BashSLURM", "BashLSF"}:
            return
        else:
            client: Client = get_cluster(
                self.run_cluster,
                str(self.exp_root),
                self.queue,
                self.reserved_memory,
            )
            run_dask_tasks(
                [dask.delayed(runner_fn)(job) for job in jobs],
                client,
                label="Phase 10 Dask jobs",
            )

    def _build_job(self, dataset_filename: str) -> ReplicationJob:
        return ReplicationJob(
            dataset_filename=dataset_filename,
            dataset_for_rep=self.dataset_for_rep,
            full_path=str(self.train_dataset_root),
            outcome_label=self.outcome_label,
            outcome_type=self.outcome_type,
            instance_label=self.instance_label,
            match_label=self.match_label,
            ignore_features=self.ignore_features,
            cv_partitions=self.cv_partitions,
            exclude_plots=self.exclude_plots,
            categorical_cutoff=self.categorical_cutoff,
            sig_cutoff=self.sig_cutoff,
            featureeng_missingness=self.featureeng_missingness,
            cleaning_missingness=self.cleaning_missingness,
            scale_data=self.scale_data,
            impute_data=self.impute_data,
            multi_impute=self.multi_impute,
            show_plots=self.show_plots,
            scoring_metric=self.scoring_metric,
            random_state=self.random_state,
        )

    # ------------------------------------------------------------------
    # Bash submission
    # ------------------------------------------------------------------

    def _submit_bash(self, dataset_filename: str) -> None:
        job_ref = str(time.time())
        jobs_dir = self.exp_root / "jobs"
        logs_dir = self.exp_root / "logs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        sh_path = jobs_dir / f"P10_{job_ref}_run.sh"
        submit_script = Path(__file__).with_name("p10_jobsubmit.py")

        args = quote_command_parts(
            [
                "python",
                str(submit_script),
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

        with sh_path.open("w") as f:
            f.write("#!/bin/bash\n")
            if self.run_cluster == "BashSLURM":
                f.write(f"#SBATCH -p {self.queue}\n")
                f.write(f"#SBATCH --job-name={job_ref}\n")
                f.write(f"#SBATCH --mem={self.reserved_memory}G\n")
                f.write(f"#SBATCH -o {logs_dir}/P10_{job_ref}.o\n")
                f.write(f"#SBATCH -e {logs_dir}/P10_{job_ref}.e\n")
                f.write(f"srun {args}\n")
                launch_cmd = f"sbatch {quote_command_parts([sh_path])}"
            else:
                f.write(f"#BSUB -q {self.queue}\n")
                f.write(f"#BSUB -J {job_ref}\n")
                f.write(f"#BSUB -R \"rusage[mem={self.reserved_memory}G]\"\n")
                f.write(f"#BSUB -M {self.reserved_memory}GB\n")
                f.write(f"#BSUB -o {logs_dir}/P10_{job_ref}.o\n")
                f.write(f"#BSUB -e {logs_dir}/P10_{job_ref}.e\n")
                f.write(f"{args}\n")
                launch_cmd = f"bsub < {quote_command_parts([sh_path])}"

        os.system(launch_cmd)
