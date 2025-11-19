from __future__ import annotations
import os
import time
import logging
from pathlib import Path
from typing import Optional, Sequence

import dask
from dask.distributed import Client, LocalCluster

from streamline.p8_post_analysis import StatisticsPhaseJob
from streamline.utils.runners import num_cores
from streamline.utils.cluster import get_cluster  # returns a connected Dask Client

logger = logging.getLogger(__name__)


class P8Runner:
    """
    Phase 8 runner (Statistics / post-analysis).
    """

    def __init__(
        self,
        output_path: str,
        experiment_name: str,
        *,
        outcome_label: str = "Class",
        outcome_type: Optional[str] = None,   # if None, inferred from metadata.pickle
        instance_label: Optional[str] = None,
        n_splits: int = 10,                   # passed for ROC/PRC per-CV logic
        scoring_metric: str = "balanced_accuracy",
        top_features: int = 40,
        sig_cutoff: float = 0.05,
        metric_weight: str = "balanced_accuracy",
        scale_data: bool = True,
        exclude_plots: Optional[Sequence[str]] = None,
        show_plots: bool = False,
        # execution
        run_cluster: str = "Serial",          # "Serial" | "Local" | "BashSLURM" | "BashLSF" | "<cluster>"
        queue: str = "defq",
        reserved_memory: int = 4,
    ):
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.outcome_label = outcome_label
        self.outcome_type = outcome_type
        self.instance_label = instance_label
        self.n_splits = int(n_splits)

        self.scoring_metric = scoring_metric
        self.top_features = int(top_features)
        self.sig_cutoff = float(sig_cutoff)
        self.metric_weight = metric_weight
        self.scale_data = bool(scale_data)
        self.exclude_plots = list(exclude_plots) if exclude_plots is not None else None
        self.show_plots = bool(show_plots)

        self.run_cluster = run_cluster or "Serial"
        self.queue = queue
        self.reserved_memory = int(reserved_memory)

        self.exp_root = os.path.join(self.output_path, self.experiment_name)
        if not os.path.isdir(self.exp_root):
            raise Exception("Experiment must exist before statistics phase can begin")

        if self.outcome_type is None:
            meta_path = os.path.join(self.exp_root, "metadata.pickle")
            if not os.path.isfile(meta_path):
                raise Exception("metadata.pickle not found; cannot infer outcome_type")
            import pickle
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
            self.outcome_type = metadata.get("Outcome Type", "Binary")

        if self.run_cluster in ("BashSLURM", "BashLSF"):
            os.makedirs(os.path.join(self.exp_root, "jobs"), exist_ok=True)
            os.makedirs(os.path.join(self.exp_root, "logs"), exist_ok=True)

    # --------------------------------------------------------------------- run
    def run(self):
        datasets = [
            os.path.join(self.exp_root, name)
            for name in sorted(os.listdir(self.exp_root))
            if os.path.isdir(os.path.join(self.exp_root, name))
            and name not in {"jobsCompleted", "jobs", "logs", "dask_logs", "DatasetComparisons"}
            and os.path.isdir(os.path.join(self.exp_root, name, "CVDatasets"))
        ]
        if not datasets:
            logging.warning("No datasets found for statistics under %s", self.exp_root)
            return

        mode = self.run_cluster
        if mode == "Serial":
            for ds in datasets:
                self._run_one(ds)
        elif mode == "Local":
            with LocalCluster(processes=True, n_workers=num_cores, threads_per_worker=1) as cluster:
                with Client(cluster) as client:
                    tasks = [dask.delayed(self._run_one)(ds) for ds in datasets]
                    dask.compute(tasks, scheduler=client)
        elif mode in ("BashSLURM", "BashLSF"):
            for ds in datasets:
                self._submit_bash(ds, mode)
        else:
            client: Client = get_cluster(mode, self.exp_root, self.queue, self.reserved_memory)
            tasks = [dask.delayed(self._run_one)(ds) for ds in datasets]
            dask.compute(tasks, scheduler=client)

    def _run_one(self, dataset_dir: str):
        job = StatisticsPhaseJob(
            dataset_dir=dataset_dir,
            outcome_label=self.outcome_label,
            outcome_type=self.outcome_type,
            instance_label=self.instance_label,
            scoring_metric=self.scoring_metric,
            cv_partitions=self.n_splits,
            top_features=self.top_features,
            sig_cutoff=self.sig_cutoff,
            metric_weight=self.metric_weight,
            scale_data=self.scale_data,
            exclude_plots=self.exclude_plots,
            show_plots=self.show_plots,
        )
        job.run()

    def _submit_bash(self, dataset_dir: str, mode: str):
        job_ref = str(time.time())
        jobs = os.path.join(self.exp_root, "jobs")
        logs = os.path.join(self.exp_root, "logs")
        os.makedirs(jobs, exist_ok=True)
        os.makedirs(logs, exist_ok=True)

        sh_path = os.path.join(jobs, f"P7_{job_ref}_run.sh")
        launcher = "sbatch" if mode == "BashSLURM" else "bsub <"

        script = str(Path(__file__).parent / "p7_jobsubmit.py")
        exclude_arg = ",".join(self.exclude_plots) if self.exclude_plots else ""

        args = [
            "python", script,
            "--dataset_dir", dataset_dir,
            "--outcome_label", self.outcome_label,
            "--outcome_type", self.outcome_type,
            "--instance_label", self.instance_label or "",
            "--n_splits", str(self.n_splits),
            "--scoring_metric", self.scoring_metric,
            "--top_features", str(self.top_features),
            "--sig_cutoff", str(self.sig_cutoff),
            "--metric_weight", self.metric_weight,
            "--scale_data", "1" if self.scale_data else "0",
            "--exclude_plots", exclude_arg,
            "--show_plots", "1" if self.show_plots else "0",
        ]
        cmd = " ".join(args)

        with open(sh_path, "w") as sh:
            sh.write("#!/bin/bash\n")
            if mode == "BashSLURM":
                sh.write(f"#SBATCH -p {self.queue}\n")
                sh.write(f"#SBATCH --job-name={job_ref}\n")
                sh.write(f"#SBATCH --mem={self.reserved_memory}G\n")
                sh.write(f"#SBATCH -o {logs}/P7_{job_ref}.o\n")
                sh.write(f"#SBATCH -e {logs}/P7_{job_ref}.e\n")
                sh.write("srun " + cmd + "\n")
            else:
                sh.write(f"#BSUB -q {self.queue}\n")
                sh.write(f"#BSUB -J {job_ref}\n")
                sh.write(f"#BSUB -R \"rusage[mem={self.reserved_memory}G]\"\n")
                sh.write(f"#BSUB -M {self.reserved_memory}GB\n")
                sh.write(f"#BSUB -o {logs}/P7_{job_ref}.o\n")
                sh.write(f"#BSUB -e {logs}/P7_{job_ref}.e\n")
                sh.write(cmd + "\n")

        os.system(f"{launcher} {sh_path}")
