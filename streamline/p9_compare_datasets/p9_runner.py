# streamline/p9_compare_datasets/p9_runner.py
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import dask
from dask.distributed import Client, LocalCluster

from streamline.utils.cluster import get_cluster
from streamline.p9_compare_datasets.compare_datasets import DatasetCompareJob
from streamline.utils.runners import num_cores


class P9Runner:
    """
    Phase 9 Runner: dataset-level comparison across all datasets in an experiment.
    """

    def __init__(
        self,
        output_path: str,
        experiment_name: str,
        outcome_label: str = "Class",
        outcome_type: str = "Binary",
        instance_label: Optional[str] = None,
        sig_cutoff: float = 0.05,
        show_plots: bool = False,
        run_cluster: str = "Serial",
        queue: str = "defq",
        reserved_memory: int = 4,
    ):
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.exp_root = Path(output_path) / experiment_name
        if not self.exp_root.is_dir():
            raise Exception(f"Experiment folder not found: {self.exp_root}")

        self.kw = dict(
            output_path=output_path,
            experiment_name=experiment_name,
            experiment_path=str(self.exp_root),
            outcome_label=outcome_label,
            outcome_type=outcome_type,
            instance_label=instance_label,
            sig_cutoff=float(sig_cutoff),
            show_plots=bool(show_plots),
        )

        self.run_cluster = run_cluster or "Serial"
        self.queue = queue
        self.reserved_memory = int(reserved_memory)

    def run(self):
        """
        Phase 9 is a single job per experiment (not per dataset).
        """
        if self.run_cluster == "Serial":
            self._run_one()
        elif self.run_cluster == "Local":
            with LocalCluster(processes=True, n_workers=num_cores, threads_per_worker=1) as cluster:
                with Client(cluster) as client:
                    dask.compute(
                        [dask.delayed(self._run_one)()],
                        scheduler=client,
                    )
        elif self.run_cluster in ("BashSLURM", "BashLSF"):
            self._submit_bash()
        else:
            client: Client = get_cluster(
                self.run_cluster, str(self.exp_root), self.queue, self.reserved_memory
            )
            dask.compute(
                [dask.delayed(self._run_one)()],
                scheduler=client,
            )

    def _run_one(self):
        DatasetCompareJob(**self.kw).run()

    def _submit_bash(self):
        """
        Submit a single experiment-level job via SLURM or LSF, using p9_jobsubmit.py.
        """
        job_ref = str(time.time())
        jobs = self.exp_root / "jobs"
        logs = self.exp_root / "logs"
        os.makedirs(jobs, exist_ok=True)
        os.makedirs(logs, exist_ok=True)

        sh = jobs / f"P9_{job_ref}_run.sh"
        launcher = "sbatch" if self.run_cluster == "BashSLURM" else "bsub <"
        script = Path(__file__).with_name("p9_jobsubmit.py")

        args = " ".join(
            [
                "python",
                str(script),
                "--output_path",
                self.output_path,
                "--experiment_name",
                self.experiment_name,
                "--outcome_label",
                self.kw["outcome_label"],
                "--outcome_type",
                self.kw["outcome_type"],
                "--instance_label",
                self.kw["instance_label"] or "",
                "--sig_cutoff",
                str(self.kw["sig_cutoff"]),
                "--show_plots",
                str(int(self.kw["show_plots"])),
            ]
        )

        with open(sh, "w") as f:
            f.write("#!/bin/bash\n")
            if self.run_cluster == "BashSLURM":
                f.write(f"#SBATCH -p {self.queue}\n")
                f.write(f"#SBATCH --job-name={job_ref}\n")
                f.write(f"#SBATCH --mem={self.reserved_memory}G\n")
                f.write(f"#SBATCH -o {logs}/P9_{job_ref}.o\n")
                f.write(f"#SBATCH -e {logs}/P9_{job_ref}.e\n")
                f.write(f"srun {args}\n")
            else:  # BashLSF
                f.write(f"#BSUB -q {self.queue}\n")
                f.write(f"#BSUB -J {job_ref}\n")
                f.write(f"#BSUB -R \"rusage[mem={self.reserved_memory}G]\"\n")
                f.write(f"#BSUB -M {self.reserved_memory}GB\n")
                f.write(f"#BSUB -o {logs}/P9_{job_ref}.o\n")
                f.write(f"#BSUB -e {logs}/P9_{job_ref}.e\n")
                f.write(f"{args}\n")

        os.system(f"{launcher} {sh}")
