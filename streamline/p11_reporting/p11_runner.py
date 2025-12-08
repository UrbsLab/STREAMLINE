# streamline/p10_reporting/p10_runner.py
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import dask
from dask.distributed import Client, LocalCluster

from streamline.utils.cluster import get_cluster
from streamline.utils.runners import num_cores
from streamline.p11_reporting.reporting import ReportPhaseJob


class P11Runner:
    """
    Phase 10 Runner: experiment-level HTML/PDF reporting (all datasets, models, ensembles).
    This is a single job per experiment (not per dataset), similar to P9Runner.
    """

    def __init__(
        self,
        output_path: str,
        experiment_name: str,
        outcome_label: str = "Class",
        outcome_type: str = "Binary",
        instance_label: Optional[str] = None,
        report_title: Optional[str] = None,
        micro_average: str = "micro",          # "micro" or "macro", passed through to plots
        include_ensembles: bool = True,
        show_plots: bool = False,
        run_cluster: str = "Serial",           # Serial | Local | BashSLURM | BashLSF | <dask-cluster-name>
        queue: str = "defq",
        reserved_memory: int = 4,
    ):
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.exp_root = Path(output_path) / experiment_name
        if not self.exp_root.is_dir():
            raise Exception(f"Experiment folder not found: {self.exp_root}")

        # kwargs handed directly to ReportingPhaseJob
        self.kw = dict(
            output_path=output_path,
            experiment_name=experiment_name,
            outcome_label=outcome_label,
            outcome_type=outcome_type,
            instance_label=instance_label,
            make_pdf=True,
            # report_title=report_title,
            # micro_average=micro_average,
            # include_ensembles=bool(include_ensembles),
            # show_plots=bool(show_plots),
        )

        self.run_cluster = run_cluster or "Serial"
        self.queue = queue
        self.reserved_memory = int(reserved_memory)

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    def run(self):
        """
        Phase 10 is a single experiment-level job (like Phase 9).
        """
        if self.run_cluster == "Serial":
            self._run_one()

        elif self.run_cluster == "Local":
            # Local dask (mainly for dev on multi-core machines)
            with LocalCluster(processes=True, n_workers=num_cores, threads_per_worker=1) as cluster:
                with Client(cluster) as client:
                    dask.compute(
                        [dask.delayed(self._run_one)()],
                        scheduler=client,
                    )

        elif self.run_cluster in ("BashSLURM", "BashLSF"):
            # Legacy-style bash script submission for SLURM / LSF
            self._submit_bash()

        else:
            # Named dask cluster (e.g. a shared HPC scheduler)
            client: Client = get_cluster(
                self.run_cluster, str(self.exp_root), self.queue, self.reserved_memory
            )
            dask.compute(
                [dask.delayed(self._run_one)()],
                scheduler=client,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_one(self):
        ReportPhaseJob(**self.kw).run()

    def _submit_bash(self):
        """
        Submit a single experiment-level job via SLURM or LSF, using p10_jobsubmit.py.
        Mirrors P9Runner._submit_bash.
        """
        job_ref = str(time.time())
        jobs = self.exp_root / "jobs"
        logs = self.exp_root / "logs"
        os.makedirs(jobs, exist_ok=True)
        os.makedirs(logs, exist_ok=True)

        sh = jobs / f"P10_{job_ref}_run.sh"
        launcher = "sbatch" if self.run_cluster == "BashSLURM" else "bsub <"
        script = Path(__file__).with_name("p11_jobsubmit.py")

        args = [
            "python",
            str(script),
            "--output_path", self.output_path,
            "--experiment_name", self.experiment_name,
            "--outcome_label", self.kw["outcome_label"],
            "--outcome_type", self.kw["outcome_type"],
            "--instance_label", self.kw["instance_label"] or "",
            "--report_title", self.kw.get("report_title") or "",
            "--micro_average", self.kw.get("micro_average", "micro"),
            "--include_ensembles", str(int(bool(self.kw.get("include_ensembles", True)))),
            "--show_plots", str(int(bool(self.kw["show_plots"]))),
        ]
        arg_str = " ".join(args)

        with open(sh, "w") as f:
            f.write("#!/bin/bash\n")
            if self.run_cluster == "BashSLURM":
                f.write(f"#SBATCH -p {self.queue}\n")
                f.write(f"#SBATCH --job-name={job_ref}\n")
                f.write(f"#SBATCH --mem={self.reserved_memory}G\n")
                f.write(f"#SBATCH -o {logs}/P10_{job_ref}.o\n")
                f.write(f"#SBATCH -e {logs}/P10_{job_ref}.e\n")
                f.write(f"srun {arg_str}\n")
            else:  # BashLSF
                f.write(f"#BSUB -q {self.queue}\n")
                f.write(f"#BSUB -J {job_ref}\n")
                f.write(f"#BSUB -R \"rusage[mem={self.reserved_memory}G]\"\n")
                f.write(f"#BSUB -M {self.reserved_memory}GB\n")
                f.write(f"#BSUB -o {logs}/P10_{job_ref}.o\n")
                f.write(f"#BSUB -e {logs}/P10_{job_ref}.e\n")
                f.write(f"{arg_str}\n")

        os.system(f"{launcher} {sh}")
