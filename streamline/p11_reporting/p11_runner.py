# streamline/p11_reporting/p11_runner.py
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import dask
from dask.distributed import Client, LocalCluster

from streamline.p11_reporting.reporting import ReportPhaseJob
from streamline.utils.cluster import get_cluster
from streamline.utils.runners import num_cores, quote_command_parts, run_dask_tasks, run_parallel_functions


class P11Runner:
    """
    Phase 11 runner for experiment-level report generation.

    Supports both path styles:
    - experiment_path
    - output_path + experiment_name
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        experiment_name: Optional[str] = None,
        experiment_path: Optional[str] = None,
        reporting_dir: Optional[str] = None,
        report_mode: str = "standard",  # standard | replication
        outcome_label: Optional[str] = "Class",
        outcome_type: Optional[str] = "Binary",
        instance_label: Optional[str] = None,
        make_pdf: bool = True,
        enable_plots: bool = True,
        reuse_existing_figures: bool = True,
        run_cluster: str = "Serial",  # Serial | Local | Parallel | BashSLURM | BashLSF | <dask-cluster-name>
        queue: str = "defq",
        reserved_memory: int = 4,
    ):
        if experiment_path:
            self.exp_root = Path(experiment_path).resolve()
            if output_path is None:
                output_path = str(self.exp_root.parent)
            if experiment_name is None:
                experiment_name = self.exp_root.name
        else:
            if not output_path or not experiment_name:
                raise ValueError("Provide experiment_path OR (output_path and experiment_name).")
            self.exp_root = (Path(output_path) / experiment_name).resolve()

        if not self.exp_root.is_dir():
            raise FileNotFoundError(f"Experiment folder not found: {self.exp_root}")

        self.output_path = str(output_path) if output_path else str(self.exp_root.parent)
        self.experiment_name = str(experiment_name) if experiment_name else self.exp_root.name

        report_mode_norm = str(report_mode or "standard").strip().lower()
        if report_mode_norm not in {"standard", "replication"}:
            raise ValueError("report_mode must be one of: standard, replication")

        # kwargs handed directly to ReportPhaseJob
        self.kw = dict(
            output_path=self.output_path,
            experiment_name=self.experiment_name,
            experiment_path=str(self.exp_root),
            reporting_dir=reporting_dir,
            report_mode=report_mode_norm,
            outcome_label=outcome_label,
            outcome_type=outcome_type,
            instance_label=instance_label,
            make_pdf=bool(make_pdf),
            enable_plots=bool(enable_plots),
            reuse_existing_figures=bool(reuse_existing_figures),
        )

        self.run_cluster = run_cluster or "Serial"
        self.queue = queue
        self.reserved_memory = int(reserved_memory)

    def run(self):
        """Phase 11 is a single experiment-level job."""
        if self.run_cluster == "Serial":
            self._run_one()

        elif self.run_cluster == "Local":
            # Local dask (mainly for development on multi-core machines)
            with LocalCluster(processes=True, n_workers=num_cores, threads_per_worker=1) as cluster:
                with Client(cluster) as client:
                    run_dask_tasks([dask.delayed(self._run_one)()], client, label="Phase 11 Dask jobs")

        elif self.run_cluster == "Parallel":
            run_parallel_functions([self._run_one], label="Phase 11 Parallel jobs")

        elif self.run_cluster in ("BashSLURM", "BashLSF"):
            self._submit_bash()

        else:
            # Named dask cluster (e.g. a shared HPC scheduler)
            client: Client = get_cluster(
                self.run_cluster, str(self.exp_root), self.queue, self.reserved_memory
            )
            run_dask_tasks([dask.delayed(self._run_one)()], client, label="Phase 11 Dask jobs")

    def _run_one(self):
        ReportPhaseJob(**self.kw).run()

    def _submit_bash(self):
        """
        Submit a single experiment-level job via SLURM or LSF using p11_jobsubmit.py.
        """
        job_ref = str(time.time())
        jobs = self.exp_root / "jobs"
        logs = self.exp_root / "logs"
        os.makedirs(jobs, exist_ok=True)
        os.makedirs(logs, exist_ok=True)

        sh = jobs / f"P11_{job_ref}_run.sh"
        launcher = "sbatch" if self.run_cluster == "BashSLURM" else "bsub <"
        script = Path(__file__).with_name("p11_jobsubmit.py")

        args = [
            "python",
            str(script),
            "--experiment_path",
            str(self.exp_root),
            "--outcome_label",
            str(self.kw.get("outcome_label") or ""),
            "--outcome_type",
            str(self.kw.get("outcome_type") or ""),
            "--instance_label",
            str(self.kw.get("instance_label") or ""),
            "--make_pdf",
            str(int(bool(self.kw.get("make_pdf", True)))),
            "--enable_plots",
            str(int(bool(self.kw.get("enable_plots", True)))),
            "--reuse_existing_figures",
            str(int(bool(self.kw.get("reuse_existing_figures", True)))),
            "--queue",
            str(self.queue),
            "--reserved_memory",
            str(int(self.reserved_memory)),
        ]

        reporting_dir = self.kw.get("reporting_dir")
        if reporting_dir:
            args.extend(["--reporting_dir", str(reporting_dir)])
        args.extend(["--report_mode", str(self.kw.get("report_mode") or "standard")])

        arg_str = quote_command_parts(args)

        with sh.open("w", encoding="utf-8") as f:
            f.write("#!/bin/bash\n")
            if self.run_cluster == "BashSLURM":
                f.write(f"#SBATCH -p {self.queue}\n")
                f.write(f"#SBATCH --job-name={job_ref}\n")
                f.write(f"#SBATCH --mem={self.reserved_memory}G\n")
                f.write(f"#SBATCH -o {logs}/P11_{job_ref}.o\n")
                f.write(f"#SBATCH -e {logs}/P11_{job_ref}.e\n")
                f.write(f"srun {arg_str}\n")
            else:  # BashLSF
                f.write(f"#BSUB -q {self.queue}\n")
                f.write(f"#BSUB -J {job_ref}\n")
                f.write(f"#BSUB -R \"rusage[mem={self.reserved_memory}G]\"\n")
                f.write(f"#BSUB -M {self.reserved_memory}GB\n")
                f.write(f"#BSUB -o {logs}/P11_{job_ref}.o\n")
                f.write(f"#BSUB -e {logs}/P11_{job_ref}.e\n")
                f.write(f"{arg_str}\n")

        os.system(f"{launcher} {quote_command_parts([sh])}")
