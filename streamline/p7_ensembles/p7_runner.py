from __future__ import annotations
import os, time, logging
from pathlib import Path
import dask
from dask.distributed import Client, LocalCluster
from streamline.utils.cluster import get_cluster
from streamline.utils.runners import num_cores
from streamline.p7_ensembles.ensembles import EnsemblePhaseJob

class P7Runner:
    def __init__(
        self, output_path, experiment_name, n_splits,
        outcome_label="Class", instance_label=None,
        ensembles="hard_voting,soft_voting,stack_lr", base_models=None,
        scoring_metric="balanced_accuracy", metric_direction="maximize",
        calibrate=0, calibrate_method="sigmoid", calibrate_cv=5,
        run_cluster="Serial", queue="defq", reserved_memory=4,
        random_state=0, n_jobs=None,
    ):
        self.exp_root = Path(output_path) / experiment_name
        if not self.exp_root.is_dir():
            raise Exception("Experiment folder not found.")
        self.kw = dict(
            n_splits=int(n_splits),
            outcome_label=outcome_label, instance_label=instance_label,
            ensembles=ensembles, base_model_filter=base_models,
            scoring_metric=scoring_metric, metric_direction=metric_direction,
            calibrate=bool(calibrate), calibrate_method=calibrate_method, calibrate_cv=int(calibrate_cv),
            random_state=random_state, n_jobs=n_jobs,
            output_path=output_path, experiment_name=experiment_name,
        )
        self.run_cluster = run_cluster or "Serial"
        self.queue = queue; self.reserved_memory = int(reserved_memory)

    def run(self):
        datasets = [p for p in sorted(self.exp_root.iterdir())
                    if p.is_dir() and (p / "CVDatasets").is_dir()
                    and p.name not in {"jobs","logs","jobsCompleted","dask_logs","DatasetComparisons"}]
        if self.run_cluster == "Serial":
            for ds in datasets: self._run_one(ds)
        elif self.run_cluster == "Local":
            with LocalCluster(processes=True, n_workers=num_cores, threads_per_worker=1) as cluster:
                with Client(cluster) as client:
                    dask.compute([dask.delayed(self._run_one)(ds) for ds in datasets], scheduler=client)
        elif self.run_cluster in ("BashSLURM","BashLSF"):
            for ds in datasets: self._submit_bash(ds)
        else:
            client: Client = get_cluster(self.run_cluster, str(self.exp_root), self.queue, self.reserved_memory)
            dask.compute([dask.delayed(self._run_one)(ds) for ds in datasets], scheduler=client)

    def _run_one(self, ds):
        EnsemblePhaseJob(dataset_dir=str(ds), **self.kw).run()

    def _submit_bash(self, ds):
        job_ref = str(time.time()); jobs = self.exp_root / "jobs"; logs = self.exp_root / "logs"
        os.makedirs(jobs, exist_ok=True); os.makedirs(logs, exist_ok=True)
        sh = jobs / f"P7_{job_ref}_run.sh"
        launcher = "sbatch" if self.run_cluster=="BashSLURM" else "bsub <"
        script = Path(__file__).with_name("p7_jobsubmit.py")
        args = " ".join([
            "python", str(script),
            "--dataset_dir", str(ds),
            "--n_splits", str(self.kw["n_splits"]),
            "--outcome_label", self.kw["outcome_label"],
            "--instance_label", self.kw["instance_label"] or "",
            "--ensembles", self.kw["ensembles"] or "",
            "--base_models", self.kw["base_model_filter"] or "",
            "--output_path", self.kw["output_path"],
            "--experiment_name", self.kw["experiment_name"],
        ])
        with open(sh, "w") as f:
            f.write("#!/bin/bash\n")
            if self.run_cluster=="BashSLURM":
                f.write(f"#SBATCH -p {self.queue}\n#SBATCH --job-name={job_ref}\n#SBATCH --mem={self.reserved_memory}G\n")
                f.write(f"#SBATCH -o {logs}/P7_{job_ref}.o\n#SBATCH -e {logs}/P7_{job_ref}.e\nsrun {args}\n")
            else:
                f.write(f"#BSUB -q {self.queue}\n#BSUB -J {job_ref}\n#BSUB -R \"rusage[mem={self.reserved_memory}G]\"\n")
                f.write(f"#BSUB -M {self.reserved_memory}GB\n#BSUB -o {logs}/P7_{job_ref}.o\n#BSUB -e {logs}/P7_{job_ref}.e\n{args}\n")
        os.system(f"{launcher} {sh}")
