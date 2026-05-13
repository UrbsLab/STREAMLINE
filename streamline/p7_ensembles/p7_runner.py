from __future__ import annotations
import os, time, logging, pickle
from pathlib import Path
import dask
from dask.distributed import Client, LocalCluster
from streamline.utils.cluster import get_cluster
from streamline.utils.runners import num_cores
from streamline.p7_ensembles.ensembles import EnsemblePhaseJob

class P7Runner:
    def __init__(
        self, output_path, experiment_name, n_splits,
        outcome_label="Class", outcome_type=None, instance_label=None,
        ensembles="hard_voting,soft_voting,stack_lr", base_models=None,
        meta_train_source="train",
        calibrate=0, calibrate_method="sigmoid", calibrate_cv=5,
        run_cluster="Serial", queue="defq", reserved_memory=4,
        random_state=0,
    ):
        self.exp_root = Path(output_path) / experiment_name
        if not self.exp_root.is_dir():
            raise Exception("Experiment folder not found.")
        self.output_path = output_path
        self.experiment_name = experiment_name
        metadata = self._load_metadata()
        resolved_outcome_type = outcome_type if outcome_type is not None else metadata.get("Outcome Type")
        self.kw = dict(
            n_splits=int(n_splits),
            outcome_label=outcome_label,
            outcome_type=resolved_outcome_type,
            instance_label=instance_label,
            ensembles=ensembles, base_models=base_models,
            meta_train_source=meta_train_source,
            calibrate=bool(calibrate), calibrate_method=calibrate_method, calibrate_cv=int(calibrate_cv),
            random_state=random_state,
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

    def _load_metadata(self):
        meta_path = self.exp_root / "metadata.pickle"
        if not meta_path.exists():
            return {}
        try:
            with meta_path.open("rb") as f:
                return pickle.load(f) or {}
        except Exception:
            return {}

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
            "--outcome_type", self.kw["outcome_type"] or "",
            "--instance_label", self.kw["instance_label"] or "",
            "--ensembles", self.kw["ensembles"] or "",
            "--base_models", self.kw["base_models"] or "",
            "--meta_train_source", self.kw.get("meta_train_source","train"),
            "--calibrate", str(int(self.kw["calibrate"])),
            "--calibrate_method", self.kw["calibrate_method"],
            "--calibrate_cv", str(self.kw["calibrate_cv"]), 
            "--output_path", self.output_path,
            "--experiment_name", self.experiment_name,
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
