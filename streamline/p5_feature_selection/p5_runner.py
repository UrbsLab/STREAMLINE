# streamline/phases/p5_feature_selection/runner.py  (replace the class with this version)
from __future__ import annotations
import os, time, json
from pathlib import Path
from typing import Optional, List

import dask
from dask.distributed import Client, LocalCluster
import logging
logger = logging.getLogger("distributed.worker")
logger.setLevel(logging.WARNING)

from streamline.utils.runners import num_cores
from streamline.utils.cluster import get_cluster
from streamline.p5_feature_selection.feature_selection import FeatureSelection
from streamline.p5_feature_selection.utils.fi_resolver import  _normalize_algorithms, _discover_algorithms



class P5Runner:
    """
    Phase 5 runner: one job per dataset directory.

    run_cluster modes:
      • "Serial" (default)
      • "Local"
      • "BashSLURM"
      • "BashLSF"
      • "<dask-cluster-name>" → use get_cluster(name, ...)
    """
    def __init__(
        self,
        output_path: str,
        experiment_name: str,
        *,
        algorithms: List[str] | str | None = "auto",   # <-- default auto
        n_splits: int = 10,
        class_label: str = "Class",
        instance_label: Optional[str] = None,
        max_features_to_keep: int = 2000,
        filter_poor_features: bool = True,
        overwrite_cv: bool = False,
        # strategy
        selector_id: str = "default",
        selector_params: dict | None = None,
        # plotting/summary
        export_scores: bool = True,
        top_features: int = 20,
        show_plots: bool = False,
        # submission / cluster
        run_cluster: str = "Serial",
        queue: str = "defq",
        reserved_memory: int = 4,
        # discovery behavior
        strict_discovery: bool = False,  # require all n_splits files present for an algorithm
    ):
        self.output_path = output_path
        self.experiment_name = experiment_name
        self._algorithms_raw = algorithms  # keep raw for "auto"
        self.n_splits = int(n_splits)
        self.class_label = class_label
        self.instance_label = instance_label
        self.max_features_to_keep = int(max_features_to_keep)
        self.filter_poor_features = bool(filter_poor_features)
        self.overwrite_cv = bool(overwrite_cv)
        self.selector_id = selector_id or "default"
        self.selector_params = selector_params or {}
        self.export_scores = bool(export_scores)
        self.top_features = int(top_features)
        self.show_plots = bool(show_plots)

        self.run_cluster = run_cluster or "Serial"
        self.queue = queue
        self.reserved_memory = int(reserved_memory)
        self.strict_discovery = bool(strict_discovery)

        self.exp_root = os.path.join(self.output_path, self.experiment_name)
        if not os.path.isdir(self.exp_root):
            raise Exception("Experiment must exist before phase 5 can begin")

        if self.run_cluster in ("BashSLURM","BashLSF"):
            os.makedirs(os.path.join(self.exp_root, "jobs"), exist_ok=True)
            os.makedirs(os.path.join(self.exp_root, "logs"), exist_ok=True)

    # ----------------------------
    # Main
    # ----------------------------
    def run(self):
        datasets = [
            os.path.join(self.exp_root, name)
            for name in sorted(os.listdir(self.exp_root))
            if os.path.isdir(os.path.join(self.exp_root, name))
            and name not in {"jobsCompleted","jobs","logs","dask_logs","DatasetComparisons"}
            and os.path.isdir(os.path.join(self.exp_root, name, "CVDatasets"))
        ]
        if not datasets:
            logging.warning("No datasets found for Phase 5 under %s", self.exp_root)
            return

        mode = str(self.run_cluster)
        if mode == "Serial":
            for ds_dir in datasets:
                self._run_one(ds_dir)
        elif mode == "Local":
            n_workers = num_cores
            with LocalCluster(processes=True, n_workers=n_workers, threads_per_worker=1) as cluster:
                with Client(cluster) as client:
                    tasks = [dask.delayed(self._run_one)(ds_dir) for ds_dir in datasets]
                    dask.compute(tasks, scheduler=client)
        elif mode in ("BashSLURM","BashLSF"):
            for ds_dir in datasets:
                # discover now so the submitted script has a concrete list
                algs = self._resolve_algorithms_for_dataset(ds_dir)
                if not algs:
                    logging.warning("Skipping dataset with no discovered algorithms: %s", ds_dir)
                    continue
                self._submit_bash_job(ds_dir, mode, algs)
        else:
            client: Client = get_cluster(
                mode,
                os.path.join(self.output_path, self.experiment_name),
                self.queue,
                self.reserved_memory
            )
            tasks = [dask.delayed(self._run_one)(ds_dir) for ds_dir in datasets]
            dask.compute(tasks, scheduler=client)

    # ----------------------------
    # Helpers
    # ----------------------------
    def _resolve_algorithms_for_dataset(self, dataset_dir: str) -> List[str]:
        """Return algorithms to use for this dataset."""
        if self._algorithms_raw in (None, "", "auto", "AUTO", "Auto"):
            algs = _discover_algorithms(dataset_dir, self.n_splits, strict=self.strict_discovery)
            if not algs:
                logging.warning("Phase 5 auto-discovery found no algorithms in %s/feature_importance", dataset_dir)
            else:
                logging.info("Phase 5 discovered algorithms for %s: %s",
                             os.path.basename(dataset_dir), ", ".join(algs))
            return algs
        # normalize any user-specified list/CSV (map MI→mutualinformation etc.)
        return _normalize_algorithms(self._algorithms_raw)

    def _run_one(self, dataset_dir: str):
        algs = self._resolve_algorithms_for_dataset(dataset_dir)
        if not algs:
            logging.warning("No algorithms to run in dataset %s; skipping.", dataset_dir)
            return

        FeatureSelection(
            dataset_dir=dataset_dir,
            n_splits=self.n_splits,
            algorithms=algs,  # pass concrete list
            class_label=self.class_label,
            instance_label=self.instance_label,
            max_features_to_keep=self.max_features_to_keep,
            filter_poor_features=self.filter_poor_features,
            overwrite_cv=self.overwrite_cv,
            selector_id=self.selector_id,
            selector_params=self.selector_params,
            export_scores=self.export_scores,
            top_features=self.top_features,
            show_plots=self.show_plots,
        ).run()

    def _submit_bash_job(self, dataset_dir: str, mode: str, algorithms_for_ds: List[str]):
        job_ref = str(time.time())
        run_dir = self.exp_root
        job_name = os.path.join(run_dir, f'jobs/P5_{job_ref}_run.sh')
        launcher = 'sbatch' if mode == "BashSLURM" else 'bsub <'

        script_path = str(Path(__file__).parent / "p5_jobsubmit.py")
        args = [
            "python", script_path,
            "--dataset_dir", dataset_dir,
            "--n_splits", str(self.n_splits),
            "--algorithms", ",".join(algorithms_for_ds),
            "--class_label", self.class_label or "Class",
            "--instance_label", self.instance_label or "",
            "--max_features_to_keep", str(self.max_features_to_keep),
            "--filter_poor_features", "1" if self.filter_poor_features else "0",
            "--overwrite_cv", "1" if self.overwrite_cv else "0",
            "--selector_id", self.selector_id or "default",
            "--selector_params", json.dumps(self.selector_params or {}),
            "--export_scores", "1" if self.export_scores else "0",
            "--top_features", str(self.top_features),
            "--show_plots", "1" if self.show_plots else "0",
        ]
        cmd = " ".join(args)

        with open(job_name, "w") as sh:
            sh.write("#!/bin/bash\n")
            if mode == "BashSLURM":
                sh.write(f"#SBATCH -p {self.queue}\n")
                sh.write(f"#SBATCH --job-name={job_ref}\n")
                sh.write(f"#SBATCH --mem={self.reserved_memory}G\n")
                sh.write(f"#SBATCH -o {run_dir}/logs/P5_{job_ref}.o\n")
                sh.write(f"#SBATCH -e {run_dir}/logs/P5_{job_ref}.e\n")
                sh.write("srun " + cmd + "\n")
            else:
                sh.write(f"#BSUB -q {self.queue}\n")
                sh.write(f"#BSUB -J {job_ref}\n")
                sh.write(f"#BSUB -R \"rusage[mem={self.reserved_memory}G]\"\n")
                sh.write(f"#BSUB -M {self.reserved_memory}GB\n")
                sh.write(f"#BSUB -o {run_dir}/logs/P5_{job_ref}.o\n")
                sh.write(f"#BSUB -e {run_dir}/logs/P5_{job_ref}.e\n")
                sh.write(cmd + "\n")

        os.system(f"{launcher} {job_name}")
