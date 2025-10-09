# streamline/phases/p4_feature_importance/runner.py
import os, glob, json, time, pickle, logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import dask
from dask.distributed import Client, LocalCluster

from streamline.utils.runners import num_cores
from streamline.utils.cluster import get_cluster
from streamline.p4_feature_importance.importance import FeatureImportance
from streamline.p4_feature_importance.utils.fi_loader import list_importances

class P4Runner:
    def __init__(
        self,
        output_path: str,
        experiment_name: str,
        *,
        importance_id: "str | None" = None,
        importance_params: "Dict[str, Any] | None" = None,
        top_k: "int | None" = None,
        threshold: "float | None" = None,
        keep_original_features: "bool | None" = None,
        overwrite_cv: "bool | None" = None,
        outcome_label: "str | None" = None,
        outcome_type: "str | None" = None,
        instance_label: "str | None" = None,
        random_state: "int | None" = None,
        run_cluster: "str | bool" = False,
        queue: str = "defq",
        reserved_memory: int = 4,
    ):
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.run_cluster = run_cluster
        self.queue = queue
        self.reserved_memory = reserved_memory

        meta = self._load_metadata()
        self.importance_id = importance_id or meta.get("P4 importance Id", "mutualinformation")
        self.importance_params = importance_params or json.loads(meta.get("P4 importance Params","{}") or "{}")
        self.top_k = top_k if top_k is not None else meta.get("P4 TopK", None)
        self.threshold = threshold if threshold is not None else meta.get("P4 Threshold", None)
        self.keep_original_features = bool(keep_original_features if keep_original_features is not None else meta.get("P4 Keep Original Features", False))
        self.overwrite_cv = bool(overwrite_cv if overwrite_cv is not None else True)
        self.outcome_label = outcome_label or meta.get("Outcome Label", "Class")
        self.outcome_type = outcome_type or meta.get("Outcome Type", None)
        self.instance_label = instance_label if instance_label is not None else meta.get("Instance Label", None)
        self.random_state = random_state if random_state is not None else meta.get("Random Seed", 0)

        exp_root = os.path.join(self.output_path, self.experiment_name)
        if not os.path.exists(exp_root):
            raise Exception("Experiment must exist before phase 4 can begin")

    def run(self):
        exp_root = os.path.join(self.output_path, self.experiment_name)
        jobs: List[Tuple[str,str]] = []

        for name in os.listdir(exp_root):
            ds_dir = os.path.join(exp_root, name)
            if not os.path.isdir(ds_dir): continue
            if name in {"jobsCompleted","jobs","logs","dask_logs","DatasetComparisons"}: continue
            os.makedirs(os.path.join(ds_dir, "feature_importance"), exist_ok=True)
            for tr in sorted(glob.glob(os.path.join(ds_dir, "CVDatasets/*Train.csv"))):
                te = tr.replace("Train.csv","Test.csv")
                if os.path.exists(te): jobs.append((tr,te))
        if not jobs: raise Exception("No CV Train/Test pairs found for Phase 4.")

        mode = str(self.run_cluster) if self.run_cluster else "Serial"
        if mode == "Local":
            with LocalCluster(processes=True, n_workers=num_cores, threads_per_worker=1) as cluster:
                with Client(cluster) as client:
                    tasks = [dask.delayed(self._run_one)(tr, te) for tr, te in jobs]
                    dask.compute(tasks, scheduler=client)
        elif self.run_cluster and self.run_cluster not in ("BashSLURM","BashLSF"):
            client: Client = get_cluster(self.run_cluster, exp_root, self.queue, self.reserved_memory)
            tasks = [dask.delayed(self._run_one)(tr, te) for tr, te in jobs]
            dask.compute(tasks, scheduler=client)
        elif self.run_cluster in ("BashSLURM","BashLSF"):
            for tr, te in jobs: self._submit_bash_job(tr, te)
        else:
            for tr, te in jobs: self._run_one(tr, te)

        self._save_run_params(mode)

    def _run_one(self, tr: str, te: str):
        exp_root = os.path.join(self.output_path, self.experiment_name)
        FeatureImportance(
            cv_train_path=tr,
            cv_test_path=te,
            experiment_path=exp_root,
            importance_id=self.importance_id,
            importance_params=self.importance_params,
            top_k=self.top_k,
            threshold=self.threshold,
            keep_original_features=self.keep_original_features,
            overwrite_cv=self.overwrite_cv,
            outcome_label=self.outcome_label,
            outcome_type=self.outcome_type,
            instance_label=self.instance_label,
            random_state=self.random_state,
        ).run()

    def _load_metadata(self):
        path = os.path.join(self.output_path, self.experiment_name, "metadata.pickle")
        if os.path.exists(path):
            with open(path,"rb") as f:
                try: return pickle.load(f) or {}
                except Exception: return {}
        return {}

    def _save_run_params(self, mode: str):
        from datetime import datetime
        exp_root = os.path.join(self.output_path, self.experiment_name)
        params_file = os.path.join(exp_root, "run_params.pickle")
        this_run = {
            "phase": "p4_feature_importance",
            "run_mode": mode,
            "importance_id": self.importance_id,
            "importance_params": self.importance_params,
            "top_k": self.top_k,
            "threshold": self.threshold,
            "keep_original_features": self.keep_original_features,
            "overwrite_cv": self.overwrite_cv,
            "outcome_label": self.outcome_label,
            "outcome_type": self.outcome_type,
            "instance_label": self.instance_label,
            "random_state": self.random_state,
        }
        all_params = {}
        if os.path.exists(params_file):
            with open(params_file,"rb") as f:
                try: all_params = pickle.load(f)
                except Exception: all_params = {}
        all_params[datetime.now().isoformat()] = this_run
        with open(params_file,"wb") as f: pickle.dump(all_params, f)

    # ---- bash submit ----
    def _submit_bash_job(self, tr: str, te: str):
        job_ref = str(time.time())
        run_dir = os.path.join(self.output_path, self.experiment_name)
        os.makedirs(os.path.join(run_dir,"jobs"), exist_ok=True)
        os.makedirs(os.path.join(run_dir,"logs"), exist_ok=True)
        job_name = os.path.join(run_dir, f"jobs/P4_{job_ref}_run.sh")
        launcher = "sbatch" if self.run_cluster == "BashSLURM" else "bsub <"

        with open(job_name, "w") as sh:
            sh.write("#!/bin/bash\n")
            if self.run_cluster == "BashSLURM":
                sh.write(f"#SBATCH -p {self.queue}\n")
                sh.write(f"#SBATCH --job-name={job_ref}\n")
                sh.write(f"#SBATCH --mem={self.reserved_memory}G\n")
                sh.write(f"#SBATCH -o {run_dir}/logs/P4_{job_ref}.o\n")
                sh.write(f"#SBATCH -e {run_dir}/logs/P4_{job_ref}.e\n")
                sh.write("srun " + self._bash_submit_command(tr, te) + "\n")
            else:
                sh.write(f"#BSUB -q {self.queue}\n")
                sh.write(f"#BSUB -J {job_ref}\n")
                sh.write(f"#BSUB -R \"rusage[mem={self.reserved_memory}G]\"\n")
                sh.write(f"#BSUB -M {self.reserved_memory}GB\n")
                sh.write(f"#BSUB -o {run_dir}/logs/P4_{job_ref}.o\n")
                sh.write(f"#BSUB -e {run_dir}/logs/P4_{job_ref}.e\n")
                sh.write(self._bash_submit_command(tr, te) + "\n")

        os.system(f"{launcher} {job_name}")

    def _bash_submit_command(self, tr: str, te: str) -> str:
        script_path = str(Path(__file__).parent / "p4_jobsubmit.py")
        exp_root = os.path.join(self.output_path, self.experiment_name)
        args = [
            "python", script_path,
            "--cv_train_path", tr,
            "--cv_test_path", te,
            "--experiment_path", exp_root,
            "--importance_id", self.importance_id or "mutualinformation",
            "--importance_params", json.dumps(self.importance_params or {}),
            "--top_k", str(self.top_k) if self.top_k is not None else "",
            "--threshold", str(self.threshold) if self.threshold is not None else "",
            "--keep_original_features", str(int(self.keep_original_features)),
            "--overwrite_cv", str(int(self.overwrite_cv)),
            "--outcome_label", self.outcome_label or "",
            "--outcome_type", self.outcome_type or "",
            "--instance_label", self.instance_label or "",
            "--random_state", str(self.random_state) if self.random_state is not None else "",
        ]
        return " ".join(args)
