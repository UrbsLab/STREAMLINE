# streamline/phases/p3_feature_learning/runner.py
import os, glob, json, time, pickle, logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import dask
from dask.distributed import Client, LocalCluster

from streamline.utils.runners import num_cores, quote_command_parts, run_dask_tasks, run_parallel_jobs
from streamline.utils.cluster import get_cluster
from streamline.p3_feature_learning.feature_learn import FeatureLearn
from streamline.p3_feature_learning.utils.fl_loader import list_learners

class P3Runner:
    def __init__(
        self,
        output_path: str,
        experiment_name: str,
        *,
        learner_id: "str | None" = None,
        learner_params: "Dict[str, Any] | None" = None,
        feature_namespace: "str | None" = None,
        keep_original_features: "bool | None" = None,
        overwrite_cv: "bool | None" = None,
        outcome_label: "str | None" = None,
        instance_label: "str | None" = None,
        random_state: "int | None" = None,
        run_cluster: "str | bool" = False,   # False | Local | Parallel | BashSLURM | BashLSF | "<dask-cluster>"
        queue: str = "defq",
        reserved_memory: int = 4,
    ):
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.run_cluster = run_cluster
        self.queue = queue
        self.reserved_memory = reserved_memory

        meta = self._load_metadata()
        self.learner_id = learner_id or meta.get("P3 Learner Id", "pca")
        self.learner_params = learner_params or json.loads(meta.get("P3 Learner Params", "{}") or "{}")
        self.feature_namespace = feature_namespace or meta.get("P3 Feature Namespace", "FL_PCA")
        self.keep_original_features = bool(keep_original_features if keep_original_features is not None else meta.get("P3 Keep Original Features", True))
        self.overwrite_cv = bool(overwrite_cv if overwrite_cv is not None else True)
        self.outcome_label = outcome_label or meta.get("Outcome Label", "Class")
        self.instance_label = instance_label if instance_label is not None else meta.get("Instance Label", None)
        self.random_state = random_state if random_state is not None else meta.get("Random Seed", 0)

        exp_root = os.path.join(self.output_path, self.experiment_name)
        if not os.path.exists(exp_root):
            raise Exception("Experiment must exist (from previous phases) before phase 3 can begin")

    # ---- main ----
    def run(self):
        exp_root = os.path.join(self.output_path, self.experiment_name)
        jobs: List[Tuple[str, str]] = []
        logging.info(
            "Phase 3 starting for experiment '%s' with learner '%s' and namespace '%s'.",
            self.experiment_name,
            self.learner_id,
            self.feature_namespace,
        )
        dataset_names = set()

        # discover CV pairs
        for name in os.listdir(exp_root):
            ds_dir = os.path.join(exp_root, name)
            if not os.path.isdir(ds_dir): continue
            if name in {"jobsCompleted", "jobs", "logs", "dask_logs", "DatasetComparisons"}: continue
            os.makedirs(os.path.join(ds_dir, "feature_learning"), exist_ok=True)
            for tr in sorted(glob.glob(os.path.join(ds_dir, "CVDatasets/*Train.csv"))):
                te = tr.replace("Train.csv", "Test.csv")
                if os.path.exists(te):
                    jobs.append((tr, te))
                    dataset_names.add(name)
        if not jobs: raise Exception("No CV Train/Test CSV pairs found for Phase 3.")
        logging.info(
            "Phase 3 discovered %d CV train/test pairs across %d dataset(s): %s",
            len(jobs),
            len(dataset_names),
            ", ".join(sorted(dataset_names)),
        )

        mode = str(self.run_cluster) if self.run_cluster else "Serial"
        logging.info("Phase 3 submitting %d jobs in mode '%s'.", len(jobs), mode)
        if mode == "Local":
            with LocalCluster(processes=True, n_workers=num_cores, threads_per_worker=1) as cluster:
                with Client(cluster) as client:
                    tasks = [dask.delayed(self._run_one)(tr, te) for tr, te in jobs]
                    run_dask_tasks(tasks, client, label="Phase 3 Dask jobs")
        elif mode == "Parallel":
            run_parallel_jobs(self._run_one, jobs, label="Phase 3 Parallel jobs")
        elif self.run_cluster and self.run_cluster != "Serial" and self.run_cluster not in ("BashSLURM", "BashLSF"):
            client: Client = get_cluster(self.run_cluster, exp_root, self.queue, self.reserved_memory)
            tasks = [dask.delayed(self._run_one)(tr, te) for tr, te in jobs]
            run_dask_tasks(tasks, client, label="Phase 3 Dask jobs")
        elif self.run_cluster in ("BashSLURM", "BashLSF"):
            for tr, te in jobs: self._submit_bash_job(tr, te)
        else:
            for tr, te in jobs: self._run_one(tr, te)

        self._save_run_params(mode)
        logging.info("Phase 3 completed: %d jobs finished.", len(jobs))

    # ---- helpers ----
    def _run_one(self, tr: str, te: str):
        exp_root = os.path.join(self.output_path, self.experiment_name)
        FeatureLearn(
            cv_train_path=tr,
            cv_test_path=te,
            experiment_path=exp_root,
            learner_id=self.learner_id,
            learner_params=self.learner_params,
            feature_namespace=self.feature_namespace,
            keep_original_features=self.keep_original_features,
            overwrite_cv=self.overwrite_cv,
            outcome_label=self.outcome_label,
            instance_label=self.instance_label,
            random_state=self.random_state,
        ).run()

    def _load_metadata(self):
        path = os.path.join(self.output_path, self.experiment_name, "metadata.pickle")
        if os.path.exists(path):
            with open(path, "rb") as f:
                try: return pickle.load(f) or {}
                except Exception: return {}
        return {}

    def _save_run_params(self, mode: str):
        from datetime import datetime
        exp_root = os.path.join(self.output_path, self.experiment_name)
        params_file = os.path.join(exp_root, "run_params.pickle")
        this_run = {
            "phase": "p3_feature_learning",
            "run_mode": mode,
            "learner_id": self.learner_id,
            "learner_params": self.learner_params,
            "feature_namespace": self.feature_namespace,
            "keep_original_features": self.keep_original_features,
            "overwrite_cv": self.overwrite_cv,
            "outcome_label": self.outcome_label,
            "instance_label": self.instance_label,
            "random_state": self.random_state,
        }
        all_params = {}
        if os.path.exists(params_file):
            with open(params_file, "rb") as f:
                try: all_params = pickle.load(f)
                except Exception: all_params = {}
        all_params[datetime.now().isoformat()] = this_run
        with open(params_file, "wb") as f: pickle.dump(all_params, f)

    # ---- bash submit ----
    def _submit_bash_job(self, cv_train_path: str, cv_test_path: str):
        job_ref = str(time.time())
        run_dir = os.path.join(self.output_path, self.experiment_name)
        os.makedirs(os.path.join(run_dir, "jobs"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
        job_name = os.path.join(run_dir, f"jobs/P3_{job_ref}_run.sh")

        if self.run_cluster == "BashSLURM":
            launcher = "sbatch"
        else:
            launcher = "bsub <"

        with open(job_name, "w") as sh:
            sh.write("#!/bin/bash\n")
            if self.run_cluster == "BashSLURM":
                sh.write(f"#SBATCH -p {self.queue}\n")
                sh.write(f"#SBATCH --job-name={job_ref}\n")
                sh.write(f"#SBATCH --mem={self.reserved_memory}G\n")
                sh.write(f"#SBATCH -o {run_dir}/logs/P3_{job_ref}.o\n")
                sh.write(f"#SBATCH -e {run_dir}/logs/P3_{job_ref}.e\n")
                cmd = self._bash_submit_command(cv_train_path, cv_test_path)
                sh.write("srun " + cmd + "\n")
            else:
                sh.write(f"#BSUB -q {self.queue}\n")
                sh.write(f"#BSUB -J {job_ref}\n")
                sh.write(f"#BSUB -R \"rusage[mem={self.reserved_memory}G]\"\n")
                sh.write(f"#BSUB -M {self.reserved_memory}GB\n")
                sh.write(f"#BSUB -o {run_dir}/logs/P3_{job_ref}.o\n")
                sh.write(f"#BSUB -e {run_dir}/logs/P3_{job_ref}.e\n")
                cmd = self._bash_submit_command(cv_train_path, cv_test_path)
                sh.write(cmd + "\n")

        os.system(f"{launcher} {quote_command_parts([job_name])}")
        logging.info("Phase 3 submitted cluster job script: %s", job_name)

    def _bash_submit_command(self, tr: str, te: str) -> str:
        script_path = str(Path(__file__).parent / "p3_jobsubmit.py")
        exp_root = os.path.join(self.output_path, self.experiment_name)
        args = [
            "python", script_path,
            "--cv_train_path", tr,
            "--cv_test_path", te,
            "--experiment_path", exp_root,
            "--learner_id", self.learner_id or "pca",
            "--learner_params", json.dumps(self.learner_params or {}),
            "--feature_namespace", self.feature_namespace,
            "--keep_original_features", str(int(self.keep_original_features)),
            "--overwrite_cv", str(int(self.overwrite_cv)),
            "--outcome_label", self.outcome_label or "",
            "--instance_label", self.instance_label or "",
            "--random_state", str(self.random_state) if self.random_state is not None else "",
        ]
        return quote_command_parts(args)
