# streamline/p4_feature_importance/runner.py
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
    """
    Phase 4: Feature Importance
    - models: list[str] of model ids (e.g., ["mutualinformation","multisurf"])
    - models_params: dict id->params (JSON)
    """
    def __init__(
        self,
        output_path: str,
        experiment_name: str,
        *,
        models: "List[str] | None" = None,
        models_params: "Dict[str, Dict[str, Any]] | None" = None,
        top_k: "int | None" = None,
        threshold: "float | None" = None,
        keep_original_features: "bool | None" = None,
        overwrite_cv: "bool | None" = None,
        outcome_label: "str | None" = None,
        outcome_type: "str | None" = None,
        instance_label: "str | None" = None,
        random_state: "int | None" = None,
        instance_subset: "int | None" = None,
        run_cluster: "str | bool" = False,
        queue: str = "defq",
        reserved_memory: int = 4,
    ):
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.run_cluster = run_cluster
        self.queue = queue
        self.reserved_memory = reserved_memory

        # defaults
        models = self._csv_to_list(models)
        meta = self._load_metadata()
        self.models = models or meta.get("P4 Models", ["mutualinformation","multisurf"])
        # if metadata also stores CSV, normalize that too
        if isinstance(self.models, str):
            self.models = self._csv_to_list(self.models)
        self.models_params = models_params or json.loads(meta.get("P4 Models Params", "{}") or "{}")
        self.top_k = top_k if top_k is not None else meta.get("P4 TopK", None)
        self.threshold = threshold if threshold is not None else meta.get("P4 Threshold", None)
        self.keep_original_features = bool(keep_original_features if keep_original_features is not None else meta.get("P4 Keep Original Features", False))
        self.overwrite_cv = bool(overwrite_cv if overwrite_cv is not None else True)
        self.outcome_label = outcome_label or meta.get("Outcome Label", "Class")
        self.outcome_type = outcome_type or meta.get("Outcome Type", None)
        self.instance_label = instance_label if instance_label is not None else meta.get("Instance Label", None)
        self.random_state = random_state if random_state is not None else meta.get("Random Seed", 0)
        self.instance_subset = instance_subset if instance_subset is not None else meta.get("P4 Instance Subset", None)

        exp_root = os.path.join(self.output_path, self.experiment_name)
        if not os.path.exists(exp_root):
            raise Exception("Experiment must exist before phase 4 can begin")

    def run(self):
        exp_root = os.path.join(self.output_path, self.experiment_name)
        jobs: List[Tuple[str,str,str]] = []  # (model_id, tr, te)
        logging.info(
            "Phase 4 starting for experiment '%s' with models: %s",
            self.experiment_name,
            ", ".join(self.models) if self.models else "(none)",
        )

        # discover CV pairs
        pairs: List[Tuple[str,str]] = []
        for name in os.listdir(exp_root):
            ds_dir = os.path.join(exp_root, name)
            if not os.path.isdir(ds_dir): continue
            if name in {"jobsCompleted","jobs","logs","dask_logs","DatasetComparisons"}: continue
            for tr in sorted(glob.glob(os.path.join(ds_dir, "CVDatasets/*Train.csv"))):
                te = tr.replace("Train.csv","Test.csv")
                if os.path.exists(te): pairs.append((tr, te))

        if not pairs: raise Exception("No CV Train/Test pairs found for Phase 4.")
        if not self.models: raise Exception("No feature-importance models specified.")
        logging.info("Phase 4 discovered %d CV train/test pairs.", len(pairs))

        # ensure model ids exist
        available = list_importances()
        for m in self.models:
            if m not in available:
                raise ValueError(f"Unknown model '{m}'. Available: {', '.join(sorted(available))}")

        # expand tasks: every (pair × model)
        for tr, te in pairs:
            for m in self.models:
                jobs.append((m, tr, te))

        mode = str(self.run_cluster) if self.run_cluster else "Serial"
        logging.info("Phase 4 submitting %d jobs in mode '%s'.", len(jobs), mode)
        if mode == "Local":
            with LocalCluster(processes=True, n_workers=num_cores, threads_per_worker=1) as cluster:
                with Client(cluster) as client:
                    tasks = [dask.delayed(self._run_one)(m, tr, te) for (m,tr,te) in jobs]
                    dask.compute(tasks, scheduler=client)
        elif self.run_cluster and self.run_cluster != "Serial" and self.run_cluster not in ("BashSLURM","BashLSF"):
            client: Client = get_cluster(self.run_cluster, exp_root, self.queue, self.reserved_memory)
            tasks = [dask.delayed(self._run_one)(m, tr, te) for (m,tr,te) in jobs]
            dask.compute(tasks, scheduler=client)
        elif self.run_cluster in ("BashSLURM","BashLSF"):
            for m,tr,te in jobs: self._submit_bash_job(m, tr, te)
        else:
            for m,tr,te in jobs: self._run_one(m, tr, te)

        self._save_run_params(mode)
        logging.info("Phase 4 completed: %d jobs finished.", len(jobs))

    def _run_one(self, model_id: str, tr: str, te: str):
        exp_root = os.path.join(self.output_path, self.experiment_name)
        train_path = Path(tr)
        dataset_name = train_path.parents[1].name if len(train_path.parents) > 1 else "unknown_dataset"
        cv_label = train_path.stem.replace("_Train", "")
        logging.info("Phase 4 running model '%s' on %s [%s].", model_id, dataset_name, cv_label)
        FeatureImportance(
            cv_train_path=tr,
            cv_test_path=te,
            experiment_path=exp_root,
            model_id=model_id,
            model_params=self.models_params.get(model_id, {}),
            top_k=self.top_k,
            threshold=self.threshold,
            keep_original_features=self.keep_original_features,
            overwrite_cv=self.overwrite_cv,
            outcome_label=self.outcome_label,
            outcome_type=self.outcome_type,
            instance_label=self.instance_label,
            random_state=self.random_state,
            instance_subset=self.instance_subset,
        ).run()
        logging.info("Phase 4 completed model '%s' on %s [%s].", model_id, dataset_name, cv_label)

    def _load_metadata(self):
        path = os.path.join(self.output_path, self.experiment_name, "metadata.pickle")
        if os.path.exists(path):
            with open(path,"rb") as f:
                try: return pickle.load(f) or {}
                except Exception: return {}
        return {}
    
    def _csv_to_list(self, v):
        if v is None: return None
        if isinstance(v, list): return v
        if isinstance(v, str):
            return [m.strip() for m in v.split(",") if m.strip()]
        return v

    def _save_run_params(self, mode: str):
        from datetime import datetime
        exp_root = os.path.join(self.output_path, self.experiment_name)
        params_file = os.path.join(exp_root, "run_params.pickle")
        this_run = {
            "phase": "p4_feature_importance",
            "run_mode": mode,
            "models": self.models,
            "models_params": self.models_params,
            "top_k": self.top_k,
            "threshold": self.threshold,
            "keep_original_features": self.keep_original_features,
            "overwrite_cv": self.overwrite_cv,
            "outcome_label": self.outcome_label,
            "outcome_type": self.outcome_type,
            "instance_label": self.instance_label,
            "random_state": self.random_state,
            "instance_subset": self.instance_subset,
        }
        all_params = {}
        if os.path.exists(params_file):
            with open(params_file,"rb") as f:
                try: all_params = pickle.load(f)
                except Exception: all_params = {}
        all_params[datetime.now().isoformat()] = this_run
        with open(params_file,"wb") as f: pickle.dump(all_params, f)

    # ---- bash submit: one job per (model × CV pair) ----
    def _submit_bash_job(self, model_id: str, tr: str, te: str):
        job_ref = str(time.time())
        run_dir = os.path.join(self.output_path, self.experiment_name)
        os.makedirs(os.path.join(run_dir,"jobs"), exist_ok=True)
        os.makedirs(os.path.join(run_dir,"logs"), exist_ok=True)
        job_name = os.path.join(run_dir, f"jobs/P4_{model_id}_{job_ref}_run.sh")
        launcher = "sbatch" if self.run_cluster == "BashSLURM" else "bsub <"

        with open(job_name, "w") as sh:
            sh.write("#!/bin/bash\n")
            if self.run_cluster == "BashSLURM":
                sh.write(f"#SBATCH -p {self.queue}\n")
                sh.write(f"#SBATCH --job-name={job_ref}\n")
                sh.write(f"#SBATCH --mem={self.reserved_memory}G\n")
                sh.write(f"#SBATCH -o {run_dir}/logs/P4_{model_id}_{job_ref}.o\n")
                sh.write(f"#SBATCH -e {run_dir}/logs/P4_{model_id}_{job_ref}.e\n")
                sh.write("srun " + self._bash_cmd(model_id, tr, te) + "\n")
            else:
                sh.write(f"#BSUB -q {self.queue}\n")
                sh.write(f"#BSUB -J {job_ref}\n")
                sh.write(f"#BSUB -R \"rusage[mem={self.reserved_memory}G]\"\n")
                sh.write(f"#BSUB -M {self.reserved_memory}GB\n")
                sh.write(f"#BSUB -o {run_dir}/logs/P4_{model_id}_{job_ref}.o\n")
                sh.write(f"#BSUB -e {run_dir}/logs/P4_{model_id}_{job_ref}.e\n")
                sh.write(self._bash_cmd(model_id, tr, te) + "\n")

        os.system(f"{launcher} {job_name}")
        logging.info("Phase 4 submitted cluster job script: %s", job_name)

    def _bash_cmd(self, model_id: str, tr: str, te: str) -> str:
        script_path = str(Path(__file__).parent / "p4_jobsubmit.py")
        exp_root = os.path.join(self.output_path, self.experiment_name)
        params = json.dumps(self.models_params.get(model_id, {}) or {})
        args = [
            "python", script_path,
            "--cv_train_path", tr,
            "--cv_test_path", te,
            "--experiment_path", exp_root,
            "--model_id", model_id,
            "--model_params", params,
            "--top_k", str(self.top_k) if self.top_k is not None else "",
            "--threshold", str(self.threshold) if self.threshold is not None else "",
            "--keep_original_features", str(int(self.keep_original_features)),
            "--overwrite_cv", str(int(self.overwrite_cv)),
            "--outcome_label", self.outcome_label or "",
            "--outcome_type", self.outcome_type or "",
            "--instance_label", self.instance_label or "",
            "--random_state", str(self.random_state) if self.random_state is not None else "",
            "--instance_subset", str(self.instance_subset) if self.instance_subset is not None else "",
        ]
        return " ".join(args)
