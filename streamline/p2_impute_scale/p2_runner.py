# streamline/phases/p2_impute_scale/runner.py
import logging
import os
import re
import glob
import json
import time
import pickle
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List

import pandas as pd

import dask
from dask.distributed import Client, LocalCluster
logger = logging.getLogger("distributed.worker")
logger.setLevel(logging.WARNING)

from streamline.utils.runners import num_cores  # runner_fn not needed; we call job.run()
from streamline.utils.cluster import get_cluster  # must return a connected Dask Client
from streamline.p2_impute_scale.impute_scale import ImputeAndScale
from streamline.p2_impute_scale.utils.impute_loader import list_imputers



class P2Runner:
    """
    Phase 2 runner (CV-based, Dask-aware, bash-job submission capable).

    Modes (set via run_cluster):
      • "Local"        → local Dask parallelization.
      • "BashSLURM"    → submit a bash script (sbatch) that runs p2_jobsubmit.py per CV pair.
      • "BashLSF"      → submit a bash script (bsub) that runs p2_jobsubmit.py per CV pair.
      • any other str  → modern Dask cluster name; get_cluster(name, ...) returns a connected Client (works in Jupyter).
      • False/None     → Serial.
    """

    def __init__(
        self,
        output_path: str,
        experiment_name: str,

        # Phase-2 flags (if None, we pull from metadata.pickle)
        scale_data: "bool | None" = None,
        impute_data: "bool | None" = None,
        multi_impute: "bool | None" = None,
        overwrite_cv: "bool | None" = None,
        outcome_label: "str | None" = None,
        instance_label: "str | None" = None,
        random_state: "int | None" = None,

        # optional registry-driven imputer
        imputer_id: "str | None" = None,
        imputer_params: "Dict[str, Any] | None" = None,
        scaler_id: "str | None" = None,
        scaler_params: "Dict[str, Any] | None" = None,

        # execution mode
        run_cluster: "str | bool" = False,   # False | "Local" | "BashSLURM" | "BashLSF" | "<dask-cluster-name>"
        queue: str = 'defq',
        reserved_memory: int = 4,
    ):
        self.output_path = output_path
        self.experiment_name = experiment_name

        # read metadata defaults; overrides apply if user provided explicit values
        meta = self._load_metadata()

        self.scale_data = self._coalesce_bool(scale_data, meta.get('Use Data Scaling', True))
        self.impute_data = self._coalesce_bool(impute_data, meta.get('Use Data Imputation', True))
        self.multi_impute = self._coalesce_bool(multi_impute, meta.get('Use Multivariate Imputation', False))
        self.overwrite_cv = self._coalesce_bool(overwrite_cv, True)

        self.outcome_label = outcome_label or meta.get('Outcome Label', 'Class')
        self.instance_label = instance_label if (instance_label is not None) else meta.get('Instance Label', None)
        self.random_state = random_state if (random_state is not None) else meta.get('Random Seed', 0)

        # phase-2 imputer choices (metadata may contain saved prior selection)
        self.imputer_id = imputer_id or meta.get('P2 Imputer Id', None)
        mp = meta.get('P2 Imputer Params', '{}')
        if isinstance(mp, str):
            try:
                mp = json.loads(mp or "{}")
            except Exception:
                mp = {}
        self.imputer_params = imputer_params or mp or {}
        self.scaler_id = (scaler_id if 'scaler_id' in locals() else None) or meta.get('P2 Scaler Id', None)
        sp = meta.get('P2 Scaler Params', '{}')
        if isinstance(sp, str):
            try: sp = json.loads(sp or "{}")
            except Exception: sp = {}
        self.scaler_params = (scaler_params or {}) or sp


        # execution
        self.run_cluster = run_cluster
        self.queue = queue
        self.reserved_memory = reserved_memory

        # sanity checks
        exp_root = os.path.join(self.output_path, self.experiment_name)
        if not os.path.exists(exp_root):
            raise Exception("Experiment must exist (from phase 1) before phase 2 can begin")

    # ----------------------------
    # Main
    # ----------------------------
    def run(self):
        exp_root = os.path.join(self.output_path, self.experiment_name)

        # discover datasets (folders directly under experiment root)
        dataset_dirs = []
        for name in os.listdir(exp_root):
            ds_dir = os.path.join(exp_root, name)
            if not os.path.isdir(ds_dir):
                continue
            if name in {'jobsCompleted', 'jobs', 'logs', 'dask_logs', 'DatasetComparisons'}:
                continue
            dataset_dirs.append(ds_dir)

        # build all CV jobs (train/test pairs)
        jobs: List[Tuple[str, str]] = []
        for ds_dir in dataset_dirs:
            os.makedirs(os.path.join(ds_dir, 'impute_scale'), exist_ok=True)
            ds_name = os.path.basename(ds_dir.rstrip('/'))
            cv_dir = os.path.join(ds_dir, "CVDatasets")
            if not os.path.isdir(cv_dir):
                logging.warning(f"Skipping {ds_name}: no CVDatasets folder")
                continue
            for tr in sorted(glob.glob(os.path.join(cv_dir, f"*Train.csv"))):
                te = tr.replace("Train.csv", "Test.csv")
                if os.path.exists(te):
                    jobs.append((tr, te))

        if not jobs:
            raise Exception("No CV Train/Test CSV pairs found under experiment CVDatasets folders.")

        # ---- EXECUTION STRATEGY ----
        run_mode = str(self.run_cluster) if self.run_cluster else "Serial"

        if run_mode == "Local":
            # Local Dask parallelization
            n_workers = num_cores
            with LocalCluster(processes=True, n_workers=n_workers, threads_per_worker=1) as cluster:
                with Client(cluster) as client:
                    tasks = [
                        dask.delayed(self._run_one_pair)(tr, te) for (tr, te) in jobs
                    ]
                    dask.compute(tasks, scheduler=client)

        elif self.run_cluster and self.run_cluster not in ("BashSLURM", "BashLSF"):
            # Modern Dask cluster (works in Jupyter)
            client: Client = get_cluster(
                self.run_cluster,
                exp_root,
                self.queue,
                self.reserved_memory
            )
            tasks = [
                dask.delayed(self._run_one_pair)(tr, te) for (tr, te) in jobs
            ]
            dask.compute(tasks, scheduler=client)

        elif self.run_cluster in ("BashSLURM", "BashLSF"):
            # Bash scripts that call p2_jobsubmit.py per CV pair
            for (tr, te) in jobs:
                self._submit_bash_job(tr, te)
        else:
            # Serial
            for (tr, te) in jobs:
                self._run_one_pair(tr, te)

        self.save_run_params(run_mode=run_mode)

    # ----------------------------
    # Helpers
    # ----------------------------
    def _run_one_pair(self, cv_train_path: str, cv_test_path: str):
        exp_root = os.path.join(self.output_path, self.experiment_name)
        job = ImputeAndScale(
            cv_train_path=cv_train_path,
            cv_test_path=cv_test_path,
            experiment_path=exp_root,
            scale_data=self.scale_data,
            impute_data=self.impute_data,
            multi_impute=self.multi_impute,
            overwrite_cv=self.overwrite_cv,
            outcome_label=self.outcome_label,
            instance_label=self.instance_label,
            random_state=self.random_state,
            imputer_id=self.imputer_id,
            imputer_params=self.imputer_params,
            scaler_id=self.scaler_id,
            scaler_params=self.scaler_params,
        )
        job.run()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata.pickle from the experiment root if present."""
        meta_path = os.path.join(self.output_path, self.experiment_name, "metadata.pickle")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                try:
                    return pickle.load(f) or {}
                except Exception:
                    return {}
        return {}

    @staticmethod
    def _coalesce_bool(v, default):
        return default if (v is None) else bool(v)

    def save_run_params(self, run_mode: str):
        """Append this run's parameters into a single pickle dict keyed by ISO timestamp."""
        from datetime import datetime
        exp_root = os.path.join(self.output_path, self.experiment_name)
        os.makedirs(exp_root, exist_ok=True)
        params_file = os.path.join(exp_root, "run_params.pickle")

        this_run = {
            "phase": "p2_impute_scale",
            "run_mode": run_mode,
            "output_path": self.output_path,
            "experiment_name": self.experiment_name,
            "scale_data": self.scale_data,
            "impute_data": self.impute_data,
            "multi_impute": self.multi_impute,
            "overwrite_cv": self.overwrite_cv,
            "outcome_label": self.outcome_label,
            "instance_label": self.instance_label,
            "random_state": self.random_state,
            "imputer_id": self.imputer_id,
            "imputer_params": self.imputer_params,
            "scaler_id": self.scaler_id,
            "scaler_params": self.scaler_params,
            "queue": self.queue,
            "reserved_memory": self.reserved_memory,
        }

        if os.path.exists(params_file):
            with open(params_file, "rb") as f:
                try:
                    all_params = pickle.load(f)
                except Exception:
                    all_params = {}
        else:
            all_params = {}

        ts = datetime.now().isoformat()
        all_params[ts] = this_run

        with open(params_file, "wb") as f:
            pickle.dump(all_params, f)

        logging.info(f"Updated run parameters in {params_file}")

    # ----------------------------
    # Bash submission (uses p2_jobsubmit.py)
    # ----------------------------
    def _submit_bash_job(self, cv_train_path: str, cv_test_path: str):
        job_ref = str(time.time())
        run_dir = os.path.join(self.output_path, self.experiment_name)
        os.makedirs(os.path.join(run_dir, 'jobs'), exist_ok=True)
        os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)
        job_name = os.path.join(run_dir, f'jobs/P2_{job_ref}_run.sh')

        if self.run_cluster == "BashSLURM":
            launcher = 'sbatch'
        elif self.run_cluster == "BashLSF":
            launcher = 'bsub <'
        else:
            raise Exception("Bash submission of HPC type unsupported")

        with open(job_name, 'w') as sh:
            sh.write('#!/bin/bash\n')
            if self.run_cluster == "BashSLURM":
                sh.write('#SBATCH -p ' + self.queue + '\n')
                sh.write('#SBATCH --job-name=' + job_ref + '\n')
                sh.write('#SBATCH --mem=' + str(self.reserved_memory) + 'G' + '\n')
                sh.write('#SBATCH -o ' + run_dir + f'/logs/P2_{job_ref}.o\n')
                sh.write('#SBATCH -e ' + run_dir + f'/logs/P2_{job_ref}.e\n')
                cmd = self._bash_submit_command(cv_train_path, cv_test_path)
                sh.write('srun ' + cmd + '\n')
            else:
                sh.write('#BSUB -q ' + self.queue + '\n')
                sh.write('#BSUB -J ' + job_ref + '\n')
                sh.write('#BSUB -R "rusage[mem=' + str(self.reserved_memory) + 'G]"' + '\n')
                sh.write('#BSUB -M ' + str(self.reserved_memory) + 'GB' + '\n')
                sh.write('#BSUB -o ' + run_dir + f'/logs/P2_{job_ref}.o\n')
                sh.write('#BSUB -e ' + run_dir + f'/logs/P2_{job_ref}.e\n')
                cmd = self._bash_submit_command(cv_train_path, cv_test_path)
                sh.write(cmd + '\n')

        os.system(f'{launcher} {job_name}')

    def _bash_submit_command(self, cv_train_path: str, cv_test_path: str) -> str:
        """
        Build command to run a single-CV-pair job via p2_jobsubmit.py (bash path).
        p2_jobsubmit.py must parse args and run ImputeAndScale once.
        Unspecified flags are resolved inside p2_jobsubmit.py by reading metadata.pickle.
        """
        script_path = str(Path(__file__).parent / "p2_jobsubmit.py")
        exp_root = os.path.join(self.output_path, self.experiment_name)
        args = [
            'python', script_path,
            '--cv_train_path', cv_train_path,
            '--cv_test_path', cv_test_path,
            '--experiment_path', exp_root,

            # Optional overrides; these can be empty and p2_jobsubmit.py will fill from metadata
            '--scale_data', str(int(self.scale_data)) if self.scale_data is not None else '',
            '--impute_data', str(int(self.impute_data)) if self.impute_data is not None else '',
            '--multi_impute', str(int(self.multi_impute)) if self.multi_impute is not None else '',
            '--overwrite_cv', str(int(self.overwrite_cv)) if self.overwrite_cv is not None else '',
            '--outcome_label', self.outcome_label or '',
            '--instance_label', self.instance_label or '',
            '--random_state', str(self.random_state) if self.random_state is not None else '',
            '--imputer_id', self.imputer_id or '',
            '--imputer_params', json.dumps(self.imputer_params or {}),
            '--scaler_id', self.scaler_id or '',
            '--scaler_params', json.dumps(self.scaler_params or {}),
        ]
        return ' '.join(args)
    
