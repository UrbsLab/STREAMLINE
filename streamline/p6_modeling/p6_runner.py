from __future__ import annotations
import os, time
from pathlib import Path
from typing import Optional, List
import logging

import dask
from dask.distributed import Client, LocalCluster
logger = logging.getLogger("distributed.worker"); logger.setLevel(logging.WARNING)

from streamline.p6_modeling.modeling import ModelingPhaseJob
from streamline.p6_modeling.utils.categorical import NATIVE_CATEGORICAL_MODELS_DEFAULT
from streamline.p6_modeling.utils.loader import modeling_type_to_outcome_type, normalize_modeling_type
from streamline.utils.runners import num_cores
from streamline.utils.cluster import get_cluster  # must return a connected Dask Client


class P6Runner:
    """
    Phase 6 runner (modeling).
    Modes via run_cluster:
      • "Serial"
      • "Local"
      • "BashSLURM" | "BashLSF"
      • "<dask-cluster-name>" (get_cluster(...) provides a connected Client)
    """
    def __init__(
        self,
        output_path: str,
        experiment_name: str,
        *,
        outcome_label: str = "Class",
        outcome_type: Optional[str] = None,  # "Binary" | "Multiclass" | "Continuous"
        model_type: Optional[str] = None,    # Backward-compatible alias; use outcome_type.
        instance_label: Optional[str] = None,
        n_splits: int = 10,
        models: List[str] | str | None = None,        # CSV/list; None = auto-discover defaults
        model_params_json: Optional[str] = None,


        # calibration (now handled inside BaseModel.fit)
        calibrate: bool = False,
        calibrate_method: str = "sigmoid",
        calibrate_cv: int = 5,

        # ModelJob controls
        scoring_metric: str = "balanced_accuracy",
        metric_direction: str = "maximize",
        n_trials: int = 200,
        timeout: int = 900,
        training_subsample: int = 0,
        uniform_fi: bool = False,
        save_plot: bool = False,
        random_state: Optional[int] = None,
        bypass_one_hot_for_native_models: bool = True,
        native_categorical_models: str | List[str] | None = NATIVE_CATEGORICAL_MODELS_DEFAULT,

        # execution
        run_cluster: str = "Serial",   # "Serial" | "Local" | "BashSLURM" | "BashLSF" | "<cluster>"
        queue: str = "defq",
        reserved_memory: int = 4,
    ):
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.outcome_label = outcome_label
        self.model_type = normalize_modeling_type(outcome_type=outcome_type, model_type=model_type)
        self.outcome_type = outcome_type or modeling_type_to_outcome_type(self.model_type)
        self.instance_label = instance_label
        self.n_splits = int(n_splits)
        self.models = models
        self.model_params_json = model_params_json


        self.calibrate = bool(calibrate)
        self.calibrate_method = calibrate_method
        self.calibrate_cv = int(calibrate_cv)

        self.scoring_metric = scoring_metric
        self.metric_direction = metric_direction
        self.n_trials = int(n_trials)
        self.timeout = int(timeout)
        self.training_subsample = int(training_subsample)
        self.uniform_fi = bool(uniform_fi)
        self.save_plot = bool(save_plot)
        self.random_state = random_state
        self.bypass_one_hot_for_native_models = bool(bypass_one_hot_for_native_models)
        self.native_categorical_models = native_categorical_models

        self.run_cluster = run_cluster or "Serial"
        self.queue = queue
        self.reserved_memory = int(reserved_memory)

        self.exp_root = os.path.join(self.output_path, self.experiment_name)
        if not os.path.isdir(self.exp_root):
            raise Exception("Experiment must exist before Phase 6 can begin")

        if self.run_cluster in ("BashSLURM", "BashLSF"):
            os.makedirs(os.path.join(self.exp_root, "jobs"), exist_ok=True)
            os.makedirs(os.path.join(self.exp_root, "logs"), exist_ok=True)

    def run(self):
        datasets = [
            os.path.join(self.exp_root, name)
            for name in sorted(os.listdir(self.exp_root))
            if os.path.isdir(os.path.join(self.exp_root, name))
            and name not in {"jobsCompleted","jobs","logs","dask_logs","DatasetComparisons"}
            and os.path.isdir(os.path.join(self.exp_root, name, "CVDatasets"))
        ]
        if not datasets:
            logging.warning("No datasets found for Phase 6 under %s", self.exp_root)
            return

        mode = self.run_cluster
        if mode == "Serial":
            for ds in datasets: self._run_one(ds)
        elif mode == "Local":
            with LocalCluster(processes=True, n_workers=num_cores, threads_per_worker=1) as cluster:
                with Client(cluster) as client:
                    tasks = [dask.delayed(self._run_one)(ds) for ds in datasets]
                    dask.compute(tasks, scheduler=client)
        elif mode in ("BashSLURM", "BashLSF"):
            for ds in datasets: self._submit_bash(ds, mode)
        else:
            client: Client = get_cluster(mode, self.exp_root, self.queue, self.reserved_memory)
            tasks = [dask.delayed(self._run_one)(ds) for ds in datasets]
            dask.compute(tasks, scheduler=client)

    def _run_one(self, dataset_dir: str):
        ModelingPhaseJob(
            dataset_dir=dataset_dir,
            outcome_label=self.outcome_label,
            model_type=self.model_type,
            instance_label=self.instance_label,
            n_splits=self.n_splits,
            models=self.models,
            model_params_json=self.model_params_json,


            # pass through to BaseModel via Job → Model construction
            calibrate=self.calibrate,
            calibrate_method=self.calibrate_method,
            calibrate_cv=self.calibrate_cv,

            output_path=self.output_path,
            experiment_name=self.experiment_name,
            scoring_metric=self.scoring_metric,
            metric_direction=self.metric_direction,
            n_trials=self.n_trials,
            timeout=self.timeout,
            training_subsample=self.training_subsample,
            uniform_fi=self.uniform_fi,
            save_plot=self.save_plot,
            random_state=self.random_state,
            bypass_one_hot_for_native_models=self.bypass_one_hot_for_native_models,
            native_categorical_models=self.native_categorical_models,
        ).run()

    def _submit_bash(self, dataset_dir: str, mode: str):
        job_ref = str(time.time())
        jobs = os.path.join(self.exp_root, "jobs")
        logs = os.path.join(self.exp_root, "logs")
        os.makedirs(jobs, exist_ok=True); os.makedirs(logs, exist_ok=True)
        sh_path = os.path.join(jobs, f"P6_{job_ref}_run.sh")
        launcher = "sbatch" if mode == "BashSLURM" else "bsub <"

        script = str(Path(__file__).parent / "p6_jobsubmit.py")
        args = [
            "python", script,
            "--dataset_dir", dataset_dir,
            "--outcome_label", self.outcome_label,
            "--outcome_type", self.outcome_type,
            "--instance_label", self.instance_label or "",
            "--n_splits", str(self.n_splits),
            "--models", (",".join(self.models) if isinstance(self.models, list) else (self.models or "")),

            "--calibrate", "1" if self.calibrate else "0",
            "--calibrate_method", self.calibrate_method,
            "--calibrate_cv", str(self.calibrate_cv),

            "--output_path", self.output_path,
            "--experiment_name", self.experiment_name,
            "--scoring_metric", self.scoring_metric,
            "--metric_direction", self.metric_direction,
            "--n_trials", str(self.n_trials),
            "--timeout", str(self.timeout),
            "--training_subsample", str(self.training_subsample),
            "--uniform_fi", "1" if self.uniform_fi else "0",
            "--save_plot", "1" if self.save_plot else "0",
            "--random_state", str(self.random_state) if self.random_state is not None else "",
            "--bypass_one_hot_for_native_models", "1" if self.bypass_one_hot_for_native_models else "0",
            "--native_categorical_models", (
                ",".join(self.native_categorical_models)
                if isinstance(self.native_categorical_models, list)
                else (self.native_categorical_models or "")
            ),
        ]
        if self.model_params_json:
            json_arg = self.model_params_json.replace('"', '\\"')
            args.extend(["--model_params_json", f'"{json_arg}"'])
        cmd = " ".join(args)

        with open(sh_path, "w") as sh:
            sh.write("#!/bin/bash\n")
            if mode == "BashSLURM":
                sh.write(f"#SBATCH -p {self.queue}\n")
                sh.write(f"#SBATCH --job-name={job_ref}\n")
                sh.write(f"#SBATCH --mem={self.reserved_memory}G\n")
                sh.write(f"#SBATCH -o {logs}/P6_{job_ref}.o\n")
                sh.write(f"#SBATCH -e {logs}/P6_{job_ref}.e\n")
                sh.write("srun " + cmd + "\n")
            else:
                sh.write(f"#BSUB -q {self.queue}\n")
                sh.write(f"#BSUB -J {job_ref}\n")
                sh.write(f"#BSUB -R \"rusage[mem={self.reserved_memory}G]\"\n")
                sh.write(f"#BSUB -M {self.reserved_memory}GB\n")
                sh.write(f"#BSUB -o {logs}/P6_{job_ref}.o\n")
                sh.write(f"#BSUB -e {logs}/P6_{job_ref}.e\n")
                sh.write(cmd + "\n")
        os.system(f"{launcher} {sh_path}")
