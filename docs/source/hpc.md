# HPC and Cluster Runs

STREAMLINE can run small examples on a laptop, but paper-scale runs often need a
cluster. The cluster path is still the same pipeline: edit a `.cfg`, dry-run it,
then launch the config runner. The difference is that selected phases submit
many scheduler jobs through SLURM or LSF and the config runner waits for those
jobs to finish before moving to the next phase.

## When To Use Each Execution Mode

| Mode | Best use |
| --- | --- |
| `Serial` | Debugging, small demos, and first config checks. |
| `Parallel` | A single machine or one allocated compute node using joblib multiprocessing. |
| `Local` | A local Dask cluster on one machine. |
| `BashSLURM` | HPC systems that submit jobs with `sbatch`. |
| `BashLSF` | HPC systems that submit jobs with `bsub`. |
| Named Dask cluster | Site-specific Dask jobqueue execution when configured by the user/site. |

Use a scheduler mode for long P4/P6/P8/P10/P11-style workloads or any analysis
that would be inappropriate to run directly on a login node. Use `Parallel` only
inside an interactive allocation or on a machine where it is acceptable to use
multiple local cores.

## Included HPC Config Templates

HPC configs live in `run_configs/hpc/`.

| Config | Scheduler | Intended starting point |
| --- | --- | --- |
| `run_configs/hpc/cedars_slurm_hcc.cfg` | SLURM | Cedars/Sinai-style SLURM clusters using `run_cluster = BashSLURM`. |
| `run_configs/hpc/upenn_lsf_hcc.cfg` | LSF | UPenn/I2C2-style LSF clusters using `run_cluster = BashLSF`. |

Both templates run the HCC binary demo by default. Copy one of them before using
it for a real project and edit at least `output_path`, `experiment_name`,
`data_path`, `queue`, `reserved_memory`, model list, and modeling budget.

## Basic Cluster Setup

From a login node:

```bash
ssh <user>@<cluster-host>
git clone --single-branch https://github.com/UrbsLab/STREAMLINE.git
cd STREAMLINE
conda create -n streamline python=3.11 pip
conda activate streamline
pip install -r requirements.txt
python run.py --help
```

Many clusters require modules before Conda, Python, or compiled libraries are
available. If your site uses modules, load the same modules before installation
and before running STREAMLINE jobs. Also make sure the repository, data, and
`output_path` are on a filesystem visible to compute nodes.

## Conda Installation Quickstart

If Conda is already available on the cluster, either directly or through a
module, create a dedicated STREAMLINE environment from the repository root:

```bash
module load anaconda  # omit or change this if your cluster uses a different module name
conda create -n streamline python=3.11 pip
conda activate streamline
pip install -r requirements.txt
```

If Conda is not available, install Miniconda in your home or project space using
your cluster's approved download method:

```bash
mkdir -p ~/miniconda3
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n streamline python=3.11 pip
conda activate streamline
pip install -r requirements.txt
```

Some HPC systems block outbound internet from compute nodes. In that case,
install packages from the login node, a site Conda mirror, or an administrator
provided module/wheelhouse, then run STREAMLINE from the same environment.

## Use tmux For Long Runs

The config runner is the phase orchestrator. Scheduler jobs can keep running if
your SSH connection drops, but the runner may stop waiting and the next phases
may not launch. Use `tmux` or `screen` for long runs.

```bash
tmux new -s streamline
conda activate streamline
python run.py -c run_configs/hpc/cedars_slurm_hcc.cfg --dry_run
python run.py -c run_configs/hpc/cedars_slurm_hcc.cfg
```

Useful `tmux` commands:

```bash
# Detach from the session without stopping STREAMLINE:
Ctrl-b, then d

# List sessions:
tmux ls

# Reattach later:
tmux attach -t streamline

# Kill the session after the run is done:
tmux kill-session -t streamline
```

The same pattern works for the UPenn LSF template:

```bash
tmux new -s streamline
conda activate streamline
python run.py -c run_configs/hpc/upenn_lsf_hcc.cfg --dry_run
python run.py -c run_configs/hpc/upenn_lsf_hcc.cfg
```

## Scheduler Settings In Configs

The core cluster settings live in the `[run]` section:

```ini
run_cluster = BashSLURM
wait_for_cluster_completion = True
cluster_phase_timeout = 86400
cluster_phase_poll_interval = 30
queue = defq
reserved_memory = 4
```

For UPenn/LSF, the same fields look like:

```ini
run_cluster = BashLSF
queue = i2c2_normal
reserved_memory = 4
```

`queue` maps to the scheduler queue or partition. `reserved_memory` is the memory
request in GB used when STREAMLINE writes scheduler scripts. The exact queue
names and memory limits are site-specific, so treat the included values as
starting points.

`wait_for_cluster_completion = True` tells the config runner to wait for
STREAMLINE completion markers in `jobsCompleted/` before it starts the next
phase. This is important because later phases depend on files written by earlier
scheduler jobs.

## Monitoring Jobs

STREAMLINE writes scheduler scripts to the experiment `jobs/` folder and
stdout/stderr files to `logs/`.

Common SLURM commands:

```bash
squeue -u $USER
sacct -j <job_id>
scancel <job_id>
```

Common LSF commands:

```bash
bjobs
bjobs -l <job_id>
bkill <job_id>
```

If a phase appears stuck, check the scheduler first, then inspect
`<output_path>/<experiment_name>/logs/` and the `jobsCompleted/` markers.

## Recovery And Reruns

Use a dry run before every large launch:

```bash
python run.py -c run_configs/hpc/cedars_slurm_hcc.cfg --dry_run
```

If one phase fails, restart from that phase instead of repeating the full run:

```bash
python run.py -c run_configs/hpc/cedars_slurm_hcc.cfg --start_at p6
python run.py -c run_configs/hpc/cedars_slurm_hcc.cfg --only p8,p11
```

Phase 6 reruns and overwrites requested model jobs by default. For recovery,
set `skip_completed_models = True` in `[p6]` or pass
`--skip_completed_models 1` to the P6 CLI. That runs missing or failed model/CV
jobs while leaving completed model jobs in place.

If the config runner times out while scheduler jobs are still queued or running,
increase `cluster_phase_timeout` and rerun from the interrupted phase after
checking the logs.

## Practical HPC Checklist

Before a paper-scale cluster run:

* Confirm the config with `--dry_run`.
* Use absolute paths for project data and outputs when running outside the repo.
* Keep `output_path` on shared storage visible to login and compute nodes.
* Start from small `models`, `n_trials`, `timeout`, and `n_splits` values.
* Use `tmux` or `screen` for any run that may outlive an SSH session.
* Confirm the Conda environment is available on compute nodes.
* Check `logs/` and `jobsCompleted/` before restarting a failed phase.
* Use `skip_completed_models = True` only for Phase 6 recovery runs where you do
  not want to overwrite completed model artifacts.
