# Installation

STREAMLINE can be run from Google Colab, a local notebook, or the command
line. Local command-line and notebook use should be done from the repository
root so Python can import the `streamline` package.

## Google Colab

No local installation is required for Colab. Open the current notebook and run
the setup cells:

[Open the STREAMLINE Colab notebook](https://colab.research.google.com/drive/1ByQuU805GzDGAAGzbUYz8wahnOTUuzvg?usp=sharing)

The notebook clones the repository with `--depth 1`, installs requirements,
and exposes a parameter block for binary, multiclass, regression, and custom
dataset runs.

## Local Conda Environment

The recommended local setup is a dedicated conda environment:

```bash
git clone --single-branch https://github.com/UrbsLab/STREAMLINE.git
cd STREAMLINE
conda create -n streamline python=3.11 pip
conda activate streamline
pip install -r requirements.txt
```

Then confirm that the config runner is available:

```bash
python run.py --help
```

## Local venv Environment

A standard Python virtual environment also works:

```bash
git clone --single-branch https://github.com/UrbsLab/STREAMLINE.git
cd STREAMLINE
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Building The Documentation

Install the docs-only packages and run Sphinx:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/build/html
```

The generated site opens from:

```text
docs/build/html/index.html
```

## Local Jupyter

Install Jupyter if it is not already available:

```bash
pip install jupyter
jupyter notebook
```

Open `STREAMLINE_Notebook.ipynb` from the repository root. The notebook keeps a
parameter block at the top so the same notebook can run binary, multiclass,
regression, or custom datasets.

## Cluster Notes

Cluster use is environment-specific. The phase CLIs and config runner accept
`run_cluster` settings such as `Serial`, `Local`, `Parallel`, `BashSLURM`, and `BashLSF`
depending on phase support. For long runs, use a persistent terminal session
such as `tmux` or `screen` so orchestration is not interrupted if your SSH
connection drops.

## Known Dependency Notes

Some modeling packages include compiled dependencies. If `lightgbm`,
`catboost`, `xgboost`, or rule-learning packages fail to install through pip
on your platform, install the compatible package through conda-forge or use the
provided conda environment as the baseline.
