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

STREAMLINE supports Python 3.10 and newer. Python 3.11 is the recommended
default for local demos because it works well across the current scientific
Python stack while remaining close to common notebook runtimes.

```bash
git clone --single-branch https://github.com/UrbsLab/STREAMLINE.git
cd STREAMLINE
conda create -n streamline python=3.11 pip
conda activate streamline
conda install pytorch=2.6 -y
pip install -r requirements.txt
```

Then confirm that the config runner is available:

```bash
python run.py --help
```

TabPFN is optional and is not required for the default STREAMLINE demos. Install
it separately with `pip install tabpfn` before running TabPFN models. TabPFN
also requires a Prior Labs token before model weights can be downloaded; see
[TabPFN Token Setup](tabpfn_token.md).

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

## Known Installation Issues

Some modeling and reporting packages include compiled dependencies. On macOS,
especially on Apple Silicon or fresh Conda environments, install the compiled
pieces through conda-forge first if pip reports build, linker, Graphviz, or
WeasyPrint errors:

```bash
conda install -c conda-forge lightgbm xgboost catboost -y
conda install -c conda-forge graphviz python-graphviz -y
conda install -c conda-forge weasyprint cairocffi cairo pango gdk-pixbuf libffi -y
pip install -r requirements.txt
```

If the failure is isolated to one package, install only that package from
conda-forge and rerun `pip install -r requirements.txt`.
