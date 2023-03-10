# Running on Local System

This section describes the steps to run STREAMLINE Locally on your system.

If you haven't installed the STREAMLINE already goto the [installation](install.md#local-installation) section.
To run STREAMLINE locally make sure you've done the local installation as per the [guide](install.md)

As a gist run the following commands or be in the root STREAMLINE folder to run STREAMLINE.
```
git clone -b dev --single-branch https://github.com/UrbsLab/STREAMLINE
cd STREAMLINE
pip install -r requirements.txt
```

## Running on Jupyter Notebook

### Running on Demo Dataset
Here we detail how to run STREAMLINE within the provided Jupyter Notebook named `STREAMLINE-Notebook.ipypnb`. 
This included notebook is set up to run on the included [demonstration datasets](sample.md#demonstration-data).

1. First, ensure all local installation is done as per the [guide](install.md#jupyter-notebook) in your environment 
   and dataset assumptions are satisfied.

2. Open Jupyter Notebook (info about Jupyter Notebooks here, https://jupyter.readthedocs.io/en/latest/running.html) by 
   going to the STREAMLINE Root folder, typing `jupyter notebook` and then opening the `STREAMLINE-Notebook.ipypnb` that
   shows up in the new page open on your web browser.

3. Scroll down to the second code block of the notebook below the header 'Mandatory Parameters to Update' and update the following run parameters to reflect paths on your PC.
    * `data_path`: Change the path, so it reflects the location of the `DemoData` folder (within the STREAMLINE folder) on your PC, e.g. `C:/Users/ryanu/Documents/GitHub/STREAMLINE/DemoData`.
    * `output_path`: Change the path to specify the desired location for the STREAMLINE output experiment folder.

4. Click `Kernel` on the Jupyter notebook GUI, and select `Restart & Run All` to run the script.  

5. Running the included [demonstration datasets](sample.md#demonstration-data) with the pre-specified notebook run parameters, 
   should only take a 3-5 minutes depending on your PC hardware.
    * Note: It may take several hours or more to run this notebook in other contexts. Parameters that impact runtimes are discussed in [this section](tips.md#reducing-runtime) above. We recommend users with larger analyses to use a computing cluster if possible.

### Running on Your Own Datasets
Move your custom dataset to the STREAMLINE root directory,
follow the same steps as in the [Colab Notebook](colab.md#running-on-your-own-datasets-tbd)

## Running on command line interface

The most efficient way of running STREAMLINE is through command line.
There's two ways to run STREAMLINE thorough a CLI interface.

1. Through picking up run parameters through a config file.
2. Through manually inputting run parameters

There is a runner file called run.py which runs the whole or part of STREAMLINE
pipeline as defined. A few examples are given below.

### Using config file

`run.py` can also be used with config parameters 
as defined in the [parameters section](parameters.md)

Then it can be run with the command defined below.
```
python run.py -c <config_file>
```

For running the Demo Dataset locally a config file is already provided and 
the user doesn't need to do any edits.

The example config setup to run all steps till report generations locally on the Demo Dataset
is given in the config 
file [here](https://github.com/UrbsLab/STREAMLINE/blob/dev/run.cfg)

The user can simply run the following command to run the whole pipeline:
```
python run.py -c local.cfg
```

Specifically the `run_cluster` parameter in the `multiprocessing` section has to been defined as
`run_cluster = "False"` which runs it locally. This specific parameter in 
the file is located [here](https://github.com/UrbsLab/STREAMLINE/blob/5c66b3286056bbd9b514c202aa0a22758a76f62c/run.cfg#L11)


### Using command-line parameters

`run.py` can also be used with command line parameters 
as defined in the [parameters section](parameters.md)

Similarly, the following additional parameters need to be given

```
python run.py <other commands> --run-cluster False --run-parallel True<or Flase, accordingly>
```

As example case to all phases till report generation is given below:

```
python run.py --data-path DemoData --out-path demo --exp-name demo --do-till-report --class-label Class --inst-label InstanceID --algorithms=NB,LR,DT --run-cluster False --run-parallel True
```

A user can also run phases of STREAMLINE individually, 
however the user must have run all the phases before the phase he wants to run, i.e. the user must run this
pipeline sequentially in the given order.

To just run Exploratory Phase (Phase 1):
```
python run.py --data-path DemoData --out-path demo --exp-name demo --do-eda --class-label Class --inst-label InstanceID --run-cluster False --run-parallel True
```

To just run Data Preparation Phase (Phase 2):
```
python run.py --out-path demo --exp-name demo --do-dataprep --run-cluster False --run-parallel True
```


To just run Feature Importance Phase (Phase 3):
```
python run.py --out-path demo --exp-name demo --do-feat-imp --run-cluster False --run-parallel True
```

To just run Feature Selection Phase (Phase 4):
```
python run.py --out-path demo --exp-name demo --do-feat-sel --run-cluster False --run-parallel True
```

To just run Modeling Phase (Phase 5):
```
python run.py --out-path demo --exp-name demo --do-model --algorithms NB,LR,DT --run-cluster False --run-parallel True
```

To just run Statistical Analysis Phase (Phase 6):
```
python run.py --out-path demo --exp-name demo --do-stats --run-cluster False --run-parallel True
```

To just run Dataset Compare Phase (Phase 7):
```
python run.py --out-path demo --exp-name demo --class-label Class --inst-label InstanceID --do-till-report False --do-compare-dataset True --algorithms NB,LR,DT --do-all False --run-cluster False --run-parallel True
```

To just run (Reporting Phase) Phase 8:
```
python run.py --out-path demo --exp-name demo --class-label Class --inst-label InstanceID --do-till-report False --do-report True --algorithms NB,LR,DT --do-all False --run-cluster False --run-parallel True
```


To just run Replication Phase (Phase 9):
```
python run.py --rep-path DemoRepData --dataset DemoData/demodata.csv --out-path demo --exp-name demo --do-replicate --run-cluster False --run-parallel True
```

To just run Replication Report Phase (Phase 10):
```
python run.py --rep-path DemoRepData --dataset DemoData/demodata.csv --out-path demo --exp-name demo --do-rep-report --run-cluster False --run-parallel True
```

To just run Cleaning Phase (Phase 11):
```
python run.py --out-path demo --exp-name demo --do-clean --del-time --del-old-cv --run-cluster False --run-parallel True
```