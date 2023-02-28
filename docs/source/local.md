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
    * `data_path`: Change the path so it reflects the location of the `DemoData` folder (within the STREAMLINE folder) on your PC, e.g. `C:/Users/ryanu/Documents/GitHub/STREAMLINE/DemoData`.
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

Then it can be run with the command defined below (considering 
the config file is `run.cfg`.
```
python run.py -c run.cfg
```

As example config setup to run all steps till report generations
is given in the config 
file [here](https://github.com/UrbsLab/STREAMLINE/blob/dev/run.cfg)

except specifically the `multiprocessing` section needs to be defined as
```
[multiprocessing]
run_parallel = True <or False, accordingly>
reserved_memory = 4
run_cluster = False
queue = 'defq'
```

This sets the cluster setting off and makes it run 
natively on the local machine.



### Using command-line parameters

`run.py` can also be used with command line parameters 
as defined in the [parameters section](parameters.md)

Similarly the following additional parameters need to given

```
python run.py <other commands> --run-cluster False --run-parallel True<or Flase, accordingly>
```

As example case to all phases till report generation is given below:

```
python run.py --data-path DemoData --out-path demo --exp-name demo \
               --class-label Class --inst-label InstanceID \
               --run-cluster False --run-parallel True
```

To just run EDA Phase:
```
python run.py --data-path DemoData --out-path demo --exp-name demo \
               --class-label Class --inst-label InstanceID \
               --do-till-report False --do-eda True \
               --run-cluster False --run-parallel True
```
