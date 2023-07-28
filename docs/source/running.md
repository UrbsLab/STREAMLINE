# Running STREAMLINE
This section details how to run STREAMLINE in any of its run modes. These include:
1. **Google Colab Notebook:** (run remotely on free google cloud resources)
2. **Jupyter Notebook:** (run locally on your PC)
3. **Command Line:** (locally or on a 'dask-compatable' CPU Computing Cluster)

While the notebooks only allow STREAMLINE to be run serially, it can be '[embarrassingly](https://en.wikipedia.org/wiki/Embarrassingly_parallel)' parallelized when run from the command line in one of two ways:
1. **Local command line:** basic CPU core parallelization 
2. **CPU Computing Cluster:** job submission parallelization

Lastly, when run from the command line, STREAMLINE can be run in one of two ways:
1. **All phases at once:** with a single command pointing to a 'configuration file' that includes all necessary run parameters
2. **One phase at a time:** using either phase-specific commands with command-line arguments, or again using the 'configuration file'

***
## Google Colab Notebook
This section on running STREAMLine
In this section we cover how to run STREAMLINE within [Google Colab](https://research.google.com/colaboratory/). If you're new to Google Colab you can also check out this [tutorial](https://www.tutorialspoint.com/google_colab/index.htm) on the basics.

Users with coding experience may wish to jump to sections covering how to run STREAMLINE [locally](local.md) (either using our provided Jupyter Notebook or from the command line) or on an [HPC Cluster](cluster.md) if you have access to one.

All users may benefit from reviewing the [guidelines](tips.md) section for tips on reducing runtime and improving modeling performance using STREAMLINE.

### Why run STREAMLINE on Google Colab?
Running STREAMLINE on Google Colab is best for:
1. Running the STREAMLINE demonstration on the included demo data
2. Users with little to no coding experience
3. Users that want the quickest/easiest approach to running STREAMLINE
4. Users that do not have access to a very powerful computer or compute cluster.
5. Applying STREAMLINE to smaller-scale analyses (in particular when only using free/limited Google Cloud resources):
    * Smaller datasets (e.g. < 500 instances and features)
    * A small number of total datasets (e.g. 1 or 2)
    * Only using the simplest/quickest modeling algorithms (e.g. Naive Bayes, Decision Trees, Logistic Regression)
    * Only using 1 or 2 modeling algorithms

### Running on Demo Dataset
Follow the steps below to get STREAMLINE running on the [demonstration datasets](sample.md#demonstration-data).
In summary, they detail the process of opening the STREAMLINE Colab Notebook to your Google Drive,
and running the notebook called `STREAMLINE-GoogleColab.ipynb` with Google Colaboratory (the link is provided below), running it
and downloading the output files.

1. Set up a Google account (if for some reason you don't already have one).
    * Click here for help: https://support.google.com/accounts/answer/27441?hl=en

2. Open the following Google Colab Notebook using this link:
[https://colab.research.google.com/drive/14AEfQ5hUPihm9JB2g730Fu3LiQ15Hhj2?usp=sharing](https://colab.research.google.com/drive/14AEfQ5hUPihm9JB2g730Fu3LiQ15Hhj2?usp=sharing)

3. [Optional] At the top of the notebook open the `Runtime` menu and select `Disconnect and delete runtime`. This clears the memory of the previous notebook run. This is only necessary when the underlying base code is modified, but it may be useful to troubleshoot if modifications to the notebook do not seem to have an effect.

4. At the top of the notebook open the `Runtime` menu and select `Run all`.  This directs the notebook to run all code cells of the notebook, i.e. all phases of STREAMLINE.  Here we have preconfigured STREAMLINE to automatically run on two [demonstration datasets](sample.md#demonstration-data) found in the `DemoData` folder.

5. Note: At this point the notebook will do the following automatically:
   1. Reserve a limited amount of free memory (RAM) and disk space on Google Cloud.
       * Note: it is also possible to set up this Notebook to run using the resources of your local PC (not covered here).
   2. Load the individual STREAMLINE run files into memory from the STREAMLINE Github.
   3. Install all other necessary python packages on the Google Colaboratory Environment.
   4. Run the entirety of STREAMLINE on the [demonstration datasets](sample.md#demonstration-data) folder (i.e. `DemoData`).
       * Note: all 5 steps should take approximately 3-5 minutes to run.
   5. Download the Main Report and the replication report automatically.
   6. If you have the last cell uncommented (by default) it will also download the complete experiment folder as zip file on your local computer.

### Inspecting your first run
During or after the notebook runs, users can inspect the individual code and text (i.e. markdown) ce
lls of the notebook. Individual cells can be collapsed or expanded by clicking on the small arrowhead
on the left side of each cell. The first set of cells set up the coding environment automatically.
Later cells are used to set the pipeline run parameters and then run the 11 phases of the pipeline in sequence.

Phase wise the cells will display output figures generated by STREAMLINE. Note that to save runtime,
this demonstration run is only applying three ML modeling algorithms: Naive Bayes, Logistic Regression,
and Decision Trees.  These are typically the three fastest algorithms available in STREAMLINE.

The Colab notebook automatically downloads the General and Replication Report,
in addition to the complete experiment folder.

The user open the zip on their local system, or alternatively you can explore the files on Google Colab
by opening the file-explorer pane on the left. How to analyze this output can be found in the [Analyzing Outputs](analysis.md) section.

### Running on your own datasets
This section explains how to update the Google Colaboratory Notebook to run on one or more user specified
datasets rather than the [demonstration datasets](sample.md#demonstration-data). This instructions are
effectively the same for running STREAMLINE from Jupyter Notebook. Note that, for brevity,
the parameter names given below are slightly different from the argument identifiers when using STREAMLINE
from the command-line (a guide for commandline parameters is given [here](parameters.md)).

1. Open the same notebook as the above section for DemoData
2. Follow the directions as written and set flag values `demo_run=False`
3. You can either use a data prompt with `use_data_prompt=True`, or set your own parameters manually, we recommend using `use_data_prompt=True`.
4. Upload files as and input `class_label`, `instance_label` and `match_label` when asked with these requirements:
    * Files are in comma-separated format with extension '.txt' or '.csv' format.
    * Missing data values should be empty or indicated with an 'NA'.
    * Dataset(s) include a header giving column labels.
    * Data columns should only include features (i.e. independant variables), a class label, and [optionally] instance (i.e. row) labels, and/or match labels (if matched cross validation will be used).
    * Binary class values are encoded as 0 (e.g. negative), and 1 (positive) with respect to true positive, true negative, false positive, false negative metrics. PRC plots focus on classification of 'positives'.
    * All feature values (both categorical and quantitative) are numerically encoded (i.e. no letters or words). Scikit-learn does not accept text-based values.
        * However, both `instance_label` and `match_label` values may be either numeric or text.
    * If multiple datasets are being analyzed they must each have the same `class_label` (e.g. 'Class'), and (if present), the same `instance_label` (e.g. 'ID') and `match_label` (e.g. 'Match_ID').
5. (Optional/Manual Mode) Update the first 6 pipeline run parameters as such:
    * `demo_run`: Change from True to False (Note, this parameter is only used by the notebooks for the demonstration analysis, and is one of the few parameters that use a Boolean rather than string value).
    * `data_path`: Change the end of the path from DemoData to the name of your new dataset folder (e.g. "/content/drive/MyDrive/STREAMLINE-main/my_data").
    * `output_path`: This can be left 'as-is' or modified to some other folder on your google drive within which to store all STREAMLINE experiments.
    * `experiment_name`: Change this to some new unique experiment name (do this each time you want to run a new experiment, either on the same or different dataset(s)), e.g. 'my_first_experiment'.
    * `class_label`: Change to the column header indicating the class label in each dataset, e.g. 'Class'.
    * `instance_label`: Change to the column header indicating unique instance ID's for each row in the dataset(s), or change to the string 'None' if your dataset does not include instance IDs.
6. (Optional/Manual Mode) Specifying replication data run parameters:
    * Scroll down to the code block with the text 'Run Parameters for Phase 10'.
    * If you don't have a replication dataset simply change `applyToReplication` to False (boolean value) and ignore the other two run parameters in this code block.
7. (Optional/Manual Mode) Update other STREAMLINE run parameters to suit your analysis needs within code blocks 6-14. We will cover some common run parameters to consider here:
    * `cv_partitions`: The number of CV training/testing partitions created, and consequently the number of models trained for each ML algorithm. We recommend setting this between 3-10. A larger value will take longer to run but produce more accurate results.
    * `categorical_cutoff`: STREAMLINE uses this parameter to automatically determine which features to treat as categorical vs. numeric. If a feature has more than this many unique values, it is considered to be numeric.
        * Note: Currently, STREAMLINE does NOT automatically apply one-hot-encoding to categorical features meaning that all features will still be treated as numerical during ML modeling. Its currently up to the users decide whether to pre-encode features.  However STREAMLINE does take feature type into account during both the exploratory analysis, data preprocessing, and feature importance phases.
        * Note: Users can also manually specify which features to treat as categorical or even to point to features in the dataset that should be ignored in the analysis with the parameters `ignore_features_path` and `categorical_feature_path`, respectively. For either, instead of the default string 'None' setting the user specifies the path to a .csv file including a row of feature names from the dataset that should either be treated as categorical or ignored, respectively.
    * `algorithms`: A list of modeling algorithms to run, setting it to None will run all the algorithms. Must be from the set of the full or abbreviated name of models found in `streamline/models` folder.
    * `exlude`: A list of modeling algorithms to exclude from the pipeline. Must be from the set of the full or abbreviated name of models found in `streamline/models` folder.
    * * `n_trials`: Set to a higher value to give Optuna more attempts to optimize hyperparameter settings.
    * `timeout`: Set higher to increase the maximum time allowed for Optuna to run the specified `n_trials` (useful for algorithms that take more time to run)
* Note: There are a number of other run parameter options, and we encourage users to read descriptions of each to see what other options are available.



## Running on Local System

This section describes the steps to run STREAMLINE Locally on your system.

If you haven't installed the STREAMLINE already goto the [installation](install.md#local-installation) section.
To run STREAMLINE locally make sure you've done the local installation as per the [guide](install.md)

As a gist run the following commands or be in the root STREAMLINE folder to run STREAMLINE.
```
git clone --single-branch https://github.com/UrbsLab/STREAMLINE
cd STREAMLINE
pip install -r requirements.txt
```

### Running on Jupyter Notebook

#### Running on Demo Dataset
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

#### Running on Your Own Datasets
Move your custom dataset to the STREAMLINE root directory,
follow the same steps as in the [Colab Notebook](colab.md#running-on-your-own-datasets-tbd)

### Running on command line interface

The most efficient way of running STREAMLINE is through command line.
There's two ways to run STREAMLINE thorough a CLI interface.

1. Through picking up run parameters through a config file.
2. Through manually inputting run parameters

There is a runner file called run.py which runs the whole or part of STREAMLINE
pipeline as defined. A few examples are given below.

#### Using config file

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
python run.py -c ./run_configs/local.cfg
```

Specifically the `run_cluster` parameter in the `multiprocessing` section has to been defined as
`run_cluster = "False"` which runs it locally. This specific parameter in 
the file is located [here](https://github.com/UrbsLab/STREAMLINE/blob/5c66b3286056bbd9b514c202aa0a22758a76f62c/run.cfg#L11)


#### Using command-line parameters

`run.py` can also be used with command line parameters 
as defined in the [parameters section](parameters.md)

Similarly, the following additional parameters need to be given

```
python run.py <other commands> --run-cluster False --run-parallel True<or Flase, accordingly>
```

As example case to all phases till report generation is given below:

```
python run.py --data-path ./data/DemoData --out-path demo --exp-name demo --do-till-report --class-label Class --inst-label InstanceID --algorithms=NB,LR,DT --run-cluster False --run-parallel True
```

A user can also run phases of STREAMLINE individually, 
however the user must have run all the phases before the phase he wants to run, i.e. the user must run this
pipeline sequentially in the given order.

To just run Exploratory Phase (Phase 1):
```
python run.py --data-path ./data/DemoData --out-path demo --exp-name demo --do-eda --class-label Class --inst-label InstanceID --algorithms NB,LR,DT --run-cluster False --run-parallel True
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
python run.py --rep-path ./data/DemoRepData --dataset ./data/DemoData/hcc-data_example_custom.csv --out-path demo --exp-name demo --do-replicate --run-cluster False --run-parallel True
```

To just run Replication Report Phase (Phase 10):
```
python run.py --rep-path ./data/DemoRepData --dataset ./data/DemoData/hcc-data_example_custom.csv --out-path demo --exp-name demo --do-rep-report --run-cluster False --run-parallel True
```

To just run Cleaning Phase (Phase 11):
```
python run.py --out-path demo --exp-name demo --do-clean --del-time --del-old-cv --run-cluster False --run-parallel True
```




## Running on HPC Clusters

The easiest way to run STREAMLINE on HPC is through the CLI interface.
The runtime parameters can easily be set up using either the config file 
of command line parameters. A few tools may be helpful in doings so and are described in
the [helpful tools](#helpful-tools) section.

You only need to additionally define 4 additional parameters to run the models
using a cluster setup.

Rest is handled similarly by `run.py` as defined in the local section.

### Helpful Tools

#### nano
GNU nano is a text editor for Unix-like computing 
systems or operating environments using a command line interface. 

This would be incredibly handy in changing opening and changing the config file through ssh terminal
for using STREAMLINE through config file.

A detailed guide can be found [here](https://www.hostinger.com/tutorials/how-to-install-and-use-nano-text-editor)

A gist of the application is that you can edit the `run_configs/cedars.cfg` config file by the following steps
1. Go to the root streamline folder.
2. Type `nano run_configs/cedars.cfg` in the terminal to open the file in tmux.
3. Make the necessary in the changes in the config file.
4. Press `Ctrl + X` to close the file and `Y` to save the changes.


#### tmux
tmux is a terminal multiplexer/emulator. It lets you switch easily between several programs in one terminal, 
detach them (they keep running in the background) and reattach them to a different terminal. 

Terminal emulators programs allow you to create several "pseudo terminals" from a single terminal.
They decouple your programs from the main terminal, 
protecting them from accidentally disconnecting. 
You can detach tmux or screen from the login terminal, 
and all your programs will continue to run safely in the background. 
Later, we can reattach them to the same or a different terminal to 
monitor the process. These are also very useful for running multiple programs with a single connection, 
such as when you're remotely connecting to a machine using Secure Shell (SSH).

A detailed guide on using it can be found [here](https://www.redhat.com/sysadmin/introduction-tmux-linux)

A gist of the application is that you can open a new terminal 
that will stay open even if you disconnect and close your terminal.

The steps to take it is as follows:
1. Go to the root streamline folder.
2. Type and run `tmux new -s mysession`
3. Open the required config file using nano (e.g. `run_configs/cedars.cfg`) 
4. Make the necessary in the changes in the config file.
5. Press `Ctrl + X` to close the file and `Y` to save the changes.
6. Run required commands.
7. Press `Ctrl + b` and then the `d` key to close the terminal.



### Running using command line interface

#### Using config file

Edit the multiprocessing section of the config file according to your needs.

The multiprocessing section has four parameters that need to be defined.
1. `run-parallel`: Flag to run parallel processing in local job, overridden if `run-cluster` is defined. 
2. `res-mem`: memory reserved per job
3. `run-cluster`: flag for type of cluster, by far the most important parameter discussed in detail below.
4. `queue`: the partition queue used for job submissions.

The `run_cluster` parameter is the most important parameter here.
It is set to False when running locally, to use a cluster implementation, specify as a 
string type of cluster. Currently, clusters supported by `dask-jobqueue` can be supported.

Additionally, the old method of manual submission can be done using the flags
`"SLURMOld"` and `"LSFOld"` instead. This will generate and submit jobs using shell files 
similar to the legacy version of STREAMLINE.

As example config setup to run all steps till report generations using SLURM dask-jobqueue on Cedars HPC Cluster Setup.
is given in the config 
file [here](https://github.com/UrbsLab/STREAMLINE/blob/main/run_configs/cedars.cfg)

We specifically focus on the multiprocessing section of the 
config file 
[here](https://github.com/UrbsLab/STREAMLINE/blob/main/run_configs/cedars.cfg#L8-L12).


Now you can run the pipeline using the following command (considering the config file is `upenn.cfg`): 
```
python run.py -c run_configs/cedars.cfg
```


#### Using command-line parameters

`run.py` can also be used with command line parameters 
as defined in the [parameters section](parameters.md)

As discussed above you need only specify 3 additional parameters in the 
CLI parameters way of running STREAMLINE

```
python run.py <other commands> --run-cluster SLURM --res-mem 4 --queue defq
```

We give examples to run all phases separately and together 
on the example DemoData on the Cedars SLURM HPC.

As example case to all phases till report generation is given below:

```
python run.py --data-path ./data/DemoData --out-path demo --exp-name demo --do-till-report --class-label Class --inst-label InstanceID --algorithms=NB,LR,DT --run-cluster SLURM --res-mem 4 --queue defq
```

A user can also run phases of STREAMLINE individually, 
however the user must have run all the phases before the phase he wants to run, i.e. the user must run this
pipeline sequentially in the given order.

To just run Exploratory Phase (Phase 1):
```
python run.py --data-path ./data/DemoData --out-path demo --exp-name demo --do-eda --class-label Class --inst-label InstanceID --run-cluster SLURM --res-mem 4 --queue defq
```

To just run Data Preparation Phase (Phase 2):
```
python run.py --out-path demo --exp-name demo --do-dataprep --run-cluster SLURM --res-mem 4 --queue defq
```


To just run Feature Importance Phase (Phase 3):
```
python run.py --out-path demo --exp-name demo --do-feat-imp --run-cluster SLURM --res-mem 4 --queue defq
```

To just run Feature Selection Phase (Phase 4):
```
python run.py --out-path demo --exp-name demo --do-feat-sel --run-cluster SLURM --res-mem 4 --queue defq
```

To just run Modeling Phase (Phase 5):
```
python run.py --out-path demo --exp-name demo --do-model --algorithms NB,LR,DT --run-cluster SLURM --res-mem 4 --queue defq
```

To just run Statistical Analysis Phase (Phase 6):
```
python run.py --out-path demo --exp-name demo --do-stats --run-cluster SLURM --res-mem 4 --queue defq
```

To just run Dataset Compare Phase (Phase 7):
```
python run.py --out-path demo --exp-name demo --do-compare-dataset --run-cluster SLURM --res-mem 4 --queue defq
```

To just run (Reporting Phase) Phase 8:
```
python run.py --out-path demo --exp-name demo --do-report --run-cluster SLURM --res-mem 4 --queue defq
```


To just run Replication Phase (Phase 9):
```
python run.py --rep-path ./data/DemoRepData --dataset ./data/DemoData/hcc-data_example_custom.csv --out-path demo --exp-name demo --do-replicate --run-cluster SLURM --res-mem 4 --queue defq
```

To just run Replication Report Phase (Phase 10):
```
python run.py --rep-path ./data/DemoRepData --dataset ./data/DemoData/hcc-data_example_custom.csv --out-path demo --exp-name demo --do-rep-report --run-cluster SLURM --res-mem 4 --queue defq
```

To just run Cleaning Phase (Phase 11):
```
python run.py --out-path demo --exp-name demo --do-clean --del-time --del-old-cv --run-cluster SLURM --res-mem 4 --queue defq
```


