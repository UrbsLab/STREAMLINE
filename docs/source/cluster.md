# Running on HPC Clusters

The easiest way to run STREAMLINE on HPC is through the CLI interface.
The runtime parameters can easily be set up using either the config file 
of command line parameters. A few tools may be helpful in doings so and are described in
the [helpful tools](#helpful-tools) section.

You only need to additionally define 4 additional parameters to run the models
using a cluster setup.

Rest is handled similarly by `run.py` as defined in the local section.

## Helpful Tools

### nano
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


### tmux
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



## Running using command line interface

### Using config file

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


### Using command-line parameters

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
python run.py --data-path ./data/DemoData --out-path demo --exp-name demo --do-eda --class-label Class --inst-label InstanceID --run-cluster SLURM --res-mem 4 --queue defq
```

A user can also run phases of STREAMLINE individually, 
however the user must have run all the phases before the phase he wants to run, i.e. the user must run this
pipeline sequentially in the given order.

To just run Exploratory Phase (Phase 1):
```
python run.py --out-path demo --exp-name demo --do-dataprep --run-cluster SLURM --res-mem 4 --queue defq
```

To just run Data Preparation Phase (Phase 2):
```
python run.py --out-path demo --exp-name demo --do-feat-imp --run-cluster SLURM --res-mem 4 --queue defq
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
python run.py --out-path demo --exp-name demo --do-clean --del-time --del-old-cv --run-cluster SLURM
```
