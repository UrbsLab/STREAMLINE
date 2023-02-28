# Running on HPC Clusters

The easiest way to run STREAMLINE on HPC is through the CLI interface.
The runtime parameters can easily be set up using either the config file 
of command line parameters.

You only need to additionally define 4 additional parameters to run the models
using a cluster setup.

Rest is handled similarly by `run.py` as defined in [local section](local.md#running-on-cli)

### Using config file

Edit the multiprocessing section of the config file according to your needs.

The multiprocessing section has four parameters that need to be defined.
1. `run_parallel`: Flag to run parallel processing in local job, overridden if `run_cluster` is defined. 
2. `reserved_memory`: memory reserved per job
3. `run_cluster`: flag for type of cluster, by far the most important parameter discussed in detail below.
4. `queue`: the partition queue used for job submissions.

The `run_cluster` parameter is the most important parameter here.
It is set to False when running locally, to use a cluster implementation, specify as a 
string type of cluster. Currently clusters supported by `dask-jobqueue` can be supported.

Additionally, the old method of manual submission can be done using the flags
`"SLURMOld"` and `"LSFOld"` instead. This will generate and submit jobs using shell files 
similar to the legacy version of STREAMLINE.

As example config setup to run all steps till report generations
is given in the config 
file [here](https://github.com/UrbsLab/STREAMLINE/blob/dev/run.cfg)

We specifically focus on the multiprocessing section of the config file.

```
[multiprocessing]
run_parallel = False
reserved_memory = 4
run_cluster = "SLURMOld"
queue = 'defq'
```

Now you can run the pipeline using the following command (considering the config file is `config.cfg`): 
```
run.py -c config.cfg
```


### Using command-line parameters

`run.py` can also be used with command line parameters 
as defined in the [parameters section](parameters.md)

As discussed above you need only specify 3 additional parameters in the 
CLI parameters way of running STREAMLINE

```
python run.py <other commands> --run_cluster SLURMOld --reserved_memory 4 --queue defq
```


As example case to run just the EDA phase and 
all phases till report generation is given below.


```
example code, to be written
```