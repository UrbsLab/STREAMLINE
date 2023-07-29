# Installation
Installation instructions for different run modes of STREAMLINE.

***
## Google Colab Notebook 
No installation is required to run STREAMLINE in the included Google Colab Notebook. The only other step is to make sure that you have a Google account (free) and click the link below:

[https://colab.research.google.com/drive/14AEfQ5hUPihm9JB2g730Fu3LiQ15Hhj2?usp=sharing](https://colab.research.google.com/drive/14AEfQ5hUPihm9JB2g730Fu3LiQ15Hhj2?usp=sharing)

***
## Local Installation
The instructions below are for installing STREAMINE locally in order to run it either in the included Jupyter Noteook or from the command line. 

### Prerequisites
First, be sure to install or confirm previous installation of the following prerequisites.

#### Git
Install git (if not already installed). 

* You can test for an existing installation by typing `git` in your command-line.

* Git installation instructions can be found [here](https://github.com/git-guides/install-git).

#### Anaconda
We recommend installing the most recent stable version of Anaconda3 appropriate for your operating system (if not already installed) which automatically includes Python3 and a number of other common packages used by STREAMLINE (e.g. pandas, scikit-learn, etc.). Python3 is the most essential prerequisite here. Additional required Python packages will automatically be installed by the installation commands (below). 

* You can test for an existing installation by typing `conda` on your command-line.
* Anaconda installation instructions can be found [here](https://docs.anaconda.com/anaconda/install/index.html)
* If you already have Anaconda installed, we recommend updating conda and anaconda (as follows) prior to running STREAMLINE to avoid downstream module/environment errors
```
conda update conda
conda update anaconda
```

While STREAMLINE can run on native Python3, this is not generally recomended, since issue resolution becomes complex, especially in MacOS based systems. 

### Installation Commands
After confirming that you have the prerequisites above, navigate to the directory where you want to save STREAMLINE, and use the following commands in the command-line terminal:

```
git clone --single-branch https://github.com/UrbsLab/STREAMLINE
cd STREAMLINE
pip install -r requirements.txt
```

The above 3 commands do the following:
1. Download the most recent release repository of STREAMLINE
2. Navigate to the root STREAMLINE directory from where the package can run
3. Install all other packages required to run STREAMLINE on the local system (see `requirements.txt` for the complete list of these packages)

Now the STREAMLINE package can be run from the STREAMLINE root directory.

#### Troubleshooting
If you see errors or warnings when running the above commands, this may indicate that the required packages might not be installing properly, which may prevent STREAMLINE from running to completion. We recommend always testing the STREAMLINE installation first by [running](running.md) it (in the desired [run mode](running.md#picking-a-run-mode)) on the [demonstration](sample.md#demonstration-data) data with included/example default run parameters. Issues related to installation will be evident if you get 'module' related errors when running STREAMLINE. This is most likely to happen if you are working from a previous installation of Anaconda, and you should update both `conda` and `anaconda` first and then retry the commands above. 

### Jupyter Notebook
If you with to run STREAMLINE using the included Jupyter Notebook, additionally do the following:

1. Make sure the jupyter package is installed using the following command:
   ```
   pip install jupyter
   ```
2. Run Jupyter Notebook using the command `jupyter notebook`
3. Within the web page that opens, navigate into the saved STREAMLINE folder and open the `STREAMLINE-Notebook.ipynb` file.

For more information on Jupyter Notebook, click [here](https://jupyter.org/).

***
## Cluster Installation
STREAMLINE installation for a CPU computing cluster (i.e. HPC) is essentially the same as for local installation, but may include extra steps or troubleshooting based on your HPC setup. As for local installation, generally within your cluster home/working directory, you'll want to install Git, Anaconda (with Python3), and use the installation commands to download STREAMLINE and other required Python packages.

### Cluster Compatability
We have set up STREAMLINE to be able to able to run on 7 different types of HPC clusters using `dask_jobqueue` including [LSF, SLURM, PBS, OAR, Moab, SGE, HTCondor] as documented [here](https://jobqueue.dask.org/en/latest/api.html). To date we have explicitly tested STREAMLINE only on LSF and SLURM clusters.

### Additional Tools
Here we recommend additional tools that may help in running big jobs across all STREAMLINE phases, from a single command. These tool include terminal emulators like `tmux` and `screen` and terminal text editors like `nano` and `vim`. In most likelihood these would already be installed in your cluster or available as modules in your cluster.

#### Terminal Emulators
A terminal emulator is particularly important when you want to run all phases of STREAMLINE automatically from a single command. To achieve this, STREAMLINE runs a script on the head node (i.e. job submission node) that monitors phase completion and submits new jobs for the next phase. Typically, closing your terminal would interupt this process. 

Terminal emulator programs allow you to create several "pseudo terminals" from a single terminal. They decouple your programs from the main terminal, protecting them from accidentally disconnecting. You can detach `tmux` or `screen` from the login terminal, and all your programs will continue to run safely in the background. Later, we can reattach them to the same or a different terminal to monitor the process. 

These are also very useful for running multiple programs with a single connection, such as when you're remotely connecting to a machine using Secure Shell (SSH).

We recommend using `tmux` as a terminal emulator. A quick guide on using it can be found [here](https://www.redhat.com/sysadmin/introduction-tmux-linux).

#### Terminal Text Editors
Terminal text editors are simple text editor programs that allow you to edit files through the terminal. These are particularly useful here for quickly editing the configuration file that specifies all STREAMLINE run parameters for running all phases from a single command. 

We recommend using `nano` as a text editor. A quick guide on using it can be found [here](https://www.hostinger.com/tutorials/how-to-install-and-use-nano-text-editor)

***
## Known Installation Issues
1. Scipy version error, the way the STREAMLINE is set up it needs scipy>=1.8.0. If you find this is not true or
   get a version error output, please run `pip install --upgrade scipy`
2. The lightgbm package on pypi doesn't work out of the box using pip on MacOS. The following command should solve the problem:
   ```conda install -c conda-forge lightgbm```
3. The most recent skrebate package (0.7) does not install correctly with the standard `pip install skrebate` command, however it does with `pip install skrebate==0.7`.