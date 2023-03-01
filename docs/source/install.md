# Installation

## Google Colaboratory
There is no local installation or additional steps required to run 
STREAMLINE on Google Colab.
Just have a Google Account and open this Colab Link:
[https://colab.research.google.com/drive/17s55GajtN5WCEV-DfegtiFhvGp73Haj4?usp=sharing](https://colab.research.google.com/drive/17s55GajtN5WCEV-DfegtiFhvGp73Haj4?usp=sharing)

## Local Installation
To install the STREAMLINE locally:
(bleeding edge development/to be updated on merge to main)


If you don't have git installed, install git
(you can test by typing `git` on your command-line):\
Reference for installation: [https://github.com/git-guides/install-git](https://github.com/git-guides/install-git)

While it is fine to run STREAMLINE on python, we recommend using Anaconda3 \
(you can test by typing `conda` on your command-line) \
Reference for installation: [https://docs.anaconda.com/anaconda/install/index.html](https://docs.anaconda.com/anaconda/install/index.html)

STREAMLINE can run on native python, but it is not recommend as resolution for issues becomes complex, especially in MacOS based systems.

After confirming that you have the above
use the following commands on the command-line terminal:
```
git clone -b dev --single-branch https://github.com/UrbsLab/STREAMLINE
cd STREAMLINE
pip install -r requirements.txt
```

These 3 commands specifically:
1. Download the most recent release repository of STREAMLINE
2. Go to the root STREAMLINE folder from where the package can run
3. Install all packages required to run STREAMLINE on the local system

Now the complete STREAMLINE package can be run 
from the STREAMLINE root directory.

(In future iterations we plan to develop streamline 
as a pip package and may not need to be run from the specific root directory)

## Jupyter Notebook
The Jupyter notebook usage is the same as Local installation with two additional steps as follows:

1. Make sure the jupyter package is installed using the following command:
   ```pip install jupyter```
2. Run jupyter-notebook using the command `jupyter notebook` and open the `STREAMLINE-Notebook.ipynb` in the 
   page that opens up in your web browser.

## Cluster Installation
Cluster installation may carry extra steps as per your HPC setup, but is essentially the same as the Local Setup.

Additional tools that may help in running big jobs include terminal emulators like `tmux` and `screen`
and terminal text editors like `nano` and `vim`.

Terminal emulators programs allow you to create several "pseudo terminals" from a single terminal.
They decouple your programs from the main terminal, 
protecting them from accidentally disconnecting. 
You can detach tmux or screen from the login terminal, 
and all your programs will continue to run safely in the background. 
Later, we can reattach them to the same or a different terminal to 
monitor the process. These are also very useful for running multiple programs with a single connection, 
such as when you're remotely connecting to a machine using Secure Shell (SSH).

Terminal text editors are simple text editor programs that allow you to edit files through the terminal.

In most likelihood these would be installed in your cluster or available as modules in your cluster.

We recommend using `tmux` as a terminal emulator, 
a quick guide on using it can be found [here](https://www.redhat.com/sysadmin/introduction-tmux-linux)

We recommend using `nano` as a text editor, 
a quick guide on using it can be found [here](https://www.hostinger.com/tutorials/how-to-install-and-use-nano-text-editor)


## Known Issues

1. Scipy version error, the way the STREAMLINE is set up it needs scipy>=1.8.0. If you find this is not true or 
   get a version error output, please run `pip install --upgrade scipy`
2. The lightgbm package on pypi doesn't work out of the box using pip on MacOS. The following command should solve the problem:
   ```conda install -c conda-forge lightgbm```
