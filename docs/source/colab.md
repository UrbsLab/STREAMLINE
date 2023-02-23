# Running on Google Colab

In this section we first provide [instructions for users with little to no coding experience](#use-mode-1-google-colaboratory). Users with some coding experience can jump to the [standard instructions](#use-modes-2-4-standard-installation-and-use). However, all users would benefit from reviewing the following sections:

* [Inspecting your first run](#inspecting-your-first-run)
* [Running STREAMLINE on your own dataset(s)](#running-streamline-on-your-own-datasets)
* [Tips for reducing STREAMLINE runtime](#tips-for-reducing-streamline-runtime)
* [Tips for improving STREAMLINE modeling performance](#tips-for-improving-streamline-modeling-performance)

As STREAMLINE is currently in its 'beta' release, we recommend users first check that they have downloaded the most recent release of STREAMLINE before applying any of the run modes described below, as we are actively updating this software as feedback is received.

***
## Google Colaboratory
This is the easiest but most limited way to run STREAMLINE. These instructions are geared towards those with little to no computing experience. All other users can skip to the next [section](#use-modes-2-4-standard-installation-and-use) but may wish to revisit later parts of this section for helpful details.
* To learn more about Google Colaboratory prior to setup please visit the following link: https://research.google.com/colaboratory/

### Setting up your first run
Follow the steps below to get STREAMLINE running on the [demonstration datasets](#demonstration-data). 
In summary, they detail the process of opening the STREAMLINE Colab Notebook to your Google Drive, 
and running the notebook called `STREAMLINE-GoogleColabNotebook.ipynb` with Google Colaboratory, running it 
and downloading the output files.

1. Set up a Google account (if for some reason you don't already have one).
    * Click here for help: https://support.google.com/accounts/answer/27441?hl=en

2. Open the following Google Colab Notebook using this link: https://colab.research.google.com/drive/17s55GajtN5WCEV-DfegtiFhvGp73Haj4?usp=sharing

3. [Optional] At the top of the notebook open the `Runtime` menu and select `Disconnect and delete runtime`. This clears the memory of the previous notebook run. This is only necessary when the underlying base code is modified, but it may be useful to troubleshoot if modifications to the notebook do not seem to have an effect.

4. At the top of the notebook open the `Runtime` menu and select `Run all`.  This directs the notebook to run all code cells of the notebook, i.e. all phases of STREAMLINE.  Here we have preconfigured STREAMLINE to automatically run on two [demonstration datasets](#demonstration-data) found in the `DemoData` folder.

5. Note: At this point the notebook will do the following automatically:
   1. Reserve a limited amount of free memory (RAM) and disk space on Google Cloud.
       * Note: it is also possible to set up this Notebook to run using the resources of your local PC (not covered here).
   2. Load the individual STREAMLINE run files into memory from the STREAMLINE Github.
   3. Install all other necessary python packages on the Google Colaboratory Environment.
   4. Run the entirety of STREAMLINE on the [demonstration datasets](#demonstration-data) folder (i.e. `DemoData`).
       * Note: all 5 steps should take approximately 3-5 minutes to run.
   5. Download the Main Report and the replication report automatically.
   6. If you have the last cell uncommented it will also download the complete experiment folder as zip file on your local computer.
