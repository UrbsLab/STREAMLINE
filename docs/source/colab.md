# Running on Google Colab

In this section we first provide [instructions for users with little to no coding experience](#use-mode-1-google-colaboratory). Users with some coding experience can jump to the [standard instructions](#use-modes-2-4-standard-installation-and-use). However, all users would benefit from reviewing the following sections:

* [Inspecting your first run](#inspecting-your-first-run)
* [Running STREAMLINE on your own dataset(s)](#running-streamline-on-your-own-datasets)
* [Tips for reducing STREAMLINE runtime](#tips-for-reducing-streamline-runtime)
* [Tips for improving STREAMLINE modeling performance](#tips-for-improving-streamline-modeling-performance)
* [Code orientation](#code-orientation)

As STREAMLINE is currently in its 'beta' release, we recommend users first check that they have downloaded the most recent release of STREAMLINE before applying any of the run modes described below, as we are actively updating this software as feedback is received.

***
## Google Colaboratory
This is the easiest but most limited way to run STREAMLINE. These instructions are geared towards those with little to no computing experience. All other users can skip to the next [section](#use-modes-2-4-standard-installation-and-use) but may wish to revisit later parts of this section for helpful details.
* To learn more about Google Colaboratory prior to setup please visit the following link: https://research.google.com/colaboratory/

### Setting up your first run
Follow the steps below to get STREAMLINE running on the [demonstration datasets](#demonstration-data). 
In summary, they detail the process of copying the STREAMLINE GitHub repository to your Google Drive, 
and running the notebook called `STREAMLINE-GoogleColabNotebook.ipynb` with Google Colaboratory.

1. Set up a Google account (if for some reason you don't already have one).
    * Click here for help: https://support.google.com/accounts/answer/27441?hl=en

2. Make sure you can access your Google Drive.
    * Click here to open Google Drive with your google account: https://drive.google.com

3. Navigate to this GitHub Repository: https://github.com/UrbsLab/STREAMLINE

4. Click the green button labeled `Code` and select `Download ZIP`.

5. Unzip this file:
    * Navigate to the folder where this file was downloaded.
    * Find the file named `STREAMLINE-main.zip`.
    * Right click on the zipped file and choose `Extract all`, then select `Extract`.
    * Note: This will typically create an unzipped folder in your `Downloads` folder with the name `STREAMLINE-main`, with another folder inside it also called `STREAMLINE-main`. This inner folder is the one you will be copying in the next steps, so that when you open it, you immediately see all the STREAMLINE files.

6. Ensure that you have located your extracted folder named `STREAMLINE-main`, and that when you open it, you immediately see the various STREAMLINE files and folders.

7. Navigate to `My Drive` in your Google Drive.  This is the base folder in your google drive account.

8. Copy the inner extracted folder named `STREAMLINE-main` to `My Drive` on your Google Drive account.

9. Open the newly copied `STREAMLINE-main` folder on Google Drive.

10. Open the `Colab_Output` folder and confirm there is no subfolder named `hcc_demo`. If there is, right clicking on it and select `Remove`.
    * Note: STREAMLINE creates a folder here using the name set by the `experiment_name` parameter. You will need to remove this folder anytime you want to re-run the demo of STREAMLINE without changing the experiment folder name. This prevents users from accidentally overwriting a previous run of the pipeline unintentionally. As an alternative, users can simply change the name of the `experiment_name` parameter within the Notebook.

11. Navigate back to the base `STREAMLINE-main` folder on Google Drive.

12. Ensure you have installed the Google Colaboratory App.
    * Right click on `STREAMLINE-GoogleColabNotebook.ipynb` (this is the notebook used to run the pipeline on Google Colaboratory only)
    * Choose `Open with` and select:
        1. `Google Colaboratory` if it's already installed, or
        2. `Connect more apps`, then search for and install `Google Colaboratory`
    * Note: Once Google Colaboratory has been installed you need only double click on the notebook file to open it in the future.
    * The STREAMLINE notebook will now open in Google Colaboratory as a webpage.

13. [Optional] At the top of the notebook open the `Runtime` menu and select `Disconnect and delete runtime`. This clears the memory of the previous notebook run. This is only necessary when the underlying base code is modified, but it may be useful to troubleshoot if modifications to the notebook do not seem to have an effect.

14. At the top of the notebook open the `Runtime` menu and select `Run all`.  This directs the notebook to run all code cells of the notebook, i.e. all phases of STREAMLINE.  Here we have preconfigured STREAMLINE to automatically run on two [demonstration datasets](#demonstration-data) found in the `DemoData` folder.

15. In order to communicate with your Google Drive, Google will ask permission for the notebook to connect to it.
    * First pop up window: Click `Connect to Google Drive`
    * Second pop up window: Choose the Google account within which you copied the `STREAMLINE-main` folder from the available list.
    * Third pop up window: Scroll down and select `Allow`.

16. Note: At this point the notebook will do the following automatically:
    1. Reserve a limited amount of free memory (RAM) and disk space on Google Cloud.
        * Note: it is also possible to set up this Notebook to run using the resources of your local PC (not covered here).
    2. Mount your google drive (so it can access the STREAMLINE run files and export output files to it).
    3. Load the individual STREAMLINE run files into memory from STREAMLINE-main/streamline/.
    4. Install all other necessary python packages not already available in Anaconda3 (which is preloaded in the Google Colaboratory Environment).
    5. Run the entirety of STREAMLINE on the [demonstration datasets](#demonstration-data) folder (i.e. `DemoData`).
        * Note: all 5 steps should take approximately 3-5 minutes to run.
    6. Save all output files to `My Drive/STREAMLINE-main/Colab_Output/hcc_demo`