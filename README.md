![alttext](https://github.com/UrbsLab/STREAMLINE/blob/main/Pictures/STREAMLINE_LOGO.jpg?raw=true)
# Overview
STREAMLINE is an end-to-end automated machine learning (AutoML) pipeline that empowers anyone to easily run, interpret, and apply a rigorous and customizable analysis for data mining or predictive modeling. Notably, this tool is currently limited to supervised learning on tabular, binary classification data but will be expanded as our development continues. The development of this pipeline focused on (1) overall automation, (2) avoiding and detecting sources of bias, (3) optimizing modeling performance, (4) ensuring reproducibility, (5) capturing complex associations in data (e.g. feature interactions), and (6) enhancing interpretability of output. Overall, the goal of this pipeline is to provide a transparent framework to learn from data as well as identify the strengths and weaknesses of ML modeling algorithms or other AutoML algorithms.

We have recently submitted a publication introducing and applying STREAMLINE. See [below](#citation) for code citation information prior to publication.

***
## Quick Start
Click below for the easiest way for anyone to run an analysis using STREAMLINE (on Google Colaboratory):

[Setting Up Your First Run](#setting-up-your-first-run)

Click below for a quick look at an example pre-run notebook (with no installation steps), and/or example output files.

[View pipeline and output before running](#view-pipeline-and-output-before-running)

***
## STREAMLINE schematic
This schematic breaks the overall pipeline down into 4 basic components: (1) preprocessing and feature transformation, (2) feature importance evaluation and selection, (3) modeling, and (4) postprocessing.

![alttext](https://github.com/UrbsLab/STREAMLINE/blob/main/Pictures/ML_pipe_schematic.png?raw=true)

***
## Table of Contents
* [Overview](#overview)
    * [Quick start](#quick-start)
    * [STREAMLINE schematic](#streamline-schematic)    
    * [Table of contents](#table-of-contents)
    * [What level of computing skill is required for use?](#what-level-of-computing-skill-is-required-for-use)
    * [What can it be used for?](#what-can-it-be-used-for)
    * [What does STREAMLINE include?](#what-does-streamline-include)
    * [How is STREAMLINE different from other AutoML tools?](#how-is-streamline-different-from-other-automl-tools)
    * [STREAMLINE run modes](#streamline-run-modes)
    * [View pipeline and output before running](#view-pipeline-and-output-before-running)
    * [Implementation](#implementation)
    * [Disclaimer](#disclaimer)
* [Installation and Use](#installation-and-use)
    * [Use Mode 1: Google Colaboratory](#use-mode-1-google-colaboratory)
        * [Setting up your first run](#setting-up-your-first-run)
        * [Inspecting your first run](#inspecting-your-first-run)
        * [Running STREAMLINE on your own dataset(s)](#running-streamline-on-your-own-datasets)
        * [Tips for reducing STREAMLINE runtime](#tips-for-reducing-streamline-runtime)
        * [Tips for improving STREAMLINE modeling performance](#tips-for-improving-streamline-modeling-performance)
    * [Use Modes 2-4: Standard Installation and Use](#use-modes-2-4-standard-installation-and-use)
        * [Prerequisites](#prerequisites)
            * [Anaconda3](#anaconda3)
            * [Additional Python Packages](#additional-python-packages)
        * [Download STREAMLINE](#download-streamline)
        * [Code orientation](#code-orientation)
        * [Run from jupyter notebook](#run-from-jupyter-notebook)
            * [Running STREAMLINE jupyter notebook on your own dataset(s)](#running-streamline-jupyter-notebook-on-your-own-datasets)
        * [Run from command line (local or cluster parallelization)](#run-from-command-line-local-or-cluster-parallelization)
            * [Local run example](#local-run-example)
            * [Computing cluster run (parallelized) example](#computing-cluster-run-parallelized-example)
        * [Checking phase completion](#checking-phase-completion)
        * [Phase details (run parameters and additional examples)](#phase-details-run-parameters-and-additional-examples)
            * [Phase 1: Exploratory Analysis](#phase-1-exploratory-analysis)
                * [Example: Data with instances matched by one or more covariates](#example-data-with-instances-matched-by-one-or-more-covariates)
                * [Example: Ignore specified feature columns in data](#example-ignore-specified-feature-columns-in-data)
                * [Example: Specify features to treat as categorical](#example-specify-features-to-treat-as-categorical)
            * [Phase 2: Data Preprocessing](#phase-2-data-preprocessing)
            * [Phase 3: Feature Importance Evaluation](#phase-3-feature-importance-evaluation)
            * [Phase 4: Feature Selection](#phase-4-feature-selection)
            * [Phase 5: Machine Learning Modeling](#phase-5-machine-learning-modeling)
                * [Example: Run only one ML modeling algorithm](#example-run-only-one-ml-modeling-algorithm)
                * [Example: Utilize built-in algorithm feature importance estimates when available](#example-utilize-built-in-algorithm-feature-importance-estimates-when-available)
                * [Example: Specify an alternative primary evaluation metric](#example-specify-an-alternative-primary-evaluation-metric)
                * [Example: Reduce computational burden of algorithms that run slow in large instance spaces](#example-reduce-computational-burden-of-algorithms-that-run-slow-in-large-instance-spaces)
            * [Phase 6: Statistics Summary](#phase-6-statistics-summary)
            * [Phase 7: Compare Datasets](#phase-7-optional-compare-datasets)
            * [Phase 8: Generate PDF Training Summary Report](#phase-8-optional-generate-pdf-training-summary-report)
            * [Phase 9: Apply Models to Replication Data](#phase-9-optional-apply-models-to-replication-data)
            * [Phase 10: Generate PDF 'Apply Replication' Summary Report](#phase-10-optional-generate-pdf-apply-replication-summary-report)
            * [Phase 11: File Cleanup](#phase-11-optional-file-cleanup)
* [Other guidelines for STREAMLINE use](#other-guidelines-for-streamline-use)
* [Unique characteristics of STREAMLINE](#unique-characteristics-of-streamline)
* [Do even more with STREAMLINE](#do-even-more-with-streamline)          
* [Demonstration data](#demonstration-data)  
* [Troubleshooting](#troubleshooting)
    * [Rerunning a failed modeling job](#rerunning-a-failed-modeling-job)
    * [Unending modeling jobs](#unending-modeling-jobs)
* [Development notes](#development-notes)
    * [History](#history)
    * [Planned extensions/improvements](#planned-extensionsimprovements)
        * [Logistical extensions](#logistical-extensions)
        * [Capabilities extensions](#capabilities-extensions)
        * [Algorithmic extensions](#algorithmic-extensions)
* [Acknowledgements](#acknowledgements)
* [Citation](#citation)

***
## What level of computing skill is required for use?
STREAMLINE offers a variety of use options making it accessible to those with little or no coding experience as well as the seasoned programmer/data scientist. While there is currently no graphical user interface (GUI), the most naïve user need only know how to navigate their PC file system, specify folder/file paths, un-zip a folder, and have or set up a google drive account. Those with a very basic knowledge of python and computer environments can apply STEAMLINE within the included jupyter notebook, and those with a bit more experience can run it serially by command line or in parallel on a computing cluster.  Notably, the easier routes for using STEAMLINE are more computationally limited. Analyses using larger datasets, or a larger number of ML modeling algorithms turned on are best completed via command line (in parallel).

***
## What can it be used for?
STREAMLINE can be used as:
1. A tool to quickly run a rigorous ML data analysis over one or more datasets using one or more of the well-known or in-development modeling algorithms included
2. A framework to compare established scikit-learn compatible ML modeling algorithms to each other or to new algorithms
3. A baseline standard of comparison (i.e. negative control) with which to evaluate other AutoML tools that seek to optimize ML pipeline assembly as part of their methodology
4. A framework to quickly run an exploratory analysis and/or feature importance estimation/feature selection prior to using some other methodology for ML modeling
5. An educational example of how to integrate some of the many amazing python-based data science tools currently available (in particular pandas, scipy, optuna, and scikit-learn).
6. A framework from which to create a new, expanded, adapted, or modified ML analysis pipeline

***
## What does STREAMLINE include?
The automated elements of STREAMLINE includes (1) exploratory analysis, (2) basic data cleaning, (3) cross validation (CV) partitioning, (4) scaling, (5) imputation, (6) filter-based feature importance estimation, (7) collective feature selection, (8) modeling with 'Optuna' hyperparameter optimization across 15 implemented ML algorithms (including three rule-based machine learning algorithms: ExSTraCS, XCS, and eLCS, implemented by our research group), (9) testing evaluations with 16 classification metrics, model feature importance estimation, (10) automatic saving all results, models, and publication-ready plots (including proposed composite feature importance plots), (11) non-parametric statistical comparisons across ML algorithms and analyzed datasets, and (12) automatically generated PDF summary reports.

The following 15 scikit-learn compatible ML modeling algorithms are currently included as options: Naive Bayes (NB), Logistic Regression (LR), Decision Tree (DT), Random Forest (RF), Gradient Boosting (GB), XGBoost (XGB), LGBoost (LGB), CatBoost (CGB), Support Vector Machine (SVM), Artificial Neural Network (ANN), K-Nearest Neighbors (k-NN), Genetic Programming (GP), Eductional Learning Classifier System (eLCS), 'X' Classifier System (XCS), and Extended Supervised Tracking and Classifying System (ExSTraCS). Classification-relevant hyperparameter values and ranges have carefully selected for each and are pre-specified for the automated (Optuna-driven) automated hyperparameter sweep.

The automatically formatted PDF reports generated by STREAMLINE are intended to give a brief summary of pipeline settings and key results. A folder containing all results, statistical analyses publication-ready plots/figures, models, and other outputs is saved allowing users to carefully examine all aspects of analysis performance.  We have also included a variety of useful Jupyter Notebooks designed to operate on this output folder giving users quick paths to do even more with the pipeline output. Examples include: (1) Accessing prediction probabilities, (2) regenerating figures with custom tweaks, (3) trying out the effect of different prediction thresholds on selected models with an interactive slider, (4) re-evaluating models using a new prediction threshold, and (5) generating an interactive model feature importance ranking visualization across all ML algorithms. STREAMLINE is also fully reproducible and will train the same models with the same performance whenever the same datasets, pipeline settings, and random seed are used. The pipeline also outputs all CV training/testing datasets generated, along with relevant scaling and imputation objects so that users can easily run their own comparable ML analyses outside of STREAMLINE.

This pipeline does NOT automate the following elements, as they are still best completed by human experts: (1) feature engineering, or feature construction, (2) feature encoding (e.g. apply one-hot-encoding to categorical features, or numerically encode text-based feature values), (3) account for bias in data collection, or (4) anything beyond simple data cleaning (i.e. the pipeline only removes instances with no class label, or where all feature values are missing). We recommend users consider conducting these items, as needed, prior to applying STREAMLINE.

***
## How is STREAMLINE different from other AutoML tools?
Unlike most other AutoML tools, STREAMLINE was designed as a framework to rigorously apply and compare a variety of ML modeling algorithms and collectively learn from them as opposed to identifying a best performing model and/or attempting to optimize the analysis pipeline configuration itself. STREAMLINE adopts a fixed series of purposefully selected steps/phases in line with data science best practices. It seeks to automate all domain generalizable elements of an ML analysis pipeline with a specific focus on biomedical data mining challenges. This tool can be run or utilized in a number of ways to suite a variety experience levels and levels of problem/data complexity.

***
## STREAMLINE run modes
This multi-phase pipeline has been set up to run in one of four ways:

1. A 'Notebook' within Google Colaboratory [Almost Anyone]:
    * Advantages: (1) No coding or PC environment experience needed, (2) computing can be performed directly on Google Cloud, (3) one-click run of whole pipeline
    * Disadvantages: (1) Can only run pipeline serially, (2) slowest of the run options, (3) limited by google cloud computing allowances
    * Notes: Requires a Google and Google Drive account (free)

2. A Jupyter Notebook (included) [Basic Experience]:
    * Advantages: (1) Allows easy customizability of nearly all aspects of the pipeline with minimal coding/environment experience, (2) offers in-notebook viewing of results, (3) offers in-notebook documentation of the run phases, (4) one-click run of whole pipeline
    * Disadvantages: (1) Can only run pipeline serially, (2) slower runtime than from command-line
    * Notes: Requires Anaconda3, Python3, and several other minor Python package installations

3. Locally from the command line [Command-line Users]:
    * Advantages: (1) Typically runs faster than within Jupyter Notebook, (2) an easier more versatile option for those with command-line experience
    * Disadvantages: (1) Can only run pipeline serially, (2) command-line experience recommended
    * Notes: Requires Anaconda3, Python3, and several other minor Python package installations

4. Run in parallel from the command line using a computing cluster (only Linux-based cluster currently tested) [Computing Cluster Users]:
    * Advantages: (1) By far the fastest, most efficient way to run STREAMLINE, (2) offers parallelization within pipeline phases over separate datasets, cross-validation partitions, and ML algorithms.
    * Disadvantages: (1) Experience with command-line recommended (2) access to a computing cluster required
    * Notes: Requires Anaconda3, Python3, and several other minor Python package installations. Parallelization occurs within phases. Individual phases must be run in sequence.

Parallelized runs of STREAMLINE were set up to run on a Linux-based computing cluster employing IBM Spectrum LSF for job scheduling. See https://github.com/UrbsLab/I2C2-Documentation for a description of the computing cluster for which this functionality was originally designed). We have not yet tested parallelized STREAMLINE on other compute clusters or within cloud computing resources such as Microsoft Azure, Amazon Web Services, or Google Cloud. We aim to provide support for doing so in the future. In the meantime we welcome help in testing and extending this pipeline for computing resources such as these.  We expect only minor tweaks to the 'Main' scripts to be required to do so.

***
## View pipeline and output before running
* To quickly pre-view the pipeline (pre-run on included [demonstration datasets](#demonstration-data) without any installation whatsoever, open the following link:

https://colab.research.google.com/github/UrbsLab/STREAMLINE/blob/main/STREAMLINE-Notebook.ipynb  

Note, that with this link, you can only view the pre-run STREAMLINE Jupyter Notebook and will not be able to run or permanently edit the code. This is an easy way to get a feel for what the pipeline is and does.

* To quickly pre-view the folder of output files generated when running STREAMLINE on the [demonstration datasets](#demonstration-data), open the following link:

https://drive.google.com/drive/folders/1dgaXnJnzdthTxP914ALdrB4IBHjJdm1a?usp=sharing

***
## Implementation
STREAMLINE is coded in Python 3 relying heavily on pandas and scikit-learn as well as a variety of other python packages.

***
## Disclaimer
We make no claim that this is the best or only viable way to assemble an ML analysis pipeline for a given classification problem, nor that the included ML modeling algorithms will yield the best performance possible. We intend many expansions/improvements to this pipeline in the future to make it easier to use and hopefully more effective in application.  We welcome feedback, suggestions, and contributions for improvement.

***
# Installation and Use
In this section we first provide [instructions for users with little to no coding experience](#use-mode-1-google-colaboratory). Users with some coding experience can jump to the [standard instructions](#use-modes-2-4-standard-installation-and-use). However, all users would benefit from reviewing the following sections:

* [Inspecting your first run](#inspecting-your-first-run)
* [Running STREAMLINE on your own dataset(s)](#running-streamline-on-your-own-datasets)
* [Tips for reducing STREAMLINE runtime](#tips-for-reducing-streamline-runtime)
* [Tips for improving STREAMLINE modeling performance](#tips-for-improving-streamline-modeling-performance)
* [Code orientation](#code-orientation)

***
## Use Mode 1: Google Colaboratory
This is the easiest but most limited way to run STREAMLINE. These instructions are geared towards those with little to no computing experience. All other users can skip to the next [section](#use-modes-2-4-standard-installation-and-use) but may wish to revisit later parts of this section for helpful details.
* To learn more about Google Colaboratory prior to setup please visit the following link: https://research.google.com/colaboratory/

### Setting up your first run
Follow the steps below to get the pipeline running on the demonstration datasets. In summary, they detail the process of copying the STREAMLINE GitHub repository to your Google Drive, and running the notebook called `STREAMLINE-GoogleColabNotebook.ipynb` with Google Colaboratory.

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
    3. Load the individual STREAMLINE run files into memory.
    4. Install all other necessary python packages not already available in Anaconda3 (which is preloaded in the Google Colaboratory Environment).
    5. Run the entirety of STREAMLINE on the [demonstration datasets](#demonstration-data) folder (i.e. `DemoData`).
        * Note: all 5 steps should take approximately 3-5 minutes to run.
    6. Save all output files to `My Drive/STREAMLINE-main/Colab_Output/hcc_demo`

### Inspecting your first run
During or after the notebook runs, users can inspect the individual code and text (i.e. markdown) cells of the notebook. Individual cells can be collapsed or expanded by clicking on the small arrowhead on the left side of each cell. The first set of cells set up the coding environment automatically. Later cells are used to set the pipeline run parameters and then run the 11 phases of the pipeline in sequence. Some cells will display output figures generated by STREAMLINE. For example, scroll down to 'Phase 6: Statistics' and open the cell below the text 'Run Statistics Summary and Figure Generation'. Scrolling down this cell will first reveal the run commands calling relevant STREAMLINE code, then the figures generated by this phase. Note that to save runtime, this demonstration run is only applying three ML modeling algorithms: Naive Bayes, Logistic Regression, and Decision Trees.  These are typically the three fastest algorithms available in STREAMLINE.

Outside of the notebook, navigate back to your Google Drive and reopen the folder: `My Drive/STREAMLINE-main/Colab_Output`. You should find the saved experiment folder that was output by the run, called `hcc_demo`. Within this folder you should find the following:

* `hcc_demo_ML_Pipeline_Report.pdf` [File]: This is an automatically formatted PDF summarizing key findings during the model training and evaluation. A great place to start!
* `metadata.csv` [File]: Another way to view the STREAMLINE parameters used by the pipeline.  These are also organized on the first page of the PDF report.
* `metadata.pickle` [File]: A binary 'pickle' file of the metadata for easy loading by the 11 pipeline phases detailed in [Code Orientation](#code-orientation). (For more experienced users)
* `algInfo.pickle` [File]: A binary 'pickle' file including a dictionary indicating which ML algorithms were used, along with abbreviations of names for figures/filenames, and colors to use for each algorithm in figures. (For more experienced users)
* `DatasetComparisons` [Folder]: Containing figures and statistical significance comparisons between the two datasets that were analyzed with STREAMLINE. (This folder only appears if more than one dataset was included in the user specified data folder, i.e. `data_path`, and phase 7 of STREAMLINE was run). Within the PDF summary, each dataset is assigned an abbreviated designation of 'D#' (e.g. D1, D2, etc) based on the alphabetical order of each dataset name. These designations are used in some of the files included within this folder.
* [Folders] - A folder for each of the two datasets analyzed (in this demo there were two: `hcc-data_example` and `hcc-data_example_no_covariates`). These folders include all results and models respective to each dataset. We summarize the contents of each folder below (feel free to skip this for now and revisit it as needed)...
    * `exploratory` [Folder]: Includes all exploratory analysis summaries and figures.
    * `CVDatasets` [Folder]: Includes all individual training and testing datasets (as .csv files) generated.
        * Note: These are the datasets passed to modeling so if imputation and scaling was conducted, these datasets will have been partitioned, imputed, and scaled.
    * `scale_impute` [Folder]: Includes all pickled files preserving how scaling and/or imputation was conducted based on respective training datasets.
    * `feature_selection` [Folder]: Includes feature importance and selection summaries and figures.
    * `models` [Folder]: Includes the ML algorithm hyperparameters selected by Optuna for each CV partition and modeling algorithm, as well as pickled files storing all models for future use.
    * `model_evaluation` [Folder]: Includes all model evaluation results, summaries, figures, and statistical comparisons.
    * `applymodel` [Folder]: Includes all model evaluation results when applied to a hold out replication datasets. This includes a new PDF summary of models when applied to this further hold-out dataset.
        * Note: In the demonstration analysis we only created and applied a replication dataset for `hcc-data_example`. Therefore this folder only appears in output folder for `hcc-data_example`.
    * `runtimes.csv` [File]: Summary file giving the total runtimes spent on each phase or ML modeling algorithm in STREAMLINE.

### Running STREAMLINE on your own dataset(s)
This section explains how to update the Google Colaboratory Notebook to run on one or more user specified datasets rather than the [demonstration datasets](#demonstration-data). This instructions are effectively the same for running STREAMLINE from Jupyter Notebook. Note that, for brevity, the parameter names given below are slightly different from the argument identifiers when using STREAMLINE from the command-line (see [here](#phase-details-run-parameters-and-additional-examples)).

1. Within your `STREAMLINE-main` folder on Google Drive, add, or copy in a new folder, that has no spaces in it's name (e.g. `my_data`)
2. Place 1 or more 'target' datasets in this folder following these requirements:
    * Files are in comma-separated format with extension '.txt' or '.csv' format.
    * Missing data values should be empty or indicated with an 'NA'.
    * Dataset(s) include a header giving column labels.
    * Data columns should only include features (i.e. independant variables), a class label, and [optionally] instance (i.e. row) labels, and/or match labels (if matched cross validation will be used).
    * Binary class values are encoded as 0 (e.g. negative), and 1 (positive) with respect to true positive, true negative, false positive, false negative metrics. PRC plots focus on classification of 'positives'.
    * All feature values (both categorical and quantitative) are numerically encoded (i.e. no letters or words). Scikit-learn does not accept text-based values.
        * However both `instance_label` and `match_label` values may be either numeric or text.
    * Place all datasets to be analyzed simultaneously into the new folder created above (e.g. `my_data`).
        * If multiple datasets are being analyzed they must each have the same `class_label` (e.g. 'Class'), and (if present), the same `instance_label` (e.g. 'ID') and `match_label` (e.g. 'Match_ID').
3. Open `STREAMLINE-GoogleColabNotebook.ipynb` in Google Colaboratory.
4. Scroll down to the 5th code block with the text 'Mandatory Run Parameters for Pipeline' above it.
5. Update the first 6 pipeline run parameters as such:
    * `demo_run`: Change from True to False (Note, this parameter is only used by the notebooks for the demonstration analysis, and is one of the few parameters that use a Boolean rather than string value).
    * `data_path`: Change the end of the path from DemoData to the name of your new dataset folder (e.g. "/content/drive/MyDrive/STREAMLINE-main/my_data").
    * `output_path`: This can be left 'as-is' or modified to some other folder on your google drive within which to store all STREAMLINE experiments.
    * `experiment_name`: Change this to some new unique experiment name (do this each time you want to run a new experiment, either on the same or different dataset(s)), e.g. 'my_first_experiment'.
    * `class_label`: Change to the column header indicating the class label in each dataset, e.g. 'Class'.
    * `instance_label`: Change to the column header indicating unique instance ID's for each row in the dataset(s), or change to the string 'None' if your dataset does not include instance IDs.
6. Specifying replication data run parameters:
    * Scroll down to the 13th code block with the text 'Run Parameters for Phase 10'.
    * If you have a hold out replication dataset (with the same set of columns as one of the original 'target' datasets), you can apply the trained models to this new data for evaluation. If so, update the following run parameters.
        * `rep_data_path`: Change the path to point to a different folder on Google Drive that contains one or more replication datasets intended to be applied to one of your original 'target' datasets.
        * `data_path_for_rep`: Change to the path+filename of the 'target' dataset used to train the models we wish to apply the replication data to. This will be a specific file from the first dataset folder you created (e.g. `my_data`)
        * Note: The run parameters described so far are the only essential ones to consider when setting up the pipeline to run on new data(sets).
    * If you don't have a replication dataset simply change `applyToReplication` to False (boolean value) and ignore the other two run parameters in this code block.
9. [Optional] Update other STREAMLINE run parameters to suit your analysis needs within code blocks 6-14. We will cover some common run parameters to consider here:
    * `cv_partitions`: The number of CV training/testing partitions created, and consequently the number of models trained for each ML algorithm. We recommend setting this between 3-10. A larger value will take longer to run but produce more accurate results.
    * `categorical_cutoff`: STREAMLINE uses this parameter to automatically determine which features to treat as categorical vs. numeric. If a feature has more than this many unique values, it is considered to be numeric.
        * Note: Currently, STREAMLINE does NOT automatically apply one-hot-encoding to categorical features meaning that all features will still be treated as numerical during ML modeling. Its currently up to the users decide whether to pre-encode features.  However STREAMLINE does take feature type into account during both the exploratory analysis, data preprocessing, and feature importance phases.
        * Note: Users can also manually specify which features to treat as categorical or even to point to features in the dataset that should be ignored in the analysis with the parameters `ignore_features_path` and `categorical_feature_path`, respectively. For either, instead of the default string 'None' setting the user specifies the path to a .csv file including a row of feature names from the dataset that should either be treated as categorical or ignored, respectively.
    * `do_all`: Setting this to 'True' will run all ML algorithms available by default, unless a specific do_ALGORITHM parameter is set to 'False' Setting this to 'False' will run no algorithms unless a specific do_ALGORITHM parameter is set to 'True'
    * do_ALGORITHM: There are 15 possible algorithms use parameters (e.g. `do_NB`, `do_XGB`, etc).  Setting any of these to 'True' or 'False' will override the default setting and run or not run that specific algorithm. Any set to 'None' (default setting from command-line) will have its 'True' or 'False' value specified by `do_all`.
    * `n_trials`: Set to a higher value to give Optuna more attempts to optimize hyperparameter settings.
    * `timeout`: Set higher to increase the maximum time allowed for Optuna to run the specified `n_trials` (useful for algorithms that take more time to run)
* Note: There are a number of other run parameter options and we encourage users to read descriptions of each to see what other options are available.
* Note: Code block 11 (i.e. Hyperparameter Sweep Options for ML Algorithms) includes carefully selected relevant hyperparameter options for each ML algorithm. We advise users to modify these with caution. Further information regarding the hyperparameters for each algorithm are included as commented links within code block 11.

### Tips for reducing STREAMLINE runtime
Conducting a more effective ML analysis typically demands a much larger amount of computing power and runtime. However, we provide general guidelines here for limiting overall runtime of a STREAMLINE experiment.
1. Run on a fewer number of datasets at once.
2. Run using fewer ML algorithms at once:
    * Naive Bayes, Logistic Regression, and Decision Trees are typically fastest.
    * Genetic Programming, eLCS, XCS, and ExSTraCS often take the longest (however other algorithms such as SVM, KNN, and ANN can take even longer when the number of instances is very large).
3. Run using a smaller number of `cv_partitions`.
4. Run without generating plots (i.e. `export_feature_correlations`, `export_univariate_plots`, `plot_PRC`, `plot_ROC`, `plot_FI_box`, `plot_metric_boxplots`).
5. In large datasets with missing values, set `multi_impute` to 'False'. This will apply simple mean imputation to numerical features instead.
6. Set `use_TURF` as 'False'. However we strongly recommend setting this to 'True' in feature spaces > 10,000 in order to avoid missing feature interactions during feature selection.
7. Set `TURF_pct` no lower than 0.5.  Setting at 0.5 is by far the fastest, but it will operate more effectively in very large feature spaces when set lower.
8. Set `instance_subset` at or below 2000 (speeds up multiSURF feature importance evaluation at potential expense of performance).
9. Set `max_features_to_keep` at or below 2000 and `filter_poor_features` = 'True' (this limits the maximum number of features that can be passed on to ML modeling).
10. Set `training_subsample` at or below 2000 (this limits the number of sample used to train particularly expensive ML modeling algorithms). However avoid setting this too low, or ML algorithms may not have enough training instances to effectively learn.
11. Set `n_trials` and/or timeout to lower values (this limits the time spent on hyperparameter optimization).
12. If using eLCS, XCS, or ExSTraCS, set `do_lcs_sweep` to 'False', `iterations` at or below 200000, and `N` at or below 2000.

### Tips for improving STREAMLINE modeling performance
* Generally speaking, the more computational time you are willing to spend on ML, the better the results. Doing the opposite of the above tips for reducing runtime, will likely improve performance.
* In certain situations, `feature_selection` to 'False', and relying on the ML algorithms alone to identify relevant features will yield better performance.  However, this may only be computationally practical when the total number of features in an original dataset is smaller (e.g. under 2000).
* Note that eLCS, XCS, and ExSTraCS are newer algorithm implementations developed by our research group.  As such, their algorithm performance may not yet be optimized in contrast to the other well established and widely utilized options. These learning classifier system (LCS) algorithms are unique however, in their ability to model very complex associations in data, while offering a largely interpretable model made up of simple, human readable IF:THEN rules. They have also been demonstrated to be able to tackle both complex feature interactions as well as heterogeneous patterns of association (i.e. different features are predictive in different subsets of the training data).
* In problems with no noise (i.e. datasets where it is possible to achieve 100% testing accuracy), LCS algorithms (i.e. eLCS, XCS, and ExSTraCS) perform better when `nu` is set larger than 1 (i.e. 5 or 10 recommended).  This applies significantly more pressure for individual rules to achieve perfect accuracy.  In noisy problems this may lead to significant overfitting.

***
## Use Modes 2-4: Standard Installation and Use
This section covers the general installation instructions for all users with basic Python/Environment experience. These instructions are relevant to use modes 2, 3, and 4 (i.e. within Jupyter Notebook, local command-line run, and parallelized linux-based computing cluster run).

### Prerequisites
#### Anaconda3
To be able to run STREAMLINE you will need Anaconda (recommended rather than individually installing all individual packages) including Python3, and additionally it requires a handful of other Python packages not included within Anaconda. Anaconda is a distribution of Python and R programming languages for scientific computing, that aims to simplify package management and deployment. The distribution includes data-science packages suitable for Windows, Linux, and macOS. We recommend installing the most recent stable version of Anaconda (https://docs.anaconda.com/anaconda/install/) within your computing environment. Make sure to install a version appropriate for your operating system. Anaconda also includes Jupyter Notebook.

We confirmed STREAMLINE functionality in Jupyter Notebook and on a local PC command-line run with Microsoft Windows 10 using:

https://repo.anaconda.com/archive/Anaconda3-2021.05-Windows-x86_64.exe which has Python 3.8.8.
    * Note: This version of anaconda worked with kaleido (v_0.0.3.post1) and scipy (v_1.5.0), see below.

We also confirmed STREAMLINE functionality on our Linux-based computing cluster using the following:
    * Note: We followed the cluster anaconda installation instructions given in the section 'Installing Anaconda' at https://github.com/UrbsLab/I2C2-Documentation.

1. https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh which has Python 3.8.3
    * Note: This version of anaconda worked with kaleido (v_0.0.3.post1) and scipy (v_1.5.0), see below.

2. https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh which has Python 3.9.12
    * Note: This version of anaconda instead required kaleido (v_0.2.1) and scipy (v_1.8.0), see below.

#### Additional Python Packages
In addition to the above you will also need to install the following packages (not included in Anaconda3) in your computing environment. This can be done at once with one of the following commands:

* For 'Anaconda3-2021.05-Windows-x86_64' or Anaconda3-2020.07-Linux-x86_64, or other older Anaconda versions (not tested) use:
```
pip install skrebate==0.7 xgboost lightgbm catboost gplearn scikit-eLCS scikit-XCS scikit-ExSTraCS optuna plotly kaleido==0.0.3.post1 fpdf  
```

* For Anaconda3-2022.05-Linux-x86_64 or other newer Anaconda versions (not tested) use:
```
pip install skrebate==0.7 xgboost lightgbm catboost gplearn scikit-eLCS scikit-XCS scikit-ExSTraCS optuna plotly kaleido fpdf scipy  
```

Below we detail these packages, each given with the respective versions used at the time of confirming STREAMLINE functionality):

* skrebate (v_0.7) scikit-learn compatible version of ReBATE, a suite of Relief-based feature selection algorithms (0.7). There is currently a PyPi issue requiring that the newest version (i.e. v_0.7) be explicitly installed.

* xgboost (v_1.2.0) Extreme Gradient Boosting ML classification algorithm

* lightgbm (v_3.0.0) Light Gradient Boosting ML classification algorithm

* catboost (v_1.0.5) Catagory Gradient Boosting ML classification algorithm

* gplearn (v_0.4.2) Suite of Genetic Programming tools from which we use Genetic Programming Symbolic Classification as an ML algorithm

* scikit-eLCS (v_1.2.4) Educational Learning Classifier System ML algorithm (Implemented by our lab)

* scikit-XCS (v_1.0.8) 'X' Learning Classifier System ML algorithm (Implemented by our lab)

* scikit-ExSTraCS (v_1.1.0) Extended Supervised Learning and Tracking Classifier System ML algorithm (Developed and Implemented by our lab)

* optuna (v_2.9.1) Optuna, a hyperparameter optimization framework

* plotly (v_5.1.0) An open-source, interactive data visualization library. Used by Optuna to generate hyperparameter sweep visualizations

* kaleido (v_0.0.3.post1 or v_0.2.1, see above) A package for static image export for web-based visualization. This again is needed to generate hyperparameter sweep visualizations in Optuna. We found that getting this package to work properly can be tricky based on the anaconda version. If the pipeline is getting hung up in modeling, try setting 'export_hyper_sweep_plots' to False to avoid the issue. These plots are nice to have but not necessary for the overall pipeline.

* scipy (v_1.5.0 or v_1.8.0, see above) When using the newest anaconda installation it is currently advised to update scipy by installing it again here to avoid a known compatability issue with mannwhitneyu.

Note: Users can check the version of each package from command-line with the following command (replacing 'package' with the corresponding package name above)
```
pip show package
```

### Download STREAMLINE
To use STREAMLINE, download this repository to your working directory.

***
### Code Orientation
The base code for STREAMLINE is organized into a series of script phases designed to best optimize the parallelization of a given analysis. These loosely correspond with the pipeline schematic above. These phases are designed to be run in order. Phases 1-6 make up the core automated pipeline, with Phase 7 and beyond being run optionally based on user needs.

* Phase 1: Exploratory Analysis
  * Conducts an initial exploratory analysis of all target datasets to be analyzed and compared
  * Conducts basic data cleaning
  * Conducts k-fold cross validation (CV) partitioning to generate k training and k testing datasets
  * \[Code]: `ExploratoryAnalysisMain.py` and `ExploratoryAnalysisJob.py`
  * \[Runtime]: Typically fast, with the exception of generating feature correlation heatmaps in datasets with a large number of features

* Phase 2: Data Preprocessing
  * Conducts feature transformations (i.e. data scaling) on all CV training datasets individually
  * Conducts imputation of missing data values (missing data is not allowed by most scikit-learn modeling packages) on all CV training datasets individually
  * Generates updated training and testing CV datasets
  * \[Code]: `DataPreprocessingMain.py` and `DataPreprocessingJob.py`
  * \[Runtime]: Typically fast, with the exception of imputing larger datasets with many missing values

* Phase 3: Feature Importance Evaluation
  * Conducts feature importance estimations on all CV training datasets individually
  * Generates updated training and testing CV datasets
  * \[Code]: `FeatureImportanceMain.py` and `FeatureImportanceJob.py`
  * \[Runtime]: Typically reasonably fast, takes more time to run MultiSURF as the number of training instances approaches the default for 'instance_subset', or this parameter set higher in larger datasets

* Phase 4: Feature Selection
  * Applies 'collective' feature selection within all CV training datasets individually
  * Features removed from a given training dataset are also removed from corresponding testing dataset
  * Generates updated training and testing CV datasets
  * [Code]: `FeatureSelectionMain.py` and `FeatureSelectionJob.py`
  * [Runtime]: Fast

* Phase 5: Machine Learning Modeling
  * Conducts hyperparameter sweep for all ML modeling algorithms individually on all CV training datasets
  * Conducts 'final' modeling for all ML algorithms individually on all CV training datasets using 'optimal' hyperparameters found in previous step
  * Calculates and saves all evaluation metrics for all 'final' models
  * \[Code]: `ModelMain.py` and `ModelJob.py`
  * \[Runtime]: Slowest phase, can be sped up by reducing the set of ML methods selected to run, or deactivating ML methods that run slowly on large datasets

* Phase 6: Statistics Summary
  * Combines all results to generate summary statistics files, generate results plots, and conduct non-parametric statistical significance analyses comparing ML model performance across CV runs
  * \[Code]: `StatsMain.py` and `StatsJob.py`
  * \[Runtime]: Moderately fast

* Phase 7: [Optional] Compare Datasets
  * NOTE: Only can be run if the STREAMLINE was run on more than dataset
  * Conducts non-parametric statistical significance analyses comparing separate original 'target' datasets analyzed by pipeline
  * \[Code]: `DataCompareMain.py` and `DataCompareJob.py`
  * \[Runtime]: Fast

* Phase 8: [Optional] Generate PDF Training Summary Report
  * Generates a pre-formatted PDF including all pipeline run parameters, basic dataset information, and key exploratory analyses, ML modeling results, statistical comparisons, and runtime.
  * \[Code]: `PDF_ReportMain.py` and `PDF_ReportJob.py`
  * \[Runtime]: Moderately fast

* Phase 9: [Optional] Apply Models to Replication Data
  * Applies all previously trained models for a single 'target' dataset to one or more new 'replication' dataset(s) that has the same features from an original 'target' dataset
  * Conducts exploratory analysis on new 'replication' dataset(s)
  * Applies scaling, imputation, and feature selection (unique to each CV partition from model training) to new 'replication' dataset(s) in preparation for model application
  * Evaluates performance of all models the prepared 'replication' dataset(s)
  * Generates summary statistics files, results plots, and conducts non-parametric statistical significance analyses comparing ML model performance across replications CV data transformations
  * NOTE: feature importance evaluation and 'target' dataset statistical comparisons are irrelevant to this phase
  * \[Code]: `ApplyModelMain.py` and `ApplyModelJob.py`
  * \[Runtime]: Moderately fast

* Phase 10: [Optional] Generate PDF 'Apply Replication' Summary Report
  * Generates a pre-formatted PDF including all pipeline run parameters, basic dataset information, and key exploratory analyses, ML modeling results, and statistics.
  * \[Code]: `PDF_ReportMain.py` and `PDF_ReportJob.py`
  * \[Runtime]: Moderately fast

* Phase 11: [Optional] File Cleanup
  * Deletes files that do not need to be kept following pipeline run.
  * \[Code]: `FileCleanup.py`
  * \[Runtime]: Fast

***
### Run From Jupyter Notebook
Here we detail how to run STREAMLINE within the provided Jupyter Notebook named `STREAMLINE-Notebook.ipypnb`. This included notebook is set up to run on the included [demonstration datasets](#demonstration-data). However users will still need to modify the local folder/file paths in this notebook for it to be able for it to correctly run the demonstration within Jupyter Notebook.

1. First, ensure all prerequisite packages are [installed](#prerequisites) in your environment and dataset assumptions ([above](#running-streamline-on-your-own-datasets)) are satisfied.

2. Open Jupyter Notebook (info about Jupyter Notebooks here, https://jupyter.readthedocs.io/en/latest/running.html). We recommend opening the 'anaconda prompt' which comes with your anaconda installation.  Once opened, type the command `jupyter notebook` which will open as a webpage. Navigate to your working directory and open the included jupyter notebook file: `STREAMLINE-Notebook.ipynb`.

3. Scroll down to the second code block of the notebook below the header 'Mandatory Parameters to Update' and update the following run parameters to reflect paths on your PC.
    * `data_path`: Change the path so it reflects the location of the `DemoData` folder (within the STREAMLINE folder) on your PC, e.g. `C:/Users/ryanu/Documents/GitHub/STREAMLINE/DemoData`.
    * `output_path`: Change the path to specify the desired location for the STREAMLINE output experiment folder.

4. Click `Kernel` on the Jupyter notebook GUI, and select `Restart & Run All` to run the script.  

5. Running the included [demonstration datasets](#demonstration-data) with the pre-specified notebook run parameters, should only take a 3-5 minutes depending on your PC hardware.
    * Note: It may take several hours or more to run this notebook in other contexts. Parameters that impact runtimes are discussed in [this section](#tips-for-reducing-streamline-runtime) above. We recommend users with larger analyses to use a computing cluster if possible.

#### Running STREAMLINE Jupyter Notebook On Your Own Dataset(s)
Follow the same steps as above, but update code blocks 2-11 in the Jupyter Notebook to reflect the correct dataset configurations and analysis needs as covered in [this section](#running-streamline-on-your-own-datasets) above.

***
### Run From Command Line (Local or Cluster Parallelization)
The primary way to run STREAMLINE is via the command line. In this section we provide example commands for running STREAMLINE on the [demonstration datasets](#demonstration-data).

Note, each phase must be run to completion before starting the next. They should not all be run at once!

As indicated above, each phase can run locally (not parallelized) or parallelized using a Linux based computing cluster. Parallelization occurs within each phase distributing the workload of running multiple datasets, across multiple CV partitions, and across multiple algorithms. It does nothing to speed up individual algorithms. For example, when running `ModelMain.py` using all 15 algorithms, 10-fold CV, applied to 4 datasets, (15 x 10 x 4) parallelization submits 600 individual jobs to the compute cluster to run simultaneously (if available compute resources allows). 

With a little tweaking of the respective 'Main' scripts, this code should be adaptable to be parallelized on other cluster frameworks (i.e. non LSF) or with cloud computing.

A detailed overview of all available STREAMLINE run parameters is given in [Phase Details (Run Parameters and Additional Examples)](#phase-details-run-parameters-and-additional-examples).

#### Local Run Example
Below we give an example of the set of commands needed to run STREAMLINE in it's entirety (all possible 11 phases) on the [demonstration datasets](#demonstration-data) using mostly default run parameters. In this example we specify instance and class label run parameters to emphasize the importance setting these values correctly. Note that arguments with file or folder paths must be updated to reflect the actual location of these files/folders on your system.
```
python ExploratoryAnalysisMain.py --data-path /mydatapath/STREAMLINE/DemoData --out-path /myoutputpath --exp-name hcc_demo --inst-label InstanceID --class-label Class --run-parallel False

python DataPreprocessingMain.py --out-path /myoutputpath --exp-name hcc_demo --run-parallel False

python FeatureImportanceMain.py --out-path /myoutputpath --exp-name hcc_demo --run-parallel False

python FeatureSelectionMain.py --out-path /myoutputpath --exp-name hcc_demo --run-parallel False

python ModelMain.py --out-path /myoutputpath --exp-name hcc_demo --run-parallel False

python StatsMain.py --out-path /myoutputpath --exp-name hcc_demo --run-parallel False

python DataCompareMain.py --out-path /myoutputpath --exp-name hcc_demo --run-parallel False

python PDF_ReportMain.py --out-path /myoutputpath --exp-name hcc_demo --run-parallel False

python ApplyModelMain.py --out-path /myoutputpath --exp-name hcc_demo --rep-data-path /myrepdatapath/STREAMLINE/DemoRepData  --data-path /mydatapath/STREAMLINE/DemoData/hcc-data_example.csv --run-parallel False

python PDF_ReportMain.py --training False --out-path /myoutputpath --exp-name hcc_demo --rep-data-path /myrepdatapath/STREAMLINE/DemoRepData  --data-path /mydatapath/STREAMLINE/DemoData/hcc-data_example.csv --run-parallel False

python FileCleanup.py --out-path /myoutputpath --exp-name hcc_demo
```

#### Computing Cluster Run (Parallelized) Example
Below we give the same set of STREAMLINE run commands, however in each, the run parameter `--run-parallel` is left to its default value of 'True'. This will submit jobs to an LSF job scheduler. Note that the last phase, `FileCleanup.py` will run on the head node, and is not submitted as a job.
```
python ExploratoryAnalysisMain.py --data-path /mydatapath/STREAMLINE/DemoData --out-path /myoutputpath --exp-name hcc_demo --inst-label InstanceID --class-label Class

python DataPreprocessingMain.py --out-path /myoutputpath --exp-name hcc_demo

python FeatureImportanceMain.py --out-path /myoutputpath --exp-name hcc_demo

python FeatureSelectionMain.py --out-path /myoutputpath --exp-name hcc_demo

python ModelMain.py --out-path /myoutputpath --exp-name hcc_demo

python StatsMain.py --out-path /myoutputpath --exp-name hcc_demo

python DataCompareMain.py --out-path /myoutputpath --exp-name hcc_demo

python PDF_ReportMain.py --out-path /myoutputpath --exp-name hcc_demo

python ApplyModelMain.py --out-path /myoutputpath --exp-name hcc_demo --rep-data-path /myrepdatapath/STREAMLINE/DemoRepData  --data-path /mydatapath/STREAMLINE/DemoData/hcc-data_example.csv

python PDF_ReportMain.py --training False --out-path /myoutputpath --exp-name hcc_demo --rep-data-path /myrepdatapath/STREAMLINE/DemoRepData  --data-path /mydatapath/STREAMLINE/DemoData/hcc-data_example.csv

python FileCleanup.py --out-path /myoutputpath --exp-name hcc_demo
```

***
### Checking Phase Completion
After running any of Phases 1-10 a 'phase-complete' file is automatically generated for each job run locally or in parallel.  Users can confirm that all jobs for that phase have been completed by running the phase command again, this time with the argument `-c`. Any incomplete jobs will be listed, or an indication of successful completion will be returned.

For example, after running `ModelMain.py`, the following command can be given to check whether all jobs have been completed.
```
python ModelMain.py --out-path /myoutputpath --exp-name hcc_test -c
```

***
### Phase Details (Run Parameters and Additional Examples)
Here we review the run parameters available for each of the 11 phases and provide some additional run examples. We remind users that the parameter names described for the above notebooks sometimes different than the argument names when using STREAMLINE from the command-line (for brevity). The additional examples illustrate how to flexibly adapt STREAMLINE to user needs. All examples below assume that class and instance labels set to default values for simplicity. Run parameters that are necessary to set are marked as 'MANDATORY' under 'Default Value'.

#### Phase 1: Exploratory Analysis
Run parameters for `ExploratoryAnalysisMain.py`:

| Argument | Description | Default Value |
|:-------- |:---------------------  | ----------- |
| --data-path | path to directory containing datasets | MANDATORY |
| --out-path | path to output directory | MANDATORY |
| --exp-name | name of experiment output folder (no spaces) | MANDATORY |
| --class-label | outcome label of all datasets | Class |
| --inst-label | instance label of all datasets (if present) | None |
| --fi | path to .csv file with feature labels to be ignored in analysis | None |
| --cf | path to .csv file with feature labels specified to be treated as categorical | None |
| --cv | number of CV partitions | 10 |
| --part | 'S', or 'R', or 'M', for stratified, random, or matched, respectively | S |
| --match-label | only applies when M selected for partition-method; indicates column with matched instance ids | None |
| --cat-cutoff | number of unique values after which a variable is considered to be quantitative vs categorical | 10 |
| --sig | significance cutoff used throughout pipeline | 0.05 |
| --export-fc | run and export feature correlation analysis (yields correlation heatmap) | True |
| --export-up | export univariate analysis plots (note: univariate analysis still output by default) | False |
| --rand-state | "Dont Panic" - sets a specific random seed for reproducible results | 42 |
| --run-parallel | if run parallel on LSF compatible computing cluster | True |
| --queue | specify name of parallel computing queue (uses our research groups queue by default) | i2c2_normal |
| --res-mem | reserved memory for the job (in Gigabytes) | 4 |
| --max-mem | maximum memory before the job is automatically terminated | 15 |
| -c | Boolean: Specify whether to check for existence of all output files | Stores False |

##### Example: Data with instances matched by one or more covariates
Run on dataset with a match label (i.e. a column that identifies groups of instances matched by one or more covariates to remove their effect). Here we specify the use of matched CV partitioning and indicate the column label including the matched instance group identifiers. All instances with the same unique identifier in this column are assumed to be a part of a matched group, and are kept together within a given data partition.
```
python ExploratoryAnalysisMain.py --data-path /mydatapath/STREAMLINE/MyMatchedData --out-path /myoutputpath --exp-name my_match_test --part M --match-label MyMatchGroups
```

##### Example: Ignore specified feature columns in data
A convenience for running the analysis, but ignoring one or more feature columns that were originally included in the dataset.  
```
python ExploratoryAnalysisMain.py --data-path /mydatapath/STREAMLINE/MyDataFolder --out-path /myoutputpath --exp-name hcc_test --fi /someOtherPath/ignoreFeatureList.csv
```

##### Example: Specify features to treat as categorical
By default STREAMLINE uses the `--cat-cutoff` parameter to try and automatically decide what features to treat as categorical (i.e. are there < 10 unique values in the feature column) vs. continuous valued. With this option the user can specify the list of feature names to explicitly treat as categorical. Currently this only impacts the exploratory analysis as well as the imputation in data preprocessing. The identification of categorical variables within STREAMLINE has no impact on ML modeling.
```
python ExploratoryAnalysisMain.py --data-path /mydatapath/STREAMLINE/MyDataFolder --out-path /myoutputpath --exp-name hcc_test --cf /someOtherPath/categoricalFeatureList.csv
```

#### Phase 2: Data Preprocessing
Run parameters for `DataPreprocessingMain.py`:

| Argument | Description | Default Value |
|:-------- |:---------------------  | ----------- |
| --out-path | path to output directory | MANDATORY |
| --exp-name | name of experiment output folder (no spaces) | MANDATORY |
| --scale | perform data scaling (required for SVM, and to use Logistic regression with non-uniform feature importance estimation) | True |
| --impute | perform missing value data imputation (required for most ML algorithms if missing data is present) | True |
| --multi-impute | applies multivariate imputation to quantitative features, otherwise uses median imputation | True |
| --over-cv | overwrites earlier cv datasets with new scaled/imputed ones | True |
| --run-parallel | if run parallel on LSF compatible computing cluster | True |
| --queue | specify name of parallel computing queue (uses our research groups queue by default) | i2c2_normal |
| --res-mem | reserved memory for the job (in Gigabytes) | 4 |
| --max-mem | maximum memory before the job is automatically terminated | 15 |
| -c | Boolean: Specify whether to check for existence of all output files | Stores False |

#### Phase 3: Feature Importance Evaluation
Run parameters for `FeatureImportanceMain.py`:

| Argument | Description | Default Value |
|:-------- |:---------------------  | ----------- |
| --out-path | path to output directory | MANDATORY |
| --exp-name | name of experiment output folder (no spaces) | MANDATORY |
| --do-mi | do mutual information analysis | True |
| --do-ms | do multiSURF analysis | True |
| --use-turf | use TURF wrapper around MultiSURF | False |
| --turf-pct | proportion of instances removed in an iteration (also dictates number of iterations) | 0.5 |
| --n-jobs | number of cores dedicated to running algorithm; setting to -1 will use all available cores | 1 |
| --inst-sub | sample subset size to use with multiSURF | 2000 |
| --run-parallel | if run parallel on LSF compatible computing cluster | True |
| --queue | specify name of parallel computing queue (uses our research groups queue by default) | i2c2_normal |
| --res-mem | reserved memory for the job (in Gigabytes) | 4 |
| --max-mem | maximum memory before the job is automatically terminated | 15 |
| -c | Boolean: Specify whether to check for existence of all output files | Stores False |

#### Phase 4: Feature Selection
Run parameters for `FeatureSelectionMain.py`:

| Argument | Description | Default Value |
|:-------- |:---------------------  | ----------- |
| --out-path | path to output directory | MANDATORY |
| --exp-name | name of experiment output folder (no spaces) | MANDATORY |
| --max-feat | max features to keep. None if no max | 2000 |
| --filter-feat | filter out the worst performing features prior to modeling | True |
| --top-features | number of top features to illustrate in figures | 20 |
| --export-scores | export figure summarizing average feature importance scores over cv partitions | True |
| --over-cv | overwrites working cv datasets with new feature subset datasets | True |
| --run-parallel | if run parallel on LSF compatible computing cluster | True |
| --queue | specify name of parallel computing queue (uses our research groups queue by default) | i2c2_normal |
| --res-mem | reserved memory for the job (in Gigabytes) | 4 |
| --max-mem | maximum memory before the job is automatically terminated | 15 |
| -c | Boolean: Specify whether to check for existence of all output files | Stores False |

#### Phase 5: Machine Learning Modeling
Run parameters for `ModelMain.py`:

| Argument | Description | Default Value |
|:-------- |:---------------------  | ----------- |
| --out-path | path to output directory | MANDATORY |
| --exp-name | name of experiment output folder (no spaces) | MANDATORY |
| --do-all | run all modeling algorithms by default (when set False, individual algorithms are activated individually) | True |
| --do-NB | run naive bayes modeling | None |
| --do-LR | run logistic regression modeling | None |
| --do-DT | run decision tree modeling | None |
| --do-RF | run random forest modeling | None |
| --do-GB | run gradient boosting modeling | None |
| --do-XGB | run XGBoost modeling | None |
| --do-LGB | run LGBoost modeling | None |
| --do-CGB | run Catboost modeling | None |
| --do-SVM | run support vector machine modeling | None |
| --do-ANN | run artificial neural network modeling | None |
| --do-KNN | run k-nearest neighbors classifier modeling | None |
| --do-GP | run genetic programming symbolic classifier modeling | None |
| --do-eLCS | run eLCS modeling (a basic supervised-learning learning classifier system) | None |
| --do-XCS | run XCS modeling (a supervised-learning-only implementation of the best studied learning classifier system) | None |
| --do-ExSTraCS | run ExSTraCS modeling (a learning classifier system designed for biomedical data mining) | None |
| --metric |primary scikit-learn specified scoring metric used for hyperparameter optimization and permutation-based model feature importance evaluation | balanced_accuracy |
| --subsample | for long running algos (XGB,SVM,ANN,KN), option to subsample training set (0 for no subsample) | 0 |
| --use-uniformFI | overrides use of any available feature importance estimate methods from models, instead using permutation_importance uniformly | True |
| --n-trials | # of bayesian hyperparameter optimization trials using optuna | 100 |
| --timeout | seconds until hyperparameter sweep stops running new trials (Note: it may run longer to finish last trial started) | 300 |
| --export-hyper-sweep | export optuna-generated hyperparameter sweep plots | False |
| --do-LCS-sweep | do LCS hyperparam tuning or use below params | False |
| --nu | fixed LCS nu param (recommended range 1-10), set to larger value for data with less or no noise | 1 |
| --iter | fixed LCS # learning iterations param | 200000 |
| --N | fixed LCS rule population maximum size param | 2000 |
| --lcs-timeout | seconds until hyperparameter sweep stops for LCS algorithms | 1200 |
| --run-parallel | if run parallel on LSF compatible computing cluster | True |
| --queue | specify name of parallel computing queue (uses our research groups queue by default) | i2c2_normal |
| --res-mem | reserved memory for the job (in Gigabytes) | 4 |
| --max-mem | maximum memory before the job is automatically terminated | 15 |
| -c | Boolean: Specify whether to check for existence of all output files | Stores False |
| -r | Boolean: Rerun any jobs that did not complete (or failed) in an earlier run. | Stores False |

##### Example: Run only one ML modeling algorithm
By default STREAMLINE runs all ML modeling algorithms. If the user only wants to run one (or a small number) of these algorithms, they can run the following command first turning all algorithms off, then specifying the ones to activate. In this example we only run random forest. Other algorithms could be specified as True here to run them as well.
```
python ModelMain.py --out-path /myoutputpath --exp-name hcc_test --do-all False --do-RF True
```

##### Example: Utilize built-in algorithm feature importance estimates when available
By default STREAMLINE uniformly applies scikit-learn's permutation feature importance estimator to all ML modeling algorithms. However a number of algorithms offer built in strategies for feature importance estimation (that can differ for each algorithm). Naive Bayes, Support Vector Machines (for non-linear kernels), ANN, and k-NN do not have such built in estimates. By setting `--use-uniformFI` to 'False', any algorithm that can return internally determined feature importance estimates will be used instead of the default permutation feature importance estimate. Generally, to more consistently compare feature importance scores across algorithms, we recommend users apply the default permutation-based estimator uniformly across all algorithms. However the example below shows how to turn this off:
```
python ModelMain.py --out-path /myoutputpath --exp-name hcc_test --use-uniformFI True
```

##### Example: Specify an alternative primary evaluation metric
By default STREAMLINE uses balanced accuracy as it's primary evaluation metric for both hyperparameter optimization and permutation-based model feature importance evaluation. However any classification metrics defined by scikit-learn (see https://scikit-learn.org/stable/modules/model_evaluation.html) could be used instead.  We chose balanced accuracy as the default because it equally values accurate prediction of both 'positive' and 'negative' classes, accounting for class imbalance. In this example illustrate how a user could change this primary metric to the F1 score.
```
python ModelMain.py --out-path /myoutputpath --exp-name hcc_test --metric f1
```

##### Example: Reduce computational burden of algorithms that run slow in large instance spaces
By default STREAMLINE uses all available training instances to train each specified ML algorithm. However XGB, SVM, ANN, k-NN and GP, can run very slowly when the number of training instances is very large. To be able to run these algorithms in a reasonable amount of time this pipeline includes the option to specify a random (class-balance-preserved) subset of the training instances upon which to train. In this example we set this training sample to 2000. This will only be applied to the 5 aforementioned algorithms.  All others will still train on the entire training set.
```
python ModelMain.py --out-path /myoutputpath --exp-name hcc_test --subsample 2000
```

#### Phase 6: Statistics Summary
Run parameters for `StatsMain.py`:

| Argument | Description | Default Value |
|:-------- |:---------------------  | ----------- |
| --out-path | path to output directory | MANDATORY |
| --exp-name | name of experiment output folder (no spaces) | MANDATORY |
| --plot-ROC | Plot ROC curves individually for each algorithm including all CV results and averages | True |
| --plot-PRC | Plot PRC curves individually for each algorithm including all CV results and averages | True |
| --plot-box | Plot box plot summaries comparing algorithms for each metric | True |
| --plot-FI_box | Plot feature importance boxplots and histograms for each algorithm | True |
| --top-features| Number of top features to illustrate in figures | 20 |
| --model-viz| Directly visualize either DT or GP models if trained | True |
| --run-parallel | if run parallel on LSF compatible computing cluster | True |
| --queue | specify name of parallel computing queue (uses our research groups queue by default) | i2c2_normal |
| --res-mem | reserved memory for the job (in Gigabytes) | 4 |
| --max-mem | maximum memory before the job is automatically terminated | 15 |
| -c | Boolean: Specify whether to check for existence of all output files | Stores False |

#### Phase 7: [Optional] Compare Datasets
Run parameters for `DataCompareMain.py`:

| Argument | Description | Default Value |
|:-------- |:---------------------  | ----------- |
| --out-path | path to output directory | MANDATORY |
| --exp-name | name of experiment output folder (no spaces) | MANDATORY |
| --run-parallel | if run parallel on LSF compatible computing cluster | True |
| --queue | specify name of parallel computing queue (uses our research groups queue by default) | i2c2_normal |
| --res-mem | reserved memory for the job (in Gigabytes) | 4 |
| --max-mem | maximum memory before the job is automatically terminated | 15 |
| -c | Boolean: Specify whether to check for existence of all output files | Stores False |

#### Phase 8: [Optional] Generate PDF Training Summary Report
Run parameters for `PDF_ReportMain.py`:

| Argument | Description | Default Value |
|:-------- |:---------------------  | ----------- |
| --training | Indicate True or False for whether to generate pdf summary for pipeline training or followup application analysis to new dataset | True |
| --out-path | path to output directory | MANDATORY |
| --exp-name | name of experiment output folder (no spaces) | MANDATORY |
| --run-parallel | if run parallel on LSF compatible computing cluster | True |
| --queue | specify name of parallel computing queue (uses our research groups queue by default) | i2c2_normal |
| --res-mem | reserved memory for the job (in Gigabytes) | 4 |
| --max-mem | maximum memory before the job is automatically terminated | 15 |
| -c | Boolean: Specify whether to check for existence of all output files | Stores False |

#### Phase 9: [Optional] Apply Models to Replication Data
Run parameters for `ApplyModelMain.py`:

| Argument | Description | Default Value |
|:-------- |:---------------------  | ----------- |
| --out-path | path to output directory | MANDATORY |
| --exp-name | name of experiment output folder (no spaces) | MANDATORY |
| --rep-path | path to directory containing replication or hold-out testing datasets (must have at least all features with same labels as in original training dataset) | MANDATORY |
| --dataset | path to target original training dataset | MANDATORY |
| --export-fc | run and export feature correlation analysis (yields correlation heatmap) | True |
| --plot-ROC | Plot ROC curves individually for each algorithm including all CV results and averages | True |
| --plot-PRC | Plot PRC curves individually for each algorithm including all CV results and averages | True |
| --plot-box | Plot box plot summaries comparing algorithms for each metric | True |
| --match-label | applies if original training data included column with matched instance ids | None |
| --run-parallel | if run parallel on LSF compatible computing cluster | True |
| --queue | specify name of parallel computing queue (uses our research groups queue by default) | i2c2_normal |
| --res-mem | reserved memory for the job (in Gigabytes) | 4 |
| --max-mem | maximum memory before the job is automatically terminated | 15 |
| -c | Boolean: Specify whether to check for existence of all output files | Stores False |

#### Phase 10: [Optional] Generate PDF 'Apply Replication' Summary Report
Note that this phase uses the same script as phase 8 but requires specifying three additional parameters `--training`, `--rep-path`, and `--dataset` so that a report geared towards applying new data to pre-trained models is generated instead.
Run parameters for `PDF_ReportMain.py`:

| Argument | Description | Default Value |
|:-------- |:---------------------  | ----------- |
| --training | Indicate True or False for whether to generate pdf summary for pipeline training or followup application analysis to new dataset | True |
| --out-path | path to output directory | MANDATORY |
| --exp-name | name of experiment output folder (no spaces) | MANDATORY |
| --rep-path | path to directory containing replication or hold-out testing datasets (must have at least all features with same labels as in original training dataset) | MANDATORY |
| --dataset | path to target original training dataset | MANDATORY |
| --run-parallel | if run parallel on LSF compatible computing cluster | True |
| --queue | specify name of parallel computing queue (uses our research groups queue by default) | i2c2_normal |
| --res-mem | reserved memory for the job (in Gigabytes) | 4 |
| --max-mem | maximum memory before the job is automatically terminated | 15 |
| -c | Boolean: Specify whether to check for existence of all output files | Stores False |

#### Phase 11: [Optional] File Cleanup
Run parameters for `FileCleanup.py`:

| Argument | Description | Default Value |
|:-------- |:---------------------  | ----------- |
| --out-path | path to output directory | MANDATORY |
| --exp-name | name of experiment output folder (no spaces) | MANDATORY |
| --del-time | delete individual run-time files (but save summary) | True |
| --del-oldCV | path to target original training dataset | True |

***
# Other Guidelines for STREAMLINE Use
* SVM and ANN modeling should only be applied when data scaling is applied by the pipeline.
* Logistic Regression' baseline model feature importance estimation is determined by the exponential of the feature's coefficient. This should only be used if data scaling is applied by the pipeline.  Otherwise `use_uniform_FI` should be True.
* While the STREAMLINE includes `impute_data` as an option that can be turned off in `DataPreprocessing`, most algorithm implementations (all those standard in scikit-learn) cannot handle missing data values with the exception of eLCS, XCS, and ExSTraCS. In general, STREAMLINE is expected to fail with an errors if run on data with missing values, while `impute_data` is set to 'False'.

***
# Unique Characteristics of STREAMLINE
* Pipeline includes reliable default run parameters that can be adjusted for further customization.
* Easily compare ML performance between multiple target datasets (e.g. with different feature subsets)
* Easily conduct an exploratory analysis including: (1) basic dataset characteristics: data dimensions, feature stats, missing value counts, and class balance, (2) detection of categorical vs. quantiative features, (3) feature correlation (with heatmap), and (4) univariate analyses with Chi-Square (categorical features), or Mann-Whitney U-Test (quantitative features).
* Option to manually specify which features to treat as categorical vs. quantitative.
* Option to manually specify features in loaded dataset to ignore in analysis.
* Option to utilize 'matched' cross validation partitioning: Case/control pairs or groups that have been matched based on one or more covariates will be kept together within CV data partitions.
* Imputation is completed using mode imputation for categorical variables first, followed by MICE-based iterative imputation for quantitaive features. There is an option to use mean imputation for quantitative features when imputation computing cost is prohibitive in large datasets.
* Data scaling, imputation, and feature selection are all conducted within respective CV partitions to prevent data leakage (i.e. testing data is not seen for any aspect of learning until final model evaluation).
* The scaling, imputation, and feature selection data transformations (based only on the training data) are saved (i.e. 'pickled') so that they can be applied in the same way to testing partitions, and in the future to any replication data.
* Collective feature selection is used: Both mutual information (proficient at detectin univariate associations) and MultiSURF (a Relief-based algorithm proficient at detecting both univariate and epistatic interactions) are run, and features are only removed from consideration if both algorithms fail to detect an informative signal (i.e. score > 0). This ensures that interacting features that may have no univariate association with class are not removed from the data prior to modeling.
* Automatically outputs average feature importance bar-plots from feature importance/feature selection phase.
* Since MultiSURF scales linearly with # of features and quadratically with # of instances, there is an option to select a random instance subset for MultiSURF scoring to reduce computational burden.
* Includes 3 rule-based machine learning algorithms: ExSTraCS, XCS, and eLCS (to run optionally). These 'learning classifier systems' have been demonstrated to be able to detect complex associations while providing human interpretable models in the form of IF:THEN rule-sets. The ExSTraCS algorithm was developed by our research group to specifically handle the challenges of scalability, noise, and detection of epistasis and genetic heterogeneity in biomedical data mining.  
* Utilizes the 'optuna' package to conduct automated Bayesian hyperparameter optimization during modeling (and optionally outputs plots summarizing the sweep).
* We have sought to specify a comprehensive range of relevant hyperparameter options for all included ML algorithms.
* Some ML algorithms that have a build in strategy to gather model feature importance estimates (i.e. LR,DT,RF,XGB,LGB,GB,eLCS,XCS,ExSTraCS) These can be used in place of permutation feature importance estimates by setting the parameter `use_uniform_FI` to 'False'.
* All other algorithms rely on estimating feature importance using permutation feature importance.
* All models are evaluated, reporting 16 classification metrics: Accuracy, Balanced Accuracy, F1 Score, Sensitivity(Recall), Specificity, Precision (PPV), True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN), Negative Predictive Value (NPV), Likeliehood Ratio + (LR+), Likeliehood Ratio - (LR-), ROC AUC, PRC AUC, and PRC APS.
* All models are saved as 'pickle' files so that they can be loaded and reapplied in the future.
* Outputs ROC and PRC plots for each ML modeling algorithm displaying individual n-fold CV runs and average the average curve.
* Outputs boxplots for each classification metric comparing ML modeling performance (across n-fold CV).
* Outputs boxplots of feature importance estimation for each ML modeling algorithm (across n-fold CV).
* Outputs our proposed 'composite feature importance plots' to examine feature importance estimate consistency (or lack of consistency) across all ML models (i.e. all algorithms)
* Outputs summary ROC and PRC plots comparing average curves across all ML algorithms.
* Collects run-time information on each phase of the pipeline and for the training of each ML algorithm model.
* For each dataset, Kruskall-Wallis and subsequent pairwise Mann-Whitney U-Tests evaluates statistical significance of ML algorithm modeling performance differences for all metrics.
* The same statistical tests (Kruskall-Wallis and Mann-Whitney U-Test) are conducted comparing datasets using the best performing modeling algorithm (for a given metric and dataset).
* Outputs boxplots comparing performance of multiple datasets analyzed either across CV runs of a single algorithm and metric, or across average for each algorithm for a single metric.
* A formatted PDF report is automatically generated giving a snapshot of all key pipeline results.
* A script is included to apply all trained (and 'pickled') models to an external replication dataset to further evaluate model generalizability. This script (1) conducts an exploratory analysis of the new dataset, (2) uses the same scaling, imputation, and feature subsets determined from n-fold cv training, yielding 'n' versions of the replication dataset to be applied to the respective models, (3) applies and evaluates all models with these respective versions of the replication data, (4) outputs the same set of aforementioned boxplots, ROC, and PRC plots, and (5) automatically generates a new, formatted PDF report summarizing these applied results.
* A collection of `UsefulNotebooks` described in the next section offers even more functionality following the initial pipeline run.

***
# Do Even More with STREAMLINE
The STREAMLINE repository includes a folder called `UsefulNotebooks`.  This set of notebooks are designed to operate on a previously run STREAMLINE experiment output folder. They open this folder and generate additional materials outside of the main analysis pipeline.  We review these notebooks below:
1. `AccessingPickledMetricsAndPredictionProbabilities.ipynb`: Gives a demonstration of how users can easily access all pickled metrics from the analysis, and also generates new files within respective `model_evaluation` folders including case (i.e. class 1) probabilities for all testing data instances for each model.
2. `CompositeFeatureImportancePlots.ipynb`: Offers users the ability to regenerate composite feature importance plots to their own specifications for publication.
3. `GenerateModelFeatureImportanceHeatmap.ipynb`: Allows users to generate a web-based interactive heatmap illustrating model feature importance ranks across all ML algorithms.
4. `ModelDecisionThresholdTestEval.ipynb`: Allows users to recalculate all testing evaluation metrics for models using an alternative decision threshold.
5. `ModelDecisionThresholdTrainEval.ipynb`: Allows users to view and/or recalculate all training evaluation metrics for models using an alternative decision threshold.
6. `ModelDecisionThresholdView.ipynb`: Allows users to generate an interactive slider within Jupyter Notebook to examine the impact of different decision thresholds on a given model trained by the pipeline.
7. `ModelViz_DT_GP.ipynb`: Allows users to generate directly interpretable visualizations of any decision tree or genetic programming models trained by the pipeline.
8. `ReportingAppliedPredictionProbabilities.ipynb`: Allows users to recalculate all testing evaluation metrics for models using an alternative decision threshold (on the applied, rather than original 'target' datasets).
9. `ROC_PRC_Plots.ipynb`: Offers users the ability to regenerate ROC and PRC Plots to their own specifications for publication.

***
# Demonstration data
Included with this pipeline is a folder named `DemoData` including two small datasets used as a demonstration of pipeline efficacy. New users can easily run the included jupyter notebook 'as-is', and it will be run automatically on these datasets. The first dataset `hcc-data_example.csv` is the Hepatocellular Carcinoma (HCC) dataset taken from the UCI Machine Learning repository. It includes 165 instances, 49 fetaures, and a binary class label. It also includes a mix of categorical and numeric features, about 10% missing values, and class imbalance, i.e. 63 deceased (class = 1), and 102 surived (class 0).  To illustrate how STREAMLINE can be applied to more than one dataset at once, we created a second dataset from this HCC dataset called `hcc-data_example_no_covariates.csv`, which is the same as the first but we have removed two covariates, i.e. `Age at Diagnosis`, and `Gender`.

Furthermore, to demonstrate how STREAMLINE-trained models may be applied to new data in the future through the phase 9 `ApplyModel.py` we have simply added a copy of `hcc-data_example.csv`, renamed as `hcc-data_example_rep.csv` to the folder `DemoRepData`. While this is not a true replication dataset (as none was available for this example) it does illustrate the functionality of `ApplyModel`. Since the cross validation (CV)-trained models are being applied to all of the original target data, the `ApplyModel.py` results in this demonstration are predictably overfit.  When applying trained models to a true replication dataset model prediction performance is generally expected to be as good or less well performing than the individual testing evaluations completed for each CV model.

***
# Troubleshooting

## Rerunning a failed modeling job
If for some reason a `ModelJob.py` job fails, or must be stopped because it's taking much longer than expected, we have implemented a run parameter `-r` in `ModelMain.py` allowing the user to only rerun those failed/stopped jobs rather than the entire modeling phase. After using `-c` to confirm that some jobs have not completed, the user can instead use the `-r` command to search for missing jobs and rerun them. Note that a new random seed or a more limited hyperparameter range may be needed for a specific modeling algorithm to resolve job failures or overly long runs (see below).

## Unending modeling jobs
One known issue is that the Optuna hyperparameter optimization does not have a way to kill a specific hyperparameter trial during optimization.  The `timeout` option does not set a global time limit for hyperparameter optimization, i.e. it won't stop a trial in progress once it's started. The result is that if a specific hyperparameter combination takes a very long time to run, that job will run indefinitely despite going past the `timeout` setting. There are currently two recommended ways to address this.

1. Try to kill the given job(s) and use the `-r` command for `ModelMain.py`.  When using this command, a different random seed will automatically which can resolve the run completion, but will impact perfect reproducibility of the results.

2. Go into the code in `ModelJob.py` and limit the hyperparameter ranges specified (or do this directly in the jupyter notebook if running from there).  Specifically eliminate possible hyperparameter combinations that might lead the hyperparameter sweep to run for a very long time (i.e. way beyond the `timeout` parameter).

***
# Development notes
Have ideas on how to improve this pipeline? We welcome suggestions, contributions, and collaborations.

## History
STREAMLINE is based on our initial development repository https://github.com/UrbsLab/AutoMLPipe-BC. STREAMLINE's codebase and functionalities have been reorganized and extended, along with the name rebranding. This STREAMLINE repository will be developed further in the future while AutoMLPipe-BC will remain as is.

## Planned extensions/improvements

### Logistical extensions
* Improved modularization of code for adding new ML modeling algorithms
* Set up code to be run easily on cloud computing options such as AWS, Azure, or Google Cloud
* Set up option to use STREAMLINE within Docker
* Set up STREAMLINE parallelization to be able to automatically run with one command rather than require phases to be run in sequence (subsequent phases only being run when the prior one completes)

### Capabilities extensions
* Support multiclass and quantitative endpoints
    * Will require significant extensions to most phases of the pipeline including exploratory analysis, CV partitioning, feature importance/selection, modeling, statistics analysis, and
* Shapley value calculation and visualizations
* Create ensemble model from all trained models which can then be evaluated on hold out replication data
* Expand available model visualization opportunities for model interpretation
* Improve Catboost implementation:
    * Allow it to use internal feature importance estimates as an option
    * Give it the list of features to be treated as categorical
* New `UsefulNotebooks` providing even more post-run data visualizations and customizations
* Clearly identify which algorithms can be run with missing values present, when user does not wish to apply `impute_data` (not yet fully tested)

### Algorithmic extensions
* Addition of other ML modeling algorithm options
* Refinement of pre-configured ML algorithm hyperparameter options considered using Optuna
* Expanded feature importance estimation algorithm options and improved, more flexible feature selection strategy improving high-order feature interaction detection
* New rule-based machine learning algorithm (in development)

***
# Acknowledgements
STREAMLINE is the result of 3 years of on-and-off development gaining feedback from multiple biomedical research collaborators at the University of Pennsylvania, Fox Chase Cancer Center, Cedars Sinai Medical Center, and the University of Kansas Medical Center. The bulk of the coding was completed by Ryan Urbanowicz and Robert Zhang. Special thanks to Yuhan Cui, Pranshu Suri, Patryk Orzechowski, Trang Le, Sy Hwang, Richard Zhang, Wilson Zhang, and Pedro Ribeiro for their code contributions and feedback.  We also thank the following collaborators for their feedback on application of the pipeline during development: Shannon Lynch, Rachael Stolzenberg-Solomon, Ulysses Magalang, Allan Pack, Brendan Keenan, Danielle Mowery, Jason Moore, and Diego Mazzotti.

***
# Citation
If you wish to cite this work prior to our first journal publication, please use:
```
@misc{streamline2022,
  author = {Urbanowicz, Ryan and Zhang, Robert},
  title = {STREAMLINE: A Simple, Transparent, End-To-End Automated Machine Learning Pipeline},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/UrbsLab/STREAMLINE/} }
}
```
