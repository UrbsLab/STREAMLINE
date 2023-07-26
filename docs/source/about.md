#  About (FAQs)

## Can I run STREAMLINE as is?
Yes, as an automated machine learning pipeline, users can easily run the pipeline in it's entirety or one phase at a time. We have set up STREAMLINE to include
reasonably reliable default pipeline run parameters that users can optionally change to suite their needs. However the overall pipeline has been designed to operated 
in a specific order utilizing a fixed set of data science elements/steps to ensure consistency and adherence to best practices.


## What can STREAMLINE be used for?
STREAMLINE can be used as:
1. A tool to quickly run a rigorous ML data analysis over one or more datasets using one or more of the included modeling algorithms
2. A framework to compare established scikit-learn compatible ML modeling algorithms to each other or to new algorithms
3. A baseline standard of comparison (i.e. positive control) with which to evaluate other AutoML tools that seek to optimize ML pipeline assembly as part of their methodology
4. A framework to quickly run exploratory analysis, data processing, and/or feature importance estimation/feature selection prior to using some other methodology for ML modeling
5. An educational example of how to integrate some of the many amazing Python-based data science tools currently available (in particular pandas, scipy, optuna, and scikit-learn).
6. A framework from which to create a new, expanded, adapted, or modified ML analysis pipeline
7. A framework to add and evaluate new modeling algorithms (see 'Adding New Modeling Algorithms')

***
## What level of computing skill is required for use?
STREAMLINE offers a variety of use options making it accessible to those with little or no coding experience as well as the seasoned programmer/data scientist. While there is currently no graphical user interface (GUI),
the most naive user needs only know how to navigate their PC file system, specify folder/file paths,
and have a Google Drive account (to run STREAMLINE serially on Google Colab).

Those with a very basic knowledge of python and computer environments can apply
STREAMLINE locally and serially using the included jupyter notebook.

Those comfortable with command lines can run STREAMLINE locally or on a computing cluster via the command line.

***
## How is STREAMLINE different from other AutoML tools?
Unlike most other AutoML tools, STREAMLINE was designed as an end-to-end framework to rigorously apply
and compare a variety of ML modeling algorithms and collectively learn from them as opposed
to only identifying a best performing model and/or attempting to optimize the analysis pipeline
configuration itself. STREAMLINE adopts a fixed series of purposefully selected steps/phases
in line with data science best practices. It seeks to automate all domain generalizable
elements of an ML analysis pipeline with a specific focus on biomedical data mining challenges.
This tool can be run or utilized in a number of ways to suite a variety experience levels and
levels of problem/data complexity.

***
## What does STREAMLINE automate?
Currently, STREAMLINE automates the following aspects of a machine learning analysis pipeline (see [schematic](index.rst)):
   1. Exploratory analysis (on the initial and processed dataset)
   2. Data processing 
      * Basic data cleaning: instances with no outcome, features users want to ignore, features and instances with high missingness (i.e. # of missing values), and one of each pair of highly correlated features.
      * Basic feature engineering: add/encode missingness features and apply one-hot encoding to categorical features.
      * CV partitioning
      * Missing value imputation
      * Scaling of features (using standard scalar)
   4. Feature processing
      * Filter-based feature importance estimation (Mutual information & MultiSURF)
      * Feature selection (using a 'collective', i.e. multi-algorithm approach)
   4. Modeling with 'Optuna' hyperparameter optimization using the 16 implemented ML algorithms (see below)
   5. Evaluation of all modeles on respective testing datasets using 16 classification metrics and model feature importance estimation
   6. Generates and organizes all results and other outputs including:
      * Tables (Model evaluations, runtimes, selected hyperparameters, etc.)
      * Publication-ready plots/figures & model visualizations
      * Trained models (stored as pickled objects for re-use)
      * Training and testing CV datasets (for external reproducibility)
   7. Non-parametric statistical comparisons across ML algorithms and analyzed datasets
   8. Summary report generation (as pre-formatted PDF) including:
      * STREAMLINE run settings used
      * Dataset characteristics summary
      * Key figures and model evaluation results averaged over CV runs
      * Runtime summary
   9. Applying and evaluating all STREAMLINE-trained models on further/future
      hold out replication data

The following 16 scikit-learn compatible ML modeling algorithms are currently included as options:
1. Naive Bayes (NB)
2. Logistic Regression (LR)
3. Elastic Net (EN)
4. Decision Tree (DT)
5. Random Forest (RF)
6. Gradient Boosting (GB)
7. XGBoost (XGB)
8. LGBoost (LGB)
9. CatBoost (CGB)
10. Support Vector Machine (SVM)
11. Artificial Neural Network (ANN)
12. K-Nearest Neighbors (k-NN)
13. Genetic Programming (GP)
14. Educational Learning Classifier System (eLCS)
15. 'X' Classifier System (XCS)
16. Extended Supervised Tracking and Classifying System (ExSTraCS).

Classification-relevant hyperparameter values and ranges have been carefully
selected for each algorithm and have been pre-specified for the automated (Optuna-driven)
automated hyperparameter sweep. Thus, the user does not need to specify any algorithm hyperparameters or value options.

The automatically formatted PDF reports generated by STREAMLINE are intended
to give a brief summary of pipeline settings and key results.
An 'experiment folder' is also output containing all results, statistical analyses publication-ready plots/figures,
models, and other outputs is also saved allowing users to carefully examine all aspects of
analysis performance in a transparent manner.  

Notably, STREAMLINE does NOT automate the following elements, as they are still best
completed by human experts: (1) accounting for bias or fairness in data
collection, (2) feature engineering and data cleaning that requires domain knowledge.
We recommend users consider conducting these items, as needed, prior to applying STREAMLINE.

***
## Can I do more with the STREAMLINE output after it completes?
Yes, we have assempled a variety of 'useful' Jupyter Notebooks
designed to operate on an experiment folder allowing users to do even more
with the pipeline output. Examples include:
1. Accessing prediction probabilities.
2. Regenerating figures to user-specifications.
3. Trying out the effect of different prediction thresholds on selected
   models with an interactive slider.
4. Re-evaluating models when applying a new prediction threshold.
5. Generating an interactive model feature importance ranking visualization across
   all ML algorithms.
6. Generating an interpretable model vizualization for either decision tree or genetic programming models.

***
## How does STREAMLINE avoid data leakage?
Assembling a machine learning pipeline unfortunately affords a user many opportunities to incorrectly allow data leakage. 
Data leakage is when information that wouldn't normally be available or that comes from outside the training dataset is used to create the model. 

First, STREAMLINE makes it easy for a user to exclude features from a dataset that may contribute to data leakage (e.g. a feature that would not 
be available when applying the model to make predictions). A user can specify features to be excluded from modeling using the `ignore_features_path` parameter.

Second, STREAMLINE's pipeline is set up to specifically avoid learning any information that might eventually be a part of a testing data partition. Following CV partitioning, all  learning required to conduct imputation, scaling, feature importance evaluation, feature selection, and modeling is done using the respective training partition alone. For imputation, the same trained imputation strategy is applied to the respective testing data. For scaling, the same trained scalar is applied to the respective testing data. For feature selection, the same features removed from the training data are removed from the testing data. And for modeling, the testing data is only used for model evaluation.

This same strategy is applied to replication data later in the pipeline. When evaluating a given trained model on replication data, that dataset is imputed and scaled in the same way as the original training dataset for that model. And further, the same features that were removed during feature selection on the training data are removed for that replication dataset evaluation.

***
## Is STREAMLINE reproducible?
Yes, STREAMLINE is completely reproducible when the `timeout` parameter is set to `None`,
ensuring training of the same models with the same performance whenever the same datasets,
pipeline settings, and random seed are used.

When `timeout` is not set to `None`, STREAMLINE output can sometimes vary slightly (particularly when parallelized)
since Optuna (for hyperparameter optimization) may not complete the same
number of optimization trials within the user specified time limit on different
computing resources. 

However, having a `timeout` value specified helps ensure STREAMLINE run completion
within a reasonable time frame.

***
## Which STREAMLINE run mode should I use?
This multi-phase pipeline has been set up to run in one of four ways:

1. On Google Cloud as a Google Colab Notebook [Anyone can run]:
    * Advantages
      * No coding or PC environment experience needed
      * Automatically installs and uses the most recent version of STREAMLINE
      * Computing can performed directly on Google Cloud from anywhere
      * One-click run of whole pipeline (all phases)
      * Offers in-notebook viewing of results and ability to save notebook as documentation of analysis
      * Allows easy customizability of nearly all aspects of the pipeline with minimal coding/environment experience
    * Disadvantages:
      * Can only run pipeline serially
      * Slowest of the run options
      * Limited by google cloud computing allowances (may only work for smaller datasets)
    * Notes: Requires a Google account (free)

2. Locally as a Jupyter Notebook [Basic experience]:
    * Advantages:
      * Does not rely on free computing limitations of Google Cloud (but rather your own computer's limitations)
      * One-click run of whole pipeline (all phases)
      * Offers in-notebook viewing of results and ability to save notebook as documentation of analysis
      * Allows easy customizability of all aspects of the pipeline with minimal coding/environment experience (including hyperparameter value ranges)
    * Disadvantages:
      * Can only run pipeline serially
      * Slower runtime than from command-line
      * Beginners have to set up their computing environment
    * Notes: Requires Anaconda3, Python3, and several other minor Python package installations

3. Locally from the command line [Command-line Users]:
    * Advantages:
      * Typically runs faster than within Jupyter Notebook
      * A more versatile option for those with command-line experience
      * One-command run of whole pipeline available when using a configuration file to run
      * Can optionally run the pipeline one phase at a time
    * Disadvantages:
      * Can only run pipeline serially or with limited local cpu core parallelization
      * Command-line experience recommended
    * Notes: Requires Anaconda3, Python3, and several other minor Python package installations

4. On HPC Clusters from command line [Computing Cluster Users]:
    * Advantages:
      * By far the fastest, most efficient way to run STREAMLINE
      * Offers ability to run STREAMLINE over 7 types of HPC systems
      * One-command run of whole pipeline available when using a configuration file to run
      * Can optionally run the pipeline one phase at a time
    * Disadvantages:
      * Experience with command-line and dask-compatible clusters recommended
      * Access to a computing cluster required
    * Notes: Requires Anaconda3, Python3, and several other minor Python package installations. Cluster runs of STREAMLINE were set up using `dask-jobqueue` and thus should support 7 types of clusters as described in the [dask documentation](https://jobqueue.dask.org/en/latest/api.html). Currently we have only directly tested STREAMLINE on SLURM and LSF clusters. Further codebase adaptation may be needed for clusters types not on the above link.

