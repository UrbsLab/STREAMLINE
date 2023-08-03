#  About (FAQs)

***
## Can I run STREAMLINE as is?
Yes, as an automated machine learning pipeline, users can easily run the pipeline in it's entirety or one phase at a time. We have set up STREAMLINE to include
reasonably reliable default pipeline run parameters that users can optionally change to suite their needs. However the overall pipeline has been designed to operated 
in a specific order utilizing a fixed set of data science elements/steps to ensure consistency and adherence to best practices.

***
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
STREAMLINE offers a variety of use options making it accessible to those with little or no coding experience as well as the seasoned programmer/data scientist. While there is currently no graphical user interface (GUI), the most naive user needs only know how to navigate their computer file system, specify folder/file paths, and have a Google Drive account (to run STREAMLINE serially on Google Colab).

Those with a very basic knowledge of python and computer environments can apply STREAMLINE locally/serially using the included jupyter notebook.

Those comfortable with command lines should run STREAMLINE locally (either serially or with CPU core parallellization) or (if available) on a computing cluster (HPC) in parallel.

***
## How is STREAMLINE different from other AutoML tools?
Unlike most other AutoML tools, STREAMLINE was designed as an end-to-end framework to rigorously apply
and compare a variety of ML modeling algorithms and collectively learn from them as opposed
to only identifying a best performing model and/or attempting to optimize the analysis pipeline
configuration itself. STREAMLINE adopts a fixed series of purposefully selected steps/phases
in line with data science best practices. It seeks to automate all domain generalizable
elements of an ML analysis pipeline with a specific focus on biomedical data mining challenges.
This tool can be run or utilized in a number of ways to suite a variety experience levels and
levels of problem/data complexity. Furthermore, STREAMLINE is currently the only autoML pipeline tool that includes [learning classifier system (LCS)](https://www.youtube.com/watch?v=CRge_cZ2cJc) rule-based ML modeling algorithms for interpretable modeling in data with complex associations. This includes an LCS algorithm developed by our lab ([ExSTraCS](https://github.com/UrbsLab/scikit-ExSTraCS)), that has been specifically implemented to address the challenges of biomedical data analysis.

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
## Does STREAMLINE always run the entire pipeline?
Not necessarily. By default the entire pipeline will run, with the exception of dataset comparison (Phase 7), or replication analysis (Phase 8) when multiple target datasets, or replication data are not available. However the user can also choose to run STREAMLINE one phase at a time, which can often be advantageous. 

For example a user could just run Phase 1 to conduct an exploratory analysis of new data. Or they could just run Phases 1-4 to generate processed, training and testing datasets to apply to modeling outside of STREAMLINE. 

One caveate is that STREAMLINE Phases are designed to run in sequence (one after the other). 

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
Yes, STREAMLINE is completely reproducible when the `timeout` parameter is set to `None`, and. This also assumes that STREAMLINE is being run on the same datasets, with the same run parameters (including `random_seed`). However, STREAMLINE is expected to take longer to run when `timeout = None`.

***
## Which STREAMLINE run mode should I use?
STREAMLINE has been set up with multiple 'run-mode' options to suite different needs, computational resources, and user skill levels.
1.  **Google Colab Notebook:** Can easily be run by anyone, even those with no coding experience. STREAMLINE output can easily be viewed within the notebook as it runs. However this mode is computationally limited by the free Google Cloud resources it has access to. This mode is best for demonstration, educational purposes, and running STREAMLINE on small datasets, or applying a limited number of machine learning modeling algorithms.
2. **Jupyter Notebook:** The advantages are mostly the same as the Colab Notebook, however this mode relies on the computing resources of your local computer, which may (or possibly not) have a faster CPU and memory. However, to use this mode you will need to know how to set up your computing environment with Anaconda, etc, which can take some troubleshooting for a beginner. This mode is best for those who want a little more control over STREAMLINE, but still wish to run it within a notebook. 
3. **Command Line (Local):** As with Jupyter Notebook, this mode relies on the computing resources of your local computer. This mode is best for those who don't care about seeing output within the notebook and who know (or are willing to learn) how to work from a command line, but who may not have access to a computing cluster. 
4. **Command Line (HPC Cluster):** STREAMLINE is an embarrassingly parallel package, that can parallelize individual phases as HPC jobs at the level of target datasets, CV partitions, and algorithms. This mode is best if you have access to a dask-compatible computing cluster. It is the fastest most efficient way to run STREAMLINE, particularly on larger datasets, or when users want to run all pipeline algorithms and elements. 

For more details on the advantages and disadvantages of different run modes, see '[Picking a Run Mode](running.md#picking-a-run-mode)'.
