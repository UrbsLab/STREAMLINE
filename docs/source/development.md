# Development Notes
This section summarizes the past, present, and future development of STREAMLINE.

***
## Release History

### Current Release - Beta 0.3.0 (August 2023)
The current version of STREAMLINE is based on our initial STREAMLINE project release Beta 0.2.5, and has since undergone a major refactoring
STREAMLINE's codebase. Many functionalities have been reorganized and extended. 

#### Major Updates
* Extended to be able to run in parallel on 7 different types of HPC clusters using `dask_jobqueue` as documented [here](https://jobqueue.dask.org/en/latest/api.html)
* Extended Phase 1 (previously EDA), to included numerical data encoding, automated data cleaning, feature engineering, and a second round of EDA:
    * Added numerical encoding for any binary, text-valued features, with a map file `Numerical_Encoding_Map.csv` output to document this numerical mapping of original text-values
    * Added [quantitative_feature_path](parameters.md#quantitative-feature_path) parameter in addition to [categorical_feature_path](parameters.md#categorical-feature_path) allowing users to indicate which features to treat as categorical vs. quantitative (or specify one list and all other features will be treated as the other type). New `.csv` output files are also generated to identify what features were treated as one feature type or the other after data processing.
    * Added automated feature engineering of 'missingness' features to evaluate missingness as being predictive (assuming [MNAR](https://en.wikipedia.org/wiki/Missing_data)) along with [featureeng_missingness](parameters.md#featureeng-missingness) parameter to control this function. `Missingness_Engineered_Features.csv` is output to document what features were added to the processed dataset as a result.
    * Added automated cleaning of features with high 'missingness'; with [cleaning_missingness](parameters.md#cleaning-missingness) parameter added to control this function. `Missingness_Feature_Cleaning.csv` is output to document what features were removed from the processed dataset as a result.
    * Added automated cleaning of instances with high 'missingness'; with [cleaning_missingness](parameters.md#cleaning-missingness) parameter added to control this function. 
    * Added automated one-hot-encoding of all numerical and text-valued categorical features (with 3 or more values) so that they will be treated as such throughout all STREAMLINE phases.
    * Added automated cleaning of highly correlated features (one feature randomly removed out of a highly correlated feature pair); with [correlation_removal_threshold](#correlation-removal-threshold) parameter added to control this function. `correlation_feature_cleaning.csv` is output to document what features were removed in this way.
    * Added `DataProcessSummary.csv` output file to document changes in feature, feature type, instance, class, and missing value counts during each new cleaning/engineering step.
    * Added a secondary EDA applied to the processed dataset, saved with separate output files to the 'initial' EDA.
* Adapted the 'replication' phase of STREAMLINE to process the replication data in the same way as the initial 'target dataset' ensuring that the same features are present. This accounts for any new 'as-of-yet' unseen values for categorical features that had previously been one-hot-encoded.
* Added ability to run the whole pipeline as a single command in the different command line run modes (i.e. from the command line locally or on an HPC). This includes the addition of a variety of new command-line specific run parameters.
* Added support for running STREAMLINE from the command line using a configuration file (in addition to commandline parameters)
* Modularize all ML modeling algorithms within classes, which adds the ability for users to (relatively easily) add other scikit-learn-compatible classification modeling algorithms to the STREAMLINE code-base by making a python file in `streamine/models/` based on the base model template. This allows code-savy users to easily add other algorithms we have not yet included, including their own.
* As a demonstration of the ability to add new ML algorithms in this way, we've added Elastic Net (EN) as the 16th ML algorithm included within STREAMLINE.
* Extended Google Colab Notebook to (1) automatically download the latest version of STREAMLINE, (2) offer separate 'Easy' and 'Manual' run modes for users to apply the notebook to their own data, where 'Easy' mode uses a prompt to gather essential run parameter information including a file navigation window to select the target dataset folder, (3) automatically download the output experiment folder and open the PDF summary reports on their screen (with user permission).

#### Minor Updates
* Reverted back to using mean (rather than median) to present and sort model feature importances in plots (which was changed in Beta 0.2.4). This is to prevent confusion when running the notebook demos on the [demonstration datasets](data.md#demonstration-data), where using 3-fold CV yields median = 0 for all decision tree model feature importance scores which confuses picking and sorting the top features for plotting, as well as eliminates decision trees from the composite feature importance plots. We have added a hard-coded option to revert back to median ranking within the `fi_stats()` function within `statistics.py`.
* Updated repository folder hierarchy, filenames, and some outputfile names.
* Updated STREAMLINE phase groupings/numberings.
* Updated the STREAMLINE schematic figure to reflect all major changes and new phase grouping.
* Updated the feature correlation heatmap outputs: (1) color scheme used (for clarity), (2) view the non-redundant triangle vs. the full square (3) scale the feature names to avoid overlap, and don't show names at all when there are a large number of features (such that names would be unreadable)
* Feature correlation results are now also documented within `FeatureCorrelations.csv`.
* Reformatted the PDF output summary files to (1) add and re-organize all run parameters on the first page, (2) indicate the STREAMLINE version on the bottom of the page, and (3) include the new data processing/counts summary.
* Univariate analysis output files now include the test run and test score in addition to p-values.
* Updated the STREAMLINE Jupyter Notebook and other 'Useful Notebooks' to function with this new code framework.
* Created a new `hcc_data_custom.csv` dataset for the demo that adds simulated features and instances to `hcc_data.csv` to explicitly test (and demonstrate the functionality of) the new automatic data cleaning and engineering steps in STREAMLINE phase 1. Similarly created a replication dataset `hcc_data_custom_rep.csv` which adds some noise to `hcc_data_custom.csv` and some other custom additions to demonstrate replication functionality. The code to generate these 'custom' datasets from `hcc_data.csv` are included in the `data` folder as the notebook `Generate_expanded_HCC_Dataset`. 

***
### Beta 0.2.5 (June 24, 2022)
* Added a minor additional catch to prevent statistical comparison results failure under specific situations. (in StatsJob.py and DataCompareJob.py)
* Cleaned up commented out old code

### Beta 0.2.4 (June 15, 2022)
* Fix - Special case when running data with no missing data and imputation was 'True', apply model error when looking for non existent imputation file. Code fixed so that importing imputed file is in try/except loop to prevent fail. Also updated apply model so that both .csv and .txt replication data can be loaded.
* At recommendation of collaborator, switched from mean to median scores for feature importance figures. Also now outputs median algorithm performance summary, and adds median performance to pdf summary. Also now present median values in statistical significance output since this pairs more appropriately with non-parametric statistics than mean and standard deviation.

### Beta 0.2.3 (May 19, 2022)
* Added fixes for (and confirmed functionality of) code to run STREAMLINE serially via the command line (in Linux - does not support Windows command line use).
* This release is considered stable and fully functional based on all tests and user feedback since the alpha release. We will make additional updates as needed for any other reported special case bugs/issues, as well as expand STREAMLINE further in future releases.

### Beta 0.2.2 (May 19, 2022)
This latest Beta update addresses key functionality issues for running STREAMLINE serially from the command line, as well as a number of other minor functionality fixes and improvements.

* Composite FI no longer fails when one algorithm used
* Composite FI plots now support weighting with both balanced accuracy and roc_auc
* Fixed major issues preventing running certain phases of STREAMLINE serially from command line
* Removed 'None' option for max features in feature selection
* Fixed pdf summary page 1 formatting issue
* Updated Optuna optimization for LR to avoid invalid hyperparameter combinations
* Enforced use of Optuna 2.0.0 for generating hyperparameter optimization figure generation, and added try catches to all algorithms so that STREAMLINE does not completely fail when there are lingering issues with Optuna versions in generating these figures.
* Updated notebooks accordingly

### Beta 0.2.1 (May 17, 2022)
* Moved codebase into 'streamline' folder and updated code accordingly
* Updated default run parameters for Optuna
* Identified that STREAMLINE does not guarantee complete replicability (due to Optuna) when parallelized.
* Ensured replicability of cv data following scaling by rounding scaled data to 7 decimal places to avoid float rounding errors beyond the control of random seed fixing.

### Beta 0.2.0 (May 14, 2022)
* After initial alpha testing with colleagues using different platforms and anaconda installations this first beta release of STREAMLINE has been demonstrated functional in all configurations tested. STREAMLINE is ready for external use, but it is still possible that there may be unforeseen issues run using configurations outside of those explicitly tested. Please let us know if you run into issues, noting the run mode, Anaconda version, and errors you are encountering so we can address such issues.

* We plan to continue to expand and improve STREAMLINE, so we recommend users keep an eye out for new releases in the upcoming months, and update to the newest release whenever it's available. After much testing and application this software is believed to be ready for general use. If investigators apply this pipeline to research submitted for publication we ask that they check this repository for the newest STREAMLINE citation reference. We welcome investigators to reach out for assistance in using and interpreting STREAMLINE output in their research. We hope this tool will lead to many new collaborations and opportunities to publish new research.

### Alpha 0.1.3 (May 12, 2022)
* Updated Readme installation instructions and default setting for model feature importance estimation. Code has been tested on the most recent Linux version of Anaconda.

### Alpha 0.1.2 (May 12, 2022)
* Fix for Anaconda version issue regarding scipy in exploratory analysis.

### Alpha 0.1.1 (May 12, 2022)
* Addressed upcoming scipy depreciation warning by replacing scipy.interp() with numpy.interp().

### Alpha Release (May 12, 2022)
* The first stable, bug-tested implementation of STREAMLINE. The bulk of the underlying code is inherited from AutoMLPipe-BC. This version has been demonstrated to operate properly only under the specific version of Anaconda, and the specified versions of other installed packages. It has not yet been tested on any MAC devices, only Windows and Linux.

***
## Planned Improvements

### Known issues
* Repair probable bugs in eLCS and XCS ML modeling algorithms (outside of STREAMLINE). Currently, we have intentionally set both to 'False' by default, so they will not run unless user explicitly turns them on.
* Set up STREAMLINE to be able to run (as an option) through all phases even if some CV model training runs have failed (as an option).
* Optuna currently prevents a guarantee of reproducibility of STREAMLINE when run in parallel, unless the user specifies `None` for the `timeout` parameter. This is explained in the Optuna documentation as an inherent result of running Optuna in parallel, since it is possible for a different optimal configuration to be found if a greater number of optimization trials are completed from one run to the next. We will consider alternative strategies for running STREAMLINE hyperparameter optimization as options in the future.
* Optuna generated visualization of hyper-parameter sweep results fails to operate correctly under certain situations (i.e. for GP most often, and for LR when using a version of Optuna other than 2.0.0)  It looks like Optuna developers intend to fix these issues in the future, and we will update STREAMLINE accordingly when they do.

### Logistical extensions
* Set up code to be run easily on cloud computing options such as AWS, Azure, or Google Cloud.
* Set up option to use STREAMLINE within Docker.

### Capabilities extensions
* Support multiclass and quantitative endpoints (in [development branch](https://github.com/STREAMLINE/tree/dev)): 
    * Requires significant extensions to most phases of the pipeline including exploratory analysis, CV partitioning, feature importance/selection, modeling, statistics analysis, and visualizations
* Shapley value calculation and visualizations
* Create ensemble model from all trained models which can then be evaluated on hold out replication data
* Expand available model visualization opportunities for model interpretation (i.e. Logistic Regression)
* Improve Catboost integration:
    * Allow it to use internal feature importance estimates as an option
    * Give it the list of features to be treated as categorical
* New code providing even more post-run data visualizations and customizations
* Clearly identify which algorithms can be run with missing values present, when user does not wish to apply [`impute_data`](parameters.md#impute-data) (not yet fully tested)
* Create a smarter approach to hyper-parameter optimization: (1) avoid hyperparameter combinations that are invalid (i.e. as seen when using Logistic Regression), (2) intelligently exclude key hyperparameters known to improve overall performance as they get larger, and apply a user defined value for these in the final model training after all other hyperparameters have been optimized (i.e. evolutionary algorithms such as genetic programming and ExSTraCS almost always benefit from larger population sizes and learning cycles. Given that we know these parameters improve performance, including them in hyperparameter optimization only slows down the process with little informational gain)

### Algorithmic extensions
* Refinement of pre-configured ML algorithm hyperparameter options considered using Optuna
* Expanded feature importance estimation algorithm options and improved, more flexible feature selection strategy improving high-order feature interaction detection
* New rule-based machine learning algorithm (in development)

