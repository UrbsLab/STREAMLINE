# Development notes 
Have ideas on how to improve this pipeline? We welcome suggestions, contributions, and collaborations.

## History
STREAMLINE is based on our initial development repository https://github.com/UrbsLab/STREAMLINE. STREAMLINE's codebase and functionalities have been reorganized and extended, along with the name rebranding. This STREAMLINE repository will be developed further in the future while AutoMLPipe-BC will remain as is.

## Planned extensions/improvements

### Known issues
* Repair probable bugs in eLCS and XCS ML modeling algorithms (outside of STREAMLINE). Currently, we have intentionally set both to 'False' by default, so they will not run unless user explicitly turns them on)
* Set up STREAMLINE to be able to run (as an option) through all phases even if some CV model training runs have failed (as an option)
* Optuna currently prevents a guarantee of reproducibility of STREAMLINE when run in parallel, unless the user specifies `None` for the `timeout` parameter. This is explained in the Optuna documentation as an inherent result of running Optuna in parallel, since it is possible for a different optimal configuration to be found if a greater number of optimization trials are completed from one run to the next. We will consider alternative strategies for running STREAMLINE hyperparameter optimization as options in the future.
* Optuna generated visualization of hyperparameter sweep results fails to operate correctly under certain situations (i.e. for GP most often, and for LR when using a version of Optuna other than 2.0.0)  It looks like Optuna developers intend to fix these issues in the future, and we will update STREAMLINE accordingly when they do.

### Logistical extensions
* Simplify documentation using github.io :heavy_check_mark:
* Improved modularization of code for adding new ML modeling algorithms :heavy_check_mark: - Planned as a pypi package
* Set up code to be run easily on cloud computing options such as AWS, Azure, or Google Cloud
* Set up option to use STREAMLINE within Docker - In Progress/TODO
* Set up STREAMLINE parallelization to be able to automatically run with one command rather than require phases to be run in sequence (subsequent phases only being run when the prior one completes) :heavy_check_mark: - In Progress

### Capabilities extensions
* Support multiclass and quantitative endpoints
    * Will require significant extensions to most phases of the pipeline including exploratory analysis, CV partitioning, feature importance/selection, modeling, statistics analysis, and visualizations
* Shapley value calculation and visualizations
* Create ensemble model from all trained models which can then be evaluated on hold out replication data
* Expand available model visualization opportunities for model interpretation (i.e. Logistic Regression)
* Improve Catboost integration:
    * Allow it to use internal feature importance estimates as an option
    * Give it the list of features to be treated as categorical
* New code providing even more post-run data visualizations and customizations
* Clearly identify which algorithms can be run with missing values present, when user does not wish to apply `impute_data` (not yet fully tested)
* Create a smarter approach to hyperparameter optimization: (1) avoid hyperparameter combinations that are invalid (i.e. as seen when using Logistic Regression), (2) intelligently exclude key hyperparameters known to improve overall performance as they get larger, and apply a user defined value for these in the final model training after all other hyperparameters have been optimized (i.e. evolutionary algorithms such as genetic programming and ExSTraCS almost always benefit from larger population sizes and learning cycles. Given that we know these parameters improve performance, including them in hyperparameter optimization only slows down the process with little informational gain)

### Algorithmic extensions
* Addition of other ML modeling algorithm options
* Refinement of pre-configured ML algorithm hyperparameter options considered using Optuna
* Expanded feature importance estimation algorithm options and improved, more flexible feature selection strategy improving high-order feature interaction detection
* New rule-based machine learning algorithm (in development)
