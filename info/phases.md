### STREAMLINE Phases Described
The base code for STREAMLINE (located in the `streamline` folder) is organized into a series of script phases designed to best optimize the parallelization of a given analysis. 
These loosely correspond with the pipeline schematic above. These phases are designed to be run in order. 
Phases 1-6 make up the core automated pipeline, with Phase 7 and beyond being run optionally based on user needs.

* Phase 1: Exploratory Analysis
  * Conducts an initial exploratory analysis of all target datasets to be analyzed and compared
  * Conducts basic data cleaning
  * Conducts k-fold cross validation (CV) partitioning to generate k training and k testing datasets
  * \[Runtime]: Typically fast, with the exception of generating feature correlation heatmaps in datasets with a large number of features

* Phase 2: Data Preprocessing
  * Conducts feature transformations (i.e. data scaling) on all CV training datasets individually
  * Conducts imputation of missing data values (missing data is not allowed by most scikit-learn modeling packages) on all CV training datasets individually
  * Generates updated training and testing CV datasets
  * \[Runtime]: Typically fast, with the exception of imputing larger datasets with many missing values

* Phase 3: Feature Importance Evaluation
  * Conducts feature importance estimations on all CV training datasets individually
  * Generates updated training and testing CV datasets
  * \[Runtime]: Typically reasonably fast, takes more time to run MultiSURF as the number of training instances approaches the default for 'instance_subset', or this parameter set higher in larger datasets

* Phase 4: Feature Selection
  * Applies 'collective' feature selection within all CV training datasets individually
  * Features removed from a given training dataset are also removed from corresponding testing dataset
  * Generates updated training and testing CV datasets
  * [Runtime]: Fast

* Phase 5: Machine Learning Modeling
  * Conducts hyperparameter sweep for all ML modeling algorithms individually on all CV training datasets
  * Conducts 'final' modeling for all ML algorithms individually on all CV training datasets using 'optimal' hyperparameters found in previous step
  * Calculates and saves all evaluation metrics for all 'final' models
  * \[Runtime]: Slowest phase, can be sped up by reducing the set of ML methods selected to run, or deactivating ML methods that run slowly on large datasets

* Phase 6: Statistics Summary
  * Combines all results to generate summary statistics files, generate results plots, and conduct non-parametric statistical significance analyses comparing ML model performance across CV runs
  * \[Runtime]: Moderately fast

* Phase 7: [Optional] Compare Datasets
  * NOTE: Only can be run if the STREAMLINE was run on more than dataset
  * Conducts non-parametric statistical significance analyses comparing separate original 'target' datasets analyzed by pipeline
  * \[Runtime]: Fast

* Phase 8: [Optional] Generate PDF Training Summary Report
  * Generates a pre-formatted PDF including all pipeline run parameters, basic dataset information, and key exploratory analyses, ML modeling results, statistical comparisons, and runtime.
  * \[Runtime]: Moderately fast

* Phase 9: [Optional] Apply Models to Replication Data
  * Applies all previously trained models for a single 'target' dataset to one or more new 'replication' dataset(s) that has the same features from an original 'target' dataset
  * Conducts exploratory analysis on new 'replication' dataset(s)
  * Applies scaling, imputation, and feature selection (unique to each CV partition from model training) to new 'replication' dataset(s) in preparation for model application
  * Evaluates performance of all models the prepared 'replication' dataset(s)
  * Generates summary statistics files, results plots, and conducts non-parametric statistical significance analyses comparing ML model performance across replications CV data transformations
  * NOTE: feature importance evaluation and 'target' dataset statistical comparisons are irrelevant to this phase
  * \[Runtime]: Moderately fast

* Phase 10: [Optional] Generate PDF 'Apply Replication' Summary Report
  * Generates a pre-formatted PDF including all pipeline run parameters, basic dataset information, and key exploratory analyses, ML modeling results, and statistics.
  * \[Runtime]: Moderately fast

* Phase 11: [Optional] File Cleanup
  * Deletes files that do not need to be kept following pipeline run.
  * \[Runtime]: Fast
