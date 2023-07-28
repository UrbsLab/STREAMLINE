# Run Parameters

## Overview
Here we review the run parameters available for each of the 11 phases and provide some additional run examples. 
We remind users that the parameter names described for the above notebooks sometimes different from the argument 
names when using STREAMLINE from the command-line (for brevity).


### General Parameters

| Command-line Parameter    | Config File Parameter | Description                                                      | Default    |
|---------------------------|-----------------------|------------------------------------------------------------------|------------|
| --out-path                | output_path           | path to output directory                                         | no default |
| --exp-name                | experiment_name       | name of experiment output folder (no spaces)                     | no default |
| --config                  | NA                    | flag to load config file instead of using commandline parameters | no default |
| --verbose                 | verbose               | give output to command line                                      | False      |
| --do-till-report or --dtr | do_till_report        | flag to do all phases                                            | False      |
| --do-eda                  | do_eda                | flag to eda                                                      | False      |
| --do-dataprep             | do_dataprep           | flag to data preprocessing                                       | False      |
| --do-feat-imp             | do_feat_imp           | flag to feature importance                                       | False      |
| --do-feat-sel             | do_feat_sel           | flag to feature selection                                        | False      |
| --do-model                | do_model              | flag to run models                                               | False      |
| --do-stats                | do_stats              | flag to run statistics                                           | False      |
| --do-compare-dataset      | do_compare_dataset    | flag to run compare dataset dataset                              | False      |
| --do-report               | do_report             | flag to run report dataset                                       | False      |
| --do-replicate            | do_replicate          | flag to run replication dataset                                  | False      |
| --do-rep-report           | do_rep_report         | flag to run replication report                                   | False      |
| --do-cleanup              | do_cleanup            | flag to run cleanup                                              | False      |


### Phase Specific Parameters

#### Exploratory Data Analysis Parameters

| Command-line Parameter | Config File Parameter         | Description                                                                                    | Default         |
|------------------------|-------------------------------|------------------------------------------------------------------------------------------------|-----------------|
| --data-path            | dataset_path                  | path to directory containing datasets                                                          | no default      |
| --inst-label           | instance_label                | instance label of all datasets (if present)                                                    | None            |
| --class-label          | class_label                   | outcome label of all datasets                                                                  | default="Class" |
| --match-label          | match_label                   | only applies when M selected for partition-method; indicates column with matched instance ids  | None            |
| --fi                   | ignore_features_path          | path to .csv file with feature labels to be ignored in analysis (e.g. ./droppedFeatures.csv))  | ""              |
| --cf                   | categorical_feature_path      | path to .csv file with feature labels specified to be treated as categorical where possible    | ""              |
| --qf                   | quantitative_feature_path     | path to .csv file with feature labels specified to be treated as categorical where possible    | ""              |
| --cv                   | cv_partitions                 | number of CV partitions                                                                        | 10              |
| --part                 | partition_method              | Stratified, Random, or Group Stratification                                                    | Stratified      |
| --cat-cutoff           | categorical_cutoff'           | number of unique values after which a variable is considered to be quantitative vs categorical | 10              |
| --top-features         | top_features'                 | number of top features to illustrate in figures                                                | 40              |
| --sig                  | sig_cutoff                    | p-values less than this cutoff are considered to be significant                                | 0.05            |
| --feat_miss            | featureeng_missingness        | features with a missingness proportion greater than this have a new missingness feature added to the data        | 0.5             |
| --clean_miss           | cleaning_missingness          | features (then instances) with a missingness count at or above this cutoff are removed         | 0.5             |
| --corr_thresh          | correlation_removal_threshold | one out of a pair of features is randomly removed if the correlation between this pair is >= this cutoff         | 0.8             |
| --export-fc            | export_feature_correlations   | run and export feature correlation analysis (yields correlation heatmap)                       | True            |
| --export-up            | export_univariate_plots       | export univariate analysis plots (note: univariate analysis still output by default)           | True            |
| --rand-state           | random_state                  | sets a specific random seed for reproducible results                                           | 42              |


#### Scaling and Imputation Parameters
    
| Command-line Parameter | Config File Parameter | Description                                                                                                           | Default |
|------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------|---------|
| --scale                | scale_data            | perform data scaling (required for SVM and to use Logistic regression with non-uniform feature importance estimation) | True    |
| --impute               | impute_data           | perform missing value data imputation (required for most ML algorithms if missing data is present)                    | True    |
| --multi-impute         | multi_impute'         | applies multivariate imputation to quantitative features otherwise uses median imputation                             | True    |
| --over-cv              | overwrite_cv          | overwrites earlier cv datasets with new scaled/imputed ones'                                                          | True    |

  
#### Feature Importance Phase Parameters
    
| Command-line Parameter | Config File Parameter | Description                                                                                                                                                                                            | Default |
|------------------------|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| --do-mi                | do_mutual_info        | do mutual information analysis                                                                                                                                                                         | True    |
| --do-ms                | do_multisurf          | do multiSURF analysis                                                                                                                                                                                  | True    |
| --use-turf             | use_turf'             | use TURF wrapper around MultiSURF to improve feature interaction detection in large feature spaces (only recommended if you have reason to believe at least half of your features are non-informative) | False   |
| --turf-pct             | turf_pct              | proportion of instances removed in an iteration (also dictates number of iterations                                                                                                                    | 0.5     |
| --n-jobs               | n_jobs                | number of cores dedicated to running algorithm; setting to -1 will use all available cores                                                                                                             | 1       |
| --inst-sub             | instance_subset       | sample subset size to use with multiSURF                                                                                                                                                               | 2000    |


#### Feature Selection Phase Parameters
    
| Command-line Parameter | Config File Parameter | Description                                                                    | Default |
|------------------------|-----------------------|--------------------------------------------------------------------------------|---------|
| --max-feat             | max_features_to_keep  | max features to keep (only applies if filter_poor_features is True)            | 2000    |
| --filter-feat          | filter_poor_features' | filter out the worst performing features prior to modeling                     | True    |
| --top-features         | top_features          | number of top features to illustrate in figures                                | 40      |
| --export-scores        | export_scores         | export figure summarizing average feature importance scores over cv partitions | True    |
| --over-cv-feat         | overwrite_cv_feat     | overwrites working cv datasets with new feature subset datasets                | True    |

    


#### Modeling Phase Parameters
    
| Command-line Parameter | Config File Parameter    | Description                                                                                                                                                                                       | Default                   |
|------------------------|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------|
| --algorithms           | algorithms               | comma seperated list of algorithms to run                                                                                                                                                         | 'LR,DT,NB'                |
| --model-resubmit       | model_resubmit           | flag to resubmit models instead                                                                                                                                                                   | False                     |
| --exclude              | exclude                  | comma seperated list of algorithms to exclude                                                                                                                                                     | 'eLCS,XCS'                |
| --metric               | primary_metric           | primary scikit-learn specified scoring metric used for hyper parameter optimization and permutation-based model feature importance evaluation                                                     | 'balanced_accuracy'       |
| --metric-direction     | metric_direction         | optimization direction on primary metric (maximize or minimize)                                                                                                                                   | 'maximize'                |
| --subsample            | training_subsample       | for long running algos option to subsample training set (0 for no subsample)                                                                                                                      | 0                         |
| --use-uniformFI        | use_uniform_fi           | overrides use of any available feature importance estimate methods from models instead using permutation_importance uniformly                                                                     | True                      |
| --n-trials             | n_trials                 | # of bayesian hyperparameter optimization trials using optuna (specify an integer or None)                                                                                                        | 200                       |
| --timeout              | timeout                  | seconds until hyperparameter sweep stops running new trials (Note: it may run longer to finish last trial started) If set to None STREAMLINE is completely replicable but will take longer to run | 900 i.e.(900 sec/15 mins) |
| --export-hyper-sweep   | export_hyper_sweep_plots | export optuna-generated hyperparameter sweep plots                                                                                                                                                | 'False'                   |
| --do-LCS-sweep         | do_lcs_sweep             | do LCS hyper-param tuning or use below params                                                                                                                                                     | 'False'                   |
| --nu                   | lcs_nu                   | fixed LCS nu param (recommended range 1-10) set to larger value for data with less or no noise                                                                                                    | 1                         |
| --iter                 | lcs_iterations           | fixed LCS # learning iterations param                                                                                                                                                             | 200000                    |
| --N                    | lcs_n                    | fixed LCS rule population maximum size param                                                                                                                                                      | 2000                      |
| --lcs-timeout          | lcs_timeout              | seconds until hyper parameter sweep stops for LCS algorithms                                                                                                                                      | 1200                      |
    


#### Statistics Phase Parameters
    
| Command-line Parameter | Config File Parameter | Description                                                                                                                                                           | Default             |
|------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| --plot-ROC             | plot_roc              | Plot ROC curves individually for each algorithm including all CV results and averages                                                                                 | True                |
| --plot-PRC             | plot_prc              | Plot PRC curves individually for each algorithm including all CV results and averages                                                                                 | True                |
| --plot-box             | plot_metric_boxplots  | Plot box plot summaries comparing algorithms for each metric                                                                                                          | True                |
| --plot-FI_box          | plot_fi_box           | Plot feature importance boxplots and histograms for each algorithm                                                                                                    | True                |
| --metric-weight        | metric_weight         | ML model metric used as weight in composite FI plots (only supports balanced_accuracy or roc_auc as options) Recommend setting the same as primary_metric if possible | 'balanced_accuracy' |
| --top-model-features   | top_model_features    | number of top features to illustrate in figures                                                                                                                       | 4                   |

    


Replication Phase Parameters
    

| Command-line Parameter | Config File Parameter           | Description                                                                                                                                              | Default |
|------------------------|---------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| --rep-path             | rep_data_path                   | path to directory containing replication or hold-out testing datasets (must have at least all features with same labels as in original training dataset) |         |
| --dataset              | dataset_for_rep                 | path to dataset on which to run replication or hold-out testing (must have at least all features with same labels as in original training dataset)       |         |
| --rep-export-fc        | rep_export_feature_correlations | run and export feature correlation analysis (yields correlation heatmap)                                                                                 | True    |
| --rep-plot-ROC         | rep_plot_roc                    | Plot ROC curves individually for each algorithm including all CV results and averages                                                                    | True    |
| --rep-plot-PRC         | rep_plot_prc                    | Plot PRC curves individually for each algorithm including all CV results and averages                                                                    | True    |
| --rep-plot-box         | rep_plot_metric_boxplots        | Plot box plot summaries comparing algorithms for each metric                                                                                             | True    |


Cleanup Phase Parameters
    
| Command-line Parameter | Config File Parameter | Description                               | Default |
|------------------------|-----------------------|-------------------------------------------|---------|
| --del-time             | del_time              | flag to delete runtime files              | True    |
| --del-old-cv           | del_old_cv            | flag to delete old cross-validation files | True    |



Multiprocessing Parameters
    
| Command-line Parameter | Config File Parameter | Description                                | Default |
|------------------------|-----------------------|--------------------------------------------|---------|
| --run-parallel         | run_parallel          | if run parallel on through multiprocessing | False   |
| --run-cluster          | run_cluster           | name of HPC Cluster/Method                 | "SLURM" |
| --res-mem              | reserved_memory       | reserved memory for the job (in Gigabytes) | 4       |
| --queue                | queue                 | default partition queue                    | "defq"  |






## Guidelines for Setting Parameters

## Reducing runtime
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

## Improving Modeling Performance
* Generally speaking, the more computational time you are willing to spend on ML, the better the results. Doing the opposite of the above tips for reducing runtime, will likely improve performance.
* In certain situations, setting `feature_selection` to 'False', and relying on the ML algorithms alone to identify relevant features will yield better performance.  However, this may only be computationally practical when the total number of features in an original dataset is smaller (e.g. under 2000).
* Note that eLCS, XCS, and ExSTraCS are newer algorithm implementations developed by our research group.  As such, their algorithm performance may not yet be optimized in contrast to the other well established and widely utilized options. These learning classifier system (LCS) algorithms are unique however, in their ability to model very complex associations in data, while offering a largely interpretable model made up of simple, human readable IF:THEN rules. They have also been demonstrated to be able to tackle both complex feature interactions as well as heterogeneous patterns of association (i.e. different features are predictive in different subsets of the training data).
* In problems with no noise (i.e. datasets where it is possible to achieve 100% testing accuracy), LCS algorithms (i.e. eLCS, XCS, and ExSTraCS) perform better when `nu` is set larger than 1 (i.e. 5 or 10 recommended).  This applies significantly more pressure for individual rules to achieve perfect accuracy.  In noisy problems this may lead to significant overfitting.

## Other Guidelines
* SVM and ANN modeling should only be applied when data scaling is applied by the pipeline.
* Logistic Regression' baseline model feature importance estimation is determined by the exponential of the feature's coefficient. This should only be used if data scaling is applied by the pipeline.  Otherwise `use_uniform_FI` should be True.
* While the STREAMLINE includes `impute_data` as an option that can be turned off in `DataPreprocessing`, most algorithm implementations (all those standard in scikit-learn) cannot handle missing data values with the exception of eLCS, XCS, and ExSTraCS. In general, STREAMLINE is expected to fail with an errors if run on data with missing values, while `impute_data` is set to 'False'.

## Modeling Algorithm Hyperparamters


7. (Optional/Manual Mode) Update other STREAMLINE run parameters to suit your analysis needs within code blocks 6-14. We will cover some common run parameters to consider here:
    * `cv_partitions`: The number of CV training/testing partitions created, and consequently the number of models trained for each ML algorithm. We recommend setting this between 3-10. A larger value will take longer to run but produce more accurate results.
    * `categorical_cutoff`: STREAMLINE uses this parameter to automatically determine which features to treat as categorical vs. numeric. If a feature has more than this many unique values, it is considered to be numeric.
        * Note: Currently, STREAMLINE does NOT automatically apply one-hot-encoding to categorical features meaning that all features will still be treated as numerical during ML modeling. Its currently up to the users decide whether to pre-encode features.  However STREAMLINE does take feature type into account during both the exploratory analysis, data preprocessing, and feature importance phases.
        * Note: Users can also manually specify which features to treat as categorical or even to point to features in the dataset that should be ignored in the analysis with the parameters `ignore_features_path` and `categorical_feature_path`, respectively. For either, instead of the default string 'None' setting the user specifies the path to a .csv file including a row of feature names from the dataset that should either be treated as categorical or ignored, respectively.
    * `algorithms`: A list of modeling algorithms to run, setting it to None will run all the algorithms. Must be from the set of the full or abbreviated name of models found in `streamline/models` folder.
    * `exlude`: A list of modeling algorithms to exclude from the pipeline. Must be from the set of the full or abbreviated name of models found in `streamline/models` folder.
    * * `n_trials`: Set to a higher value to give Optuna more attempts to optimize hyperparameter settings.
    * `timeout`: Set higher to increase the maximum time allowed for Optuna to run the specified `n_trials` (useful for algorithms that take more time to run)
* Note: There are a number of other run parameter options, and we encourage users to read descriptions of each to see what other options are available.
