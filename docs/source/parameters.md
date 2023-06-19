# STREAMLINE Run Parameters

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
| --sig                  | sig_cutoff                    | significance cutoff used throughout pipeline'                                                  | 0.05            |
| --feat_miss            | featureeng_missingness        | feature missingness cutoff used throughout pipeline                                            | 0.5             |
| --clean_miss           | cleaning_missingness          | cleaning missingness cutoff used throughout pipeline'                                          | 0.5             |
| --corr_thresh          | correlation_removal_threshold | correlation removal threshold                                                                  | 0.8             |
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
| --do-all               | do_all                   | run all modeling algorithms by default (when set False individual algorithms are activated individually                                                                                           | False                     |
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
