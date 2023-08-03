# Run Parameters
Here we review the run parameters available across the 9 phases of STREAMLINE. We begin with a quick guide/summary of all run parameters according to run mode along with their default values (when applicable). Then we provide further descriptions, formatting, valid values, and guidance (as needed) for each run parameter. Lastly, we provide overall guidance on setting STEAMLINE run parameters. 

***
## Quick Guide
The quick guide below distinguishes essential from non-essential run parameters within streamline, and further breaks down non-essential run paramters by pipeline phase. The name of each parameter is given for the command-line, configuration file, and notebooks (same for both Colab and Jupyter Notebooks), as well as the internal STREAMLINE default value (which ocassionally differ from the default values used in the notebooks for the [demonstration datasets](data.md#demonstration-data)). 
* Run parameters without default values are incidated with 'no default'. 
* Run parameters that are not used in one of the run modes are indicated with 'NA'.
* All run parameters include quick links to their respective details in [Parameter Details](#parameter-details), including their description, format, values, and other tips.

### Essential Parameters (Phases 1-9)

| Command-line Parameter    | Config File Parameter                                   | Notebook Parameter                           | Default    |
|---------------------------|---------------------------------------------------------|----------------------------------------------|------------|
| --data-path               | [dataset_path](#dataset-path)                           | data_path                                    | no default |
| --out-path                | [output_path](#output-path)                             | output_path                                  | no default |
| --exp-name                | [experiment_name](#experiment-name)                     | experiment_name                              | no default |
| --class-label             | [class_label](#class-label)                             | class_label                                  | 'Class'    |
| --inst-label              | [instance_label](#instance-label)                       | instance_label                               | None       |
| --match-label             | [match_label](#match-label)                             | match_label                                  | None       |
| --fi                      | [ignore_features_path](#ignore-features-path)           | ignore_features                              | None       |
| --cf                      | [categorical_feature_path](#categorical-feature_path)   | categorical_feature_headers                  | None       |
| --qf                      | [quantitative_feature_path](#quantitative-feature_path) | quantitiative_feature_headers                | None       |
| --rep-path                | [rep_data_path](#rep-data-path)                         | rep_data_path                                | no default |
| --dataset                 | [dataset_for_rep](#dataset-for-rep)                     | dataset_for_rep                              | no default |
| [--config](#config)       | NA                                                      | NA                                           | no default |
| --do-till-report or --dtr | [do_till_report](#do-till-report)                       | NA                                           | False      |
| --do-eda                  | [do_eda](#do-eda)                                       | NA                                           | False      |
| --do-dataprep             | [do_dataprep](#do-dataprep)                             | NA                                           | False      |
| --do-feat-imp             | [do_feat_imp](#do-feat_imp)                             | NA                                           | False      |
| --do-feat-sel             | [do_feat_sel](#do-feat_sel)                             | NA                                           | False      |
| --do-model                | [do_model](#do-model)                                   | NA                                           | False      |
| --do-stats                | [do_stats](#do-stats)                                   | NA                                           | False      |
| --do-compare-dataset      | [do_compare_dataset](#do-compare-dataset)               | NA                                           | False      |
| --do-report               | [do_report](#do-report)                                 | NA                                           | False      |
| --do-replicate            | [do_replicate](#do-replicate)                           | NA                                           | False      |
| --do-rep-report           | [do_rep_report](#do-rep-report)                         | NA                                           | False      |
| --do-cleanup              | [do_cleanup](#do-cleanup)                               | NA                                           | False      |
| NA                        | NA                                                      | [applyToReplication](#applyToReplication)    | True       |
| NA                        | NA                                                      | [demo_run](#demo-run)                        | True       |
| NA                        | NA                                                      | [use_data_prompt](#use-data-prompt) (Colab)  | True       |

### General Parameters (Phase 1)

| Command-line Parameter    | Config File Parameter                     | Notebook Parameter                | Default      |
|---------------------------|-------------------------------------------|-----------------------------------|--------------|
| --cv                      | [cv_partitions](#cv-partitions)           | n_splits                          | 10           |
| --part                    | [partition_method](#partition-method)     | partition_method                  | 'Stratified' |
| --cat-cutoff              | [categorical_cutoff](#categorical-cutoff) | categorical_cutoff                | 10           |
| --sig                     | [sig_cutoff](#sig-cutoff)                 | sig_cutoff                        | 0.05         |
| --rand-state              | [random_state](#random-state)             | random_state                      | 42           |

### Data Processing Parameters (Phase 1)

| Command-line Parameter    | Config File Parameter                                            | Notebook Parameter                | Default    |
|---------------------------|------------------------------------------------------------------|-----------------------------------|------------|
| --exclude-eda-output      | [exclude_eda_output](#exclude-eda-output)                        | exclude_eda_output                | None       |
| --top-uni-feature         | [top_uni_features](#top-uni-features)                            | top_uni_features                  | 20         |
| --feat_miss               | [featureeng_missingness](#featureeng-missingness)                | featureeng_missingness            | 0.5        |
| --clean_miss              | [cleaning_missingness](#cleaning-missingness)                    | cleaning_missingness              | 0.5        |
| --corr_thresh             | [correlation_removal_threshold](#correlation-removal-threshold)  | correlation_removal_threshold     | 1.0        |


### Imputation & Scaling Parameters (Phase 2)

| Command-line Parameter | Config File Parameter          | Notebook Parameter          | Default |
|------------------------|--------------------------------|-----------------------------|---------|
| --impute               | [impute_data](#impute-data)    | impute_data                 | True    |
| --multi-impute         | [multi_impute](#multi-impute)  | multi_impute                | True    |
| --scale                | [scale_data](#scale-data)      | scale_data                  | True    |
| --over-cv              | [overwrite_cv](#overwrite-cv)  | overwrite_cv                | True    |

### Feature Importance Estimation Parameters (Phase 3)

| Command-line Parameter | Config File Parameter                | Notebook Parameter                            | Default |
|------------------------|--------------------------------------|-----------------------------------------------|---------|
| --do-mi                | [do_mutual_info](#do-mutual-info)    | do_mutual_info                                | True    |
| --do-ms                | [do_multisurf](#do-multisurf)        | do_multisurf                                  | True    |
| --use-turf             | [use_turf](#use-turf)                | use_TURF                                      | False   |
| --turf-pct             | [turf_pct](#turf-pct)                | TURF_pct                                      | 0.5     |
| --inst-sub             | [instance_subset](#instance-subset)  | instance_subset                               | 2000    |
| --n-jobs               | [n_jobs](#n-jobs)                    | cores                                         | 1       |

### Feature Selection Parameters (Phase 4)

| Command-line Parameter | Config File Parameter                          | Notebook Parameter                            | Default |
|------------------------|------------------------------------------------|-----------------------------------------------|---------|
| --filter-feat          | [filter_poor_features](#filter-poor-features)  | filter_poor_features                          | True    |
| --max-feat             | [max_features_to_keep](#max-features-to-keep)  | max_features_to_keep                          | 2000    |
| --export-scores        | [export_scores](#export-scores)                | export_scores                                 | True    |
| --top-fi-features      | [top_fi_features](#top-fi-features)            | top_fi_features                               | 40      |

### Modeling Parameters (Phase 5)
 Command-line Parameter  | Config File Parameter                                 | Notebook Parameter                 | Default                   |
|------------------------|-------------------------------------------------------|------------------------------------|---------------------------|
| --algorithms           | [algorithms](#algorithms)                             | algorithms                         | None                      |
| --exclude              | [exclude](#exclude)                                   | exclude                            | 'eLCS,XCS'                |
| --subsample            | [training_subsample](#training-subsample)             | training_subsample                 | 0                         |
| --use-uniformFI        | [use_uniform_fi](#use-uniform-fi)                     | use_uniform_FI                     | True                      |
| --metric               | [primary_metric](#primary-metric)                     | primary_metric                     | 'balanced_accuracy'       |
| --metric-direction     | [metric_direction](#metric-direction)                 | metric_direction                   | 'maximize'                |
| --n-trials             | [n_trials](#n-trials)                                 | n_trials                           | 200                       |
| --timeout              | [timeout](#timeout)                                   | timeout                            | 900                       |    
| --export-hyper-sweep   | [export_hyper_sweep_plots](#export-hyper-sweep-plots) | export_hyper_sweep_plots           | False                     |
| --do-LCS-sweep         | [do_lcs_sweep](#do-lcs-sweep)                         | do_lcs_sweep                       | False                     |
| --nu                   | [lcs_nu](#lcs-nu)                                     | lcs_nu                             | 1                         |
| --iter                 | [lcs_iterations](#lcs-iterations)                     | lcs_iterations                     | 200000                    |
| --N                    | [lcs_n](#lcs-n)                                       | lcs_N                              | 2000                      |
| --lcs-timeout          | [lcs_timeout](#lcs-timeout)                           | lcs_timeout                        | 1200                      |
| --model-resubmit       | [model_resubmit](#model-resubmit)                     | NA                                 | False                     |

### Post-Analysis Parameters (Phase 6)
| Command-line Parameter   | Config File Parameter                            | Notebook Parameter      | Default             |
|--------------------------|--------------------------------------------------|-------------------------|---------------------|
| --exclude-plots          | [exclude_plots](#exclude-plots)                  | exclude_plots           | None                |
| --metric-weight          | [metric_weight](#metric-weight)                  | metric_weight           | 'balanced_accuracy' |
| --top-model-fi-features  | [top_model_fi_features](#top-model-fi-features)  | top_model_fi_features   | 40                  |

### Compare Data Parameters (Phase 7)
There are currently no run parameters to adjust for this phase.

### Replication Parameters (Phase 8)

| Command-line Parameter | Config File Parameter                    | Notebook Parameter  | Default |
|------------------------|------------------------------------------|---------------------|---------|
| --exclude-rep-plots    | [exclude_rep_plots](#exclude-rep-plots)  | exclude_rep_plots   | None    |

### Summary Report Parameters (Phase 9)
There are currently no run parameters to adjust for this phase.

### Cleanup Parameters

| Command-line Parameter | Config File Parameter      | Notebook Parameter | Default |
|------------------------|----------------------------|--------------------|---------|
| --del-time             | [del_time](#del-time)      | del_time           | True    |
| --del-old-cv           | [del_old_cv](#del-old-cv)  | del_old_cv         | True    |

### Multiprocessing Parameters

| Command-line Parameter | Config File Parameter                | Notebook Parameter  | Default |
|------------------------|--------------------------------------|---------------------|---------|
| --run-parallel         | [run_parallel](#run-parallel)        | NA                  | False   |
| --run-cluster          | [run_cluster](#run-cluster)          | NA                  | "SLURM" |
| --res-mem              | [reserved_memory](#reserved-memory)  | NA                  | 4       |
| --queue                | [queue](#queue)                      | NA                  | "defq"  |

### Logging Parameters
| Command-line Parameter    | Config File Parameter                     | Notebook Parameter                | Default      |
|---------------------------|-------------------------------------------|-----------------------------------|--------------|
| --verbose                 | [verbose](#verbose)                       | NA                                | False        |
| --logging-level           | [logging_level](#logging-level)           | NA                                | 'INFO'       |

***
## Parameter Details
This section will go into greater depth for each run parameter, primarily using the configuration file parameter name to identify each. 
* *Parameters identified as (str) format should be entered with single quotation marks within notebooks, or when using a configuration file, but without them when using command line arguments (CLA).* 

***
### Essential Parameters (Phase 1-9)

#### dataset_path
* **Description:** path to the folder containing one or more 'target datasets' to be analyzed that meet dataset [formatting requirements](data.md#input-data-requirements)
* **Format:** (str), e.g. `'/content/STREAMLINE/data/DemoData'`
* **Values:** must be a valid folder-path
* **Tips:** STREAMLINE automatically detects the number of 'target datasets' in this folder and will run a complete analysis on each, comparing dataset performance in phase 7

#### output_path  
* **Description:** path to an output folder where STREAMLINE will save the experiment folder (containing all output files)
* **Format:** (str), e.g. `'/content/DemoOutput'`
* **Values:** must be a valid folder-path, however the lowest level of the folder (e.g. DemoOutput) does not already have to exist, and will be automatically created if it does not
* **Tips:** When running multiple STREAMLINE experiments, it's convenient to leave this parameter the same and just update `experiment_name`

#### experiment_name
* **Description:** a unique name for the current STREAMLINE experiment output folder that will be created within `output_path`
* **Format:** (str), e.g. `'demo_experiment'`
* **Values:** any string value name (avoid spaces)
* **Tips:** a short, unique, and descriptive name is encouraged

#### class_label
* **Description:** the name of the class/outcome column found in the dataset header
* **Format:** (str), e.g. `'Class'`
* **Values:** the case-sensitive name used in the dataset to identify the outcome labels column

#### instance_label  
* **Description:** the name of the instance ID column that may (or may not) be included in the dataset
* **Format:** (str), e.g. `'InstanceID'`
* **Values:** `None`, or the case-sensitive name used in the dataset to identify the instance ID column (if present)
* **Tips:** having an instance ID column in the data allows users to later identify model predictions for specific instances in the dataset, as well as reverse-engineer instance subgroups in the dataset downstream using the ExSTraCS modeling algorithm's capability to detect and characterize heterogeneous associations. This may not be necessesary for most users. 

#### match_label
* **Description:** the name of the match/group ID column that can be included in a dataset to keep instances with the same match label together within the same CV partition
* **Format:** (str), e.g. `'MatchID'`
* **Values:** `None`, or the case-sensitive name used in the dataset to identify the match/group ID column (if present)
* **Tips:** having a match/group ID column in the data allows users to apply machine learning modeling to datasets where instances with different outcomes have been matched based on other covariates that the user wants to account for (e.g. age, sex, race, etc)

#### ignore_features_path  
* **Description:** a list of feature names for STREAMLINE to immediately drop from the target datasets
* **Format:** 
    1. for notebook or config file modes: provide a (list) of (str) feature names that can be found in any of the 'target datasets', e.g. `['IgnoredFeature1','IgnoredFeature2']`
    2. for command line arguments: provide a (str) path to a `.csv` file including a row of feature names that can be found in any of the 'target datasets', e.g. `'/content/STREAMLINE/data/MadeUp/ignoreFeat.csv'`
* **Values:** `None`, or (for either format) should include case-sensitive feature names found in at least one of the 'target datasets'
* **Tips:** useful for easily dropping features found in the datasets that users may wish to exclude if those features might lead to data leakage, or for other data quality reasons

#### categorical_feature_path
* **Description:** a list of feature names for STREAMLINE to explicitly treat as categorical feature types
* **Format:**
    1. for notebook or config file modes: provide a (list) of (str) feature names that can be found in any of the 'target datasets', e.g. `['Feature1','Feature7']`
    2. for command line arguments: provide a (str) path to a `.csv` file including a row of feature names that can be found in any of the 'target datasets', e.g. `'/content/STREAMLINE/data/DemoFeatureTypes/hcc_cat_feat.csv'`
* **Values:** `None`, or (for either format) should include case-sensitive feature names found in at least one of the 'target datasets'
* **Tips:** 
    * When specifying `categorical_feature_path` feature names and leaving `quantiative_feature_path = None` all other features will be automatically treated as quanatiative
    * When specifying `quantiative_feature_path` feature names and leaving `categorical_feature_path = None` all other features will be automatically treated as categorical
    * When specifying feature names for both `categorical_feature_path` and `quantiative_feature_path`, any features in the data not specified by one of theses lists will have it's feature type determined automatically using [categorical_cutoff](#categorical_cutoff)
    * Note: any text-valued features in a dataset will automatically be numerically encoded and treated as categorical features (overriding any other user specifications)

#### quantitative_feature_path
* **Description:** a list of feature names for STREAMLINE to explicitly treat as quantitative feature types
    * All other aspects of this parameter are the same as for [categorical_feature_path](#categorical_feature_path)

#### rep_data_path
* **Description:** path to the folder containing one or more 'replication datasets' to be evaluated using previously trained models for a specific 'target dataset' (see [data formatting requirements](data.md#input-data-requirements))
* **Format:** (str), e.g. `'/content/STREAMLINE/data/DemoRepData'`
* **Values:** must be a valid folder-path
* **Tips:** STREAMLINE automatically detects the number of 'replication datasets' in this folder and will run a complete evaluation on each.

#### dataset_for_rep
* **Description:** path to the individual 'target dataset' file used to train the models which you want to evaluate with the above 'replication datasets' (see [data formatting requirements](data.md#input-data-requirements))
* **Format:** (str), e.g. `'/content/STREAMLINE/data/DemoData/hcc-data_example_custom.csv'`
* **Values:** must be a valid file-path
* **Tips:** STREAMLINE's replication phase is set up to evaluate all models trained from a single 'target datasets' at once using one or more replication datasets, specific to that 'target dataset'. The replication phase can be run multiple times, each for a new 'target dataset', and it's own respective 'replication dataset(s)'.

#### config
* **Description:** path to the configuration file used to run STREAMLINE from the command line using a configuration file [locally](running.md#using-a-configuration-file-locally) or on a [cluster](running.md#using-a-configuration-file-cluster)
* **Format:** (str), e.g. `run_configs/local.cfg`
* **Values:** must be a valid file-path to a properly formatted configuration file

#### do_till_report 
* **Description:** boolean flag telling STREAMLINE to automatically run all phases excluding phase 8 (i.e. replication), and part of phase 9 (i.e. PDF report for replication)
* **Format:** [Command Line Argument] just use flag (i.e. `--do-till-report`), [Configuration File] (bool) 
* **Values:** `True` or `False`

#### do_eda
* **Description:** boolean flag telling STREAMLINE to run phase 1 (i.e. EDA and Processing)
* **Format:** [Command Line Argument] just use flag (i.e. `--do-eda`), [Configuration File] (bool) 
* **Values:** `True` or `False`

#### do_dataprep 
* **Description:** boolean flag telling STREAMLINE to run phase 2 (i.e. Imputation and Scaling)
* **Format:** [Command Line Argument] just use flag (i.e. `--do-dataprep`), [Configuration File] (bool) 
* **Values:** `True` or `False`

#### do_feat_imp
* **Description:** boolean flag telling STREAMLINE to run phase 3 (i.e. Feature Importance Estimation)
* **Format:** [Command Line Argument] just use flag (i.e. `--do-feat-imp`), [Configuration File] (bool) 
* **Values:** `True` or `False`

#### do_feat_sel
* **Description:** boolean flag telling STREAMLINE to run phase 4 (i.e. Feature Selection)
* **Format:** [Command Line Argument] just use flag (i.e. `--do-feat-sel`), [Configuration File] (bool) 
* **Values:** `True` or `False`

#### do_model
* **Description:** boolean flag telling STREAMLINE to run phase 5 (i.e. Modeling)
* **Format:** [Command Line Argument] just use flag (i.e. `--do-model`), [Configuration File] (bool) 
* **Values:** `True` or `False`

#### do_stats
* **Description:** boolean flag telling STREAMLINE to run phase 6 (i.e. Post-Analysis)
* **Format:** [Command Line Argument] just use flag (i.e. `--do-stats`), [Configuration File] (bool) 
* **Values:** `True` or `False`

#### do_compare_dataset
* **Description:** boolean flag telling STREAMLINE to run phase 7 (i.e. Compare Datasets)
* **Format:** [Command Line Argument] just use flag (i.e. `--do-compare-dataset`), [Configuration File] (bool) 
* **Values:** `True` or `False`

#### do_report
* **Description:** boolean flag telling STREAMLINE to run phase 9 (i.e. Summary Report) specific to phases 1-7
* **Format:** [Command Line Argument] just use flag (i.e. `--do-report`), [Configuration File] (bool) 
* **Values:** `True` or `False`

#### do_replicate
* **Description:** boolean flag telling STREAMLINE to run phase 8 (i.e. Replication) specific to phases 1-7
* **Format:** [Command Line Argument] just use flag (i.e. `--do-replicate`), [Configuration File] (bool) 
* **Values:** `True` or `False`

#### do_rep_report
* **Description:** boolean flag telling STREAMLINE to run phase 9 (i.e. Summary Report) specific to phase 8
* **Format:** [Command Line Argument] just use flag (i.e. `--do-rep-report`), [Configuration File] (bool) 
* **Values:** `True` or `False`

#### do_cleanup
* **Description:** boolean flag telling STREAMLINE to run output file cleanup (optional)
* **Format:** [Command Line Argument] just use flag (i.e. `--do-cleanup`), [Configuration File] (bool) 
* **Values:** `True` or `False`

#### applyToReplication
* **Description:** a notebook-specific parameter indicating whether to include running phase 8 (i.e. Replication) 
* **Format:** (bool) 
* **Values:** `True` or `False`

#### demo_run
* **Description:** a notebook-specific parameter indicating whether to automatically run the notebook on the [demonstration datasets](data.md#demonstration-data)
* **Format:** (bool) 
* **Values:** `True` or `False`

#### use_data_prompt
* **Description:** a notebook-specific parameter that activates a notebook prompt to gather essential run parameter information directly from the user rather than have them manually update code cells
* **Format:** (bool) 
* **Values:** `True` or `False`

***
### General Parameters (Phase 1)

#### cv_partitions
* **Description:** *k*, the number of *k*-fold cross validation training/testing data partitions to create and apply throughout pipeline
* **Format:** (int)
* **Values:** an integer between `3` and `10` is recommended
* **Tips:** smaller values will yield shorter STREAMLINE run times, but training datasets will have a smaller number of instances

#### partition_method 
* **Description:** the cross validation strategy used
* **Format:** (str)
* **Values:** `'Stratified'`, `'Random'`, or `'Group'`
* **Tips:** `'Stratified'` is generally recommended in order to keep class balance as similar as possible within respective partitions, however `'Group'` can be selected when `match_label` has been specified to keep instances with the same match/group ID together within a respective partition

#### categorical_cutoff 
* **Description:** the number of unique values observed for a given feature in a 'target dataset' after which a variable is automatcially considered to be quantitative
* **Format:** (int)
* **Values:** an integer between `3` and `10` is generally recommended, but should be set in a dataset-specific manner
* **Tips:** this parameter will only be used if the user hasn't specifically indicated which features to treat as categorical or quantitative using `categorical_feature_path` and/or `quantiative_feature_path`, respectively. However depending on the specific dataset, users can sometimes conveniently set this parameter to correctly assign variable types, e.g. if all categorical features in the dataset have fewer than 5 unique values, but quantitative ones all have more than 10 unique values, setting `categorical_cutoff = 7` will make correct feature type assignments automatically.

#### sig_cutoff 
* **Description:** the statistical significance cutoff used throughout the pipeline used in deciding whether to run pair-wise non-parametric statistical comparisons following group comparisons, and for identifying significant results in output files with a '*'
* **Format:** (float)
* **Values:** a value <= `0.05` is recommended
* **Tips:** Note: STREAMLINE does not currently automatically account for multiple testing - users should take this into consideration themselves

#### random_state 
* **Description:** sets a specific random seed for the STREAMLINE run (important for pipeline reproducibility)
* **Format:** (int) 
* **Values:** any positive integer value is fine
* **Tips:** make sure to use the same value for `random_state` in a separate run along with the same datasets and run parameters to obtain reproducible pipeline results

***
### Data Processing Parameters (Phase 1)

#### exclude_eda_output
* **Description:** allows users to exclude some of the outputs automatically generated by STREAMLINE during phase 1
* **Format:** 
    1. for notebook or config file modes: provide a (list) of valid options (str) , e.g. `['describe','univariate_plots','correlation_plots']`
    2. for command line arguments: provide as a list of comma separated values with no spaces, e.g. `describe,univariate_plots,correlation_plots`
* **Values:** `None`, or [`'describe'`, `'univariate_plots'`, or `'correlation_plots'`] - provided in format above
    * `describe` - don't run or output the set of standard pandas functions (i.e. `Describe()`, `Dtypes()`, and `nunique()`) as `.csv` files
    * `univariate_plots` - don't output individual univariate analysis plots illustrating features vs. outcome (by default STREAMLINE outputs these plots for any feature with a significant univariate association based on `sig_cutoff`)
    * `correlation_plots` - don't output feature correlation heatmaps for the 'initial' or 'processed' data EDA

#### top_uni_features
* **Description:** number of most significant features to report in the notebook and PDF summary
* **Format:** (int)
* **Values:** an integer between `10` and `40` is recommended

#### featureeng_missingness
* **Description:** the proportion of missing values within a feature (*above which*) a new binary categorical feature is generated that indicates if the value for an instance was missing or not
* **Format:** (float)
* **Values:** (`0.0` - `1.0`)
* **Tips:** this parameter controls automated feature engineering of a new 'missingness' feature, generated for another pre-existing feature in the 'target dataset'. It's useful for identifying the potentially predictive value of any feature who's missingness is not completely at random (NCAR)

#### cleaning_missingness
* **Description:** the proportion of missing values, within a feature or instance, (*at which*) the given feature or instance will be automatically cleaned (i.e. removed) from the processed 'target dataset'
* **Format:** (float)
* **Values:** (`0.0` - `1.0`)
* **Tips:** this parameter controls automated data cleaning based on feature or instance 'missingness'. STREAMLINE will first remove features with high missingness, then subsequently remove any instances with missingness over this proportion.

#### correlation_removal_threshold
* **Description:** the (pearson) feature correlation at which one out of a pair of features is randomly removed from the processed 'target dataset'
* **Format:** (float)
* **Values:** (`0.0` - `1.0`)
* **Tips:** this parameter controls automated data cleaning based on feature correlation. The safest setting (to avoid missing predictive information) is the default of 1.0 (i.e. perfect correlation between two features). Note: STREAMLINE interprets this parameter as both a positive and negative correlation threshold.

***
### Imputation & Scaling Parameters (Phase 2)

#### impute_data
* **Description:** indicates whether or not to apply missing data imputation to features in the data or not
* **Format:** (bool)
* **Values:** `True` or `False`
* **Tips:** leaving to the default value of `True` is recommended but not always neccessary depending on whether missing data is present in the original datasets or what algorithms a user wishes to run (e.g. ExSTraCS can handle missing values in data)

#### multi_impute
* **Description:** indicates whether or not to apply multiple imputation using scikit-learn's [IterativeImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html) for imputing missing values in quantiative features. Mode imputation is always applied for categorical features.
* **Format:** (bool)
* **Values:** `True` or `False`
* **Tips:** for larger datasets, multiple imputation can run very slowly, and take up alot of disk space in the pickled imputation files that are automatically stored for downstream imputation of replication data or further external application of the models. When `False`, median imputation is instead used for quantiative features.

#### scale_data
* **Description:** indicates whether or not to apply standard scaling to features in the data or not
* **Format:** (bool)
* **Values:** `True` or `False`
* **Tips:** leaving to the default value of `True` is recommended but not always neccessary depending on what algorithms a user wishes to run (see [Imputation and Scaling](pipeline.md#Phase-2-imputation-and-scaling))

#### overwrite_cv
* **Description:** indicates whether or not to overwrite earlier versions of CV (training and testing) datasets with newly imputed and scaled CV datasets. This parameter is also applied after phase 4 (feature selection)
* **Format:** (bool)
* **Values:** `True` or `False`
* **Tips:** `True` will reduce the number of output files generated (and storage space) keeping only the final processed, imputed, scaled, and feature selected CV datasets, however `False` allows users to view intermediary CV datasets following phase one data processing and CV partitioning, as wel as intermediary CV datasets after additional feature selection

***
### Feature Importance Estimation Parameters (Phase 3)

#### do_mutual_info
* **Description:** indicates whether or not to run mutual information as a feature importance estimation algorithm (prior to modeling)
* **Format:** (bool)
* **Values:** `True` or `False`
* **Tips:** mutual information is good at detecting univariate association between a given feature and outcome. While we recommend running both feature importance algorithms, users should specify `True` for at least one algorithm.

#### do_multisurf   
* **Description:** indicates whether or not to run MultiSURF as a feature importance estimation algorithm (prior to modeling)
* **Format:** (bool)
* **Values:** `True` or `False`
* **Tips:** MultiSURF is good at detecting both features involved in an interaction and univariate association with outcome. While we recommend running both feature importance algorithms, users should specify `True` for at least one algorithm.

#### use_turf
* **Description:** indicates whether or not to run TuRF, a wrapper algorithm that operates around MultiSURF, improving it's ability to detect feature interactions in data with larger numbers of features
* **Format:** (bool)
* **Values:** `True` or `False`
* **Tips:** using TuRF is strongly recommended in datasets with >10,000 features, but can improve feature importance rankings in datasets with fewer features as well

#### turf_pct
* **Description:** this parameter currently serves two functions: (1) it determines the propotion of instances removed from consideration during a TuRF iteration, and (2) it dictates the number of TuRF iteractions (where the nubmer of iterations is 1/`turf_pct`)
* **Format:** (float)
* **Values:** (`0.01`- `0.5`)
* **Tips:** setting `turf_pct` to 0.5 will run MultiSURF twice, removing the lowest scoring half of features in the first iteration (and giving them a very low feature importance score), then running MultiSURF again on the remaining features to rescore them. A setting of 0.2 would remove 20% of features each iteration, over 5 iterations. Thus lower values for this parameter will increase run time.

#### instance_subset
* **Description:** the number of randomly chosen instances in the training data used to use for running MultiSURF
* **Format:** (int)
* **Values:** any integer above `500` is recommended, but the default of `2000` seems to be a reasonable trade-off in many cases between run time and performance
* **Tips:** the MultiSURF algorithm scales quadratically with the number of features in the data, but linearly with the number of features. Thus a dataset with a large number of training instances can make MultiSURF run very slowly. However, MultiSURF does not necessarily need to see all training instances to reasonably estimate feature imporance. If this parameter is set larger than the number of instances in a given training dataset, it will simply use all available training instances.

#### n_jobs
* **Description:** the number of CPU cores dedicated to running MultiSURF
* **Format:** (int)
* **Values:** `-1`, or a positive integer <= the number of cores available on your machine
* **Tips:** -1 will run MultiSURF on all available cores when run locally

***
### Feature Selection Parameters (Phase 4)

#### filter_poor_features
* **Description:** indicates whether or not to apply feature selection to the dataset
* **Format:** (bool)
* **Values:** `True` or `False`
* **Tips:** when set to `False` all features will be preserved in the datasets for phase 5 modeling

#### max_features_to_keep
* **Description:** indicates the maximum number of top scorign features to retain in the datasets prior to phase 5 modeling (based on the scores of the feature importance estimation algorithms, i.e. Mutual Information and MultiSURF)
* **Format:** (int or `None`)
* **Values:** any positive integer > `1` is acceptable
* **Tips:** we have set the default of this parameter to `2000` primarily to limit the computational burden of modeling. Users should use their own judgment in setting this parameter for the dataset/task in hand. When set to `None` (and `filter_poor_features = True`), STREAMLINE will automatically remove any feature that scored <= 0 for each feature importance estimation algorithm run. When set to an integer such as `2000` (and `filter_poor_features = True`), STREAMLINE will first remove any feature that scored <= `0` for each feature importance estimation algorithm run, then alternate between the sets of feature importance rankings keeping the top scoring (non-redundant) features from each algorithm.

#### export_scores 
* **Description:** indicates whether or not to export barplots for the feature importance estimation algorithms (Mutual Information and MultiSURF) summarizing average feature importance scores over CV training partitions
* **Format:** (bool)
* **Values:** `True` or `False`

#### top_fi_features 
* **Description:** number of top scoring features (mean over CV runs) to illustrate in the above feature importance estimation bar plots generated when `export_scores = True` 
* **Format:** (int)
* **Values:** an integer between `10` and `40` is recommended

***
### Modeling Parameters (Phase 5)

#### algorithms 
* **Description:** used to specify which machine learning modeling algorithms will be applied
* **Format:** (list of 'str' values, or `None`)
    1. for notebook or config file modes: provide a (list) of (str) algorithm identifiers, e.g. `['NB','LR','EN','DT','RF','XGB','SVM','ANN','KNN','GP','ExSTraCS]`
    2. for command line arguments: provide as a list of comma separated values with no spaces, e.g. `NB,LR,EN,DT,RF,XGB,SVM,ANN,KNN,GP,ExSTraCS`
* **Values:** `None`, or any subset of the following ['NB','LR','EN','DT','RF','GB','XGB','LBG','CGB','SVM','ANN','KNN','GP','eLCS','XCS','ExSTraCS], where:
    * Naive Bayes (NB)
    * Logistic Regression (LR)
    * Elastic Net (EN)
    * Decision Tree (DT)
    * Random Forest (RF)
    * Gradient Boosting (GB)
    * Extreame Gradient Boosting (XGB)
    * Light Gradient Boosting (LGB)
    * Category Gradient Boosting (CGB)
    * Support Vector Machines (SVM)
    * Artificial Neural Networks (ANN)
    * K-Nearest Neighbors (KNN)
    * Genetic Programming, i.e. symbolic classification (GP)
    * Educational Learning Classifier System (eLCS)
    * 'X' Classifier System (XCS)
    * Extended Supervised Tracking Classifier System (ExSTraCS)
* **Tips:** setting this parameter to `None` will run all algorithms in STREAMLINE with the exception of any algorithms specified within `exclude`. To run a fairly comprehensive subset of algorithms (without running them all), we recommend `['NB','LR','EN','DT','RF','XGB','SVM','ANN','KNN','GP','ExSTraCS]`. Specifying algorithms using this parameter is most convenient when you want to run a small subset of algorithms, e.g. `['NB','LR','DT']`

#### exclude  
* **Description:** used to specify which machine learning modeling algorithms to exclude from analysis 
* **Format:** (list of 'str' values, or `None`)
    1. for notebook or config file modes: provide a (list) of (str) algorithm identifiers, e.g. `['eLCS','XCS']`
    2. for command line arguments: provide as a list of comma separated values with no spaces, e.g. `eLCS,XCS`
* **Values:** same as for `algorithms` above
* **Tips:** setting this parameter to `None` just tells STREAMLINE not to exclude any additional algorithms not already specified within `algorithms`. Currently, by default STREAMLINE excludes `eLCS` and `XCS` from an analysis. Specifying algorithms using this parameter is most convenient when you want to exclude a small subset of algorithms, e.g. `['SVM','eLCS','XCS']`.

#### training_subsample 
* **Description:**  the number of randomly chosen instances in the training data used to use for training certain longer running algorithms (i.e. XGB,SVM,KN,ANN,LR,eLCS,XCS,ExStraCS)
* **Format:** (`0`, or another int)
* **Values:** the default of `0` will use all training data. Otherwise, any positive integer is acceptable.
* **Tips:** In general, we recommend leaving this parameter to `0`, however some algorithms may take a very long time to run. If you're worried about this recommend setting this parameter to `2000` as a reasonable trade-off in many cases between run time and performance.

#### use_uniform_fi
* **Description:** indicates whether or not to override any available (modeling-algorithm-specific) model-feature-importance estimation methods, instead using scikit-learn's [permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html) estimator uniformly for all algorithms
* **Format:** (bool)
* **Values:** `True` or `False`
* **Tips:** when `True`, model feature importance will be estimated in the same way for all models/algorithms. However, when `False` the following algorithms have their own unique strategies of estimating model feature importance, that will be used instead: (i.e. LR,DT,RF,XGB,LGB,GB,eLCS,XCS,ExSTraCS). Any algorithms without an internal strategy for estimating model feature importance will rely on permuation importance by default.

#### primary_metric
* **Description:** the evaluation metric used to optimize hyperparameters
* **Format:** (str)
* **Values:** We recommend `'balanced_accuracy'`, `'roc_auc'`, or `'f1'` (based on the users needs/priorities), however it can be any available metric identifier from (https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)

#### metric_direction
* **Description:** indicates whether the `primary_metric` should be maximized or minimized during hyperparameter optimization
* **Format:** (str)
* **Values:** `maximize` or `minimize`
* **Tips:** For almost all metrics (including `'balanced_accuracy'`, `'roc_auc'`, or `'f1'`), this should be `maximize`

#### n_trials
* **Description:** an [Optuna](https://optuna.org/) parameter controlling the number of hyperparameter optimization trials to be conducted
* **Format:** (int)
* **Values:** any positive integer > `1`, (`200` by default)
* **Tips:** When this parameter is set to a larger value, hyperparameter optimization will take longer to complete, but a broader range of hyperparameter configurations will be considered which can improve algorithm modeling performance

#### timeout
* **Description:** an [Optuna](https://optuna.org/) parameter controlling the total number of *seconds* until a given hyperparameter sweep stops running new trials
* **Format:** (int, or `None`)
* **Values:** any positive integer > `1`, (`900` by default, i.e. 15 minutes), or `None`
* **Tips:** To ensure STREAMLINE reproducibility, this parameter must be set to `None`, however this will force all algorithms to fully complete the number of trials specified by `n_trials`. When set to an integer, Optuna will submit new trials (as previous ones complete), up until this time limit, and then only use the hyperparameter sweep trials it has completed to pick the best hyperparameter settings for the given algorithm. Any trial already started after this time limit is reached, will continue to run until completion. This means that one algorithm can spend more total time on hyperparameter trials than another, when this parameter is given a time limit.

#### export_hyper_sweep_plots
* **Description:** indicates whether or not to generate an [Optuna](https://optuna.org/)-plot visualizing the hyperparameter sweep of an algorithm on a given dataset
* **Format:** (bool)
* **Values:** `True` or `False`

#### do_lcs_sweep 
* **Description:** indicates whether or not to apply an [Optuna](https://optuna.org/) hyperparameter sweep to one of the rule-based ML algorithms, i.e. (eLCS, XCS, ExSTraCS)
* **Format:** (bool)
* **Values:** `True` or `False`
* **Tips:** Learning classifier system (LCS), i.e. rule-based ML modeling algorithms can be computationally expensive, but have fairly reliable default run parameter settings. This parameter allow users to avoid a hyperparameter sweep, and train each LCS algorithm only once on manually specified run parameters. To save run time, in general we recommend leaving this parameter to `False` and specifying the LCS run parameters described below. Watch this [video](https://www.youtube.com/watch?v=CRge_cZ2cJc) to learn LCS basics.

#### lcs_nu 
* **Description:** specifies the *nu* parameter used by LCS algorithms (i.e. eLCS,XCS,ExSTraCS)
* **Format:** (int)
* **Values:** (`1` - `10`)
* **Tips:** higher values place more pressure for these algorithms to generate perfectly accurate rules, which easily leads to overfitting in noisy problems. Unless you know that your models should be able to achieve 100% testing accuracy on the target data, we recommend leaving this parameter to the default of `1`. Watch this [video](https://www.youtube.com/watch?v=CRge_cZ2cJc) to learn LCS basics.

#### lcs_iterations  
* **Description:** specifies the number of learning iterations an LCS algorithm will run (i.e. eLCS,XCS,ExSTraCS)
* **Format:** (int)
* **Values:** a positive integer at least two times larger than the number of training instances in the target data
* **Tips:** each iteration, an LCS algorithm focuses on one instance in the training dataset, thus this parameter should always be larger (ideally much larger) than the number of training instances in the data. For most users we recommend the default value of `200000` as a starting point, however, as a key run parameter, more learning iterations is typically expected to improve LCS algorithm performance. Watch this [video](https://www.youtube.com/watch?v=CRge_cZ2cJc) to learn LCS basics.

#### lcs_N 
* **Description:** specifies the maximum rule-population size for an LCS algorithm (i.e. eLCS,XCS,ExSTraCS)
* **Format:** (int)
* **Values:** a positive integer > `50`
* **Tips:** LCS algorithms learn a population (i.e set) of rules that collectively constitute the learned model. When this parameter is larger, LCS will take longer to run. However, LCS algorithms require a larger rule-population to solve more complex problems or analyze larger datasets. For most users we recommend the default value of `2000` as a starting point, however, as a key run parameter, a larger rule-population is typically expected to improve LCS algorithm performance. Watch this [video](https://www.youtube.com/watch?v=CRge_cZ2cJc) to learn LCS basics.

#### lcs_timeout 
* **Description:** similar to `timeout`, this [Optuna](https://optuna.org/) parameter controlling the total number of *seconds* until an LCS algorithm hyperparameter sweep stops running new trials. LCS uses a separate run parameter for this since it can take alot longer to run an LCS hyperparameter sweep.
* **Format:** (int, or `None`)
* **Values:** any positive integer > `1`, (`1200` by default, i.e. 20 minutes), or `None`
* **Tips:** To ensure STREAMLINE reproducibility, this parameter must be set to `None` if `do_lcs_sweep = True`, however this will force LCS algorithms to fully complete the number of trials specified by `n_trials`. When set to an integer, Optuna will submit new trials (as previous ones complete), up until this time limit, and then only use the hyperparameter sweep trials it has completed to pick the best hyperparameter settings for the given LCS algorithm. Any trial already started after this time limit is reached, will continue to run until completion. This means that one LCS algorithm can spend more total time on hyperparameter trials than another, when this parameter is given a time limit.

#### model_resubmit
* **Description:** boolean flag telling STREAMLINE that this is a secondary run attempt of phase 5 (i.e. modeling)
* **Format:** [Command Line Argument] just use flag (i.e. `--do-report`), [Configuration File] (bool) 
* **Values:** `True` or `False`
* **Tips:** set this parameter to `True` either because (1) one of the previous model training jobs timed-out, or failed and the user wants to re-submit them or (2) the user had previously run phase 5 on a subset of available algorithms, but now they'd like to run additional algorithms

***
### Post-Analysis Parameters (Phase 6)

#### exclude_plots
* **Description:** allows users to exclude some of the outputs automatically generated by STREAMLINE during phase 6 (post-analysis)
* **Format:**
    1. for notebook or config file modes: provide a (list) of valid options (str), e.g. `['plot_ROC','plot_PRC']`
    2. for command line arguments: provide as a list of comma separated values with no spaces, e.g. `plot_ROC,plot_PRC`
* **Values:** `None`, or [`'plot_ROC'`, `'plot_PRC'`, `'plot_FI_box'`, or `'plot_metric_boxplots'`] - provided in format above
    * `plot_ROC` - don't output ROC plots individually for each algorithm including all CV results and averages
    * `plot_PRC` - don't output PRC plots individually for each algorithm including all CV results and averages
    * `plot_FI_box` - don't output model feature importance boxplots for each algorithm
    * `plot_metric_boxplots` - don't output evaluation metric boxplots for each metric comparing algorithm performance

#### metric_weight
* **Description:** the evaluation metric used to weigh model feature importance estimates in the composite feature importance plots
* **Format:** (str)
* **Values:** `balanced_accuracy` or `roc_auc`
* **Tips:** we recommend setting the this parameter the same as `primary_metric` if possible

#### top_model_fi_features
* **Description:** the number of top scoring features (based on model feature importance estimates) to illustrate in feature importance figures (i.e. feature importance boxplots, and composite feature importance plots)
* **Format:** (int)
* **Values:** an integer between `10` and `40` is recommended
* **Tips:** 

***
### Replication Parameters (Phase 8)

#### exclude_rep_plots
* **Description:** allows users to exclude some of the outputs automatically generated by STREAMLINE during phase 8 (replication)
* **Format:**
    1. for notebook or config file modes: provide a (list) of valid options (str), e.g. `['plot_ROC', 'plot_PRC']`
    2. for command line arguments: provide as a list of comma separated values with no spaces, e.g. `plot_ROC,plot_PRC`
* **Values:** `None`, or [`'feature_correlations'`,`'plot_ROC'`, `'plot_PRC'`, or `'plot_metric_boxplots'`] - provided in format above
    * `feature_correlations` - don't output feature correlation heatmaps for the replication datasets during replication EDA
    * `plot_ROC` - don't output ROC plots individually for each algorithm including all CV results and averages
    * `plot_PRC` - don't output PRC plots individually for each algorithm including all CV results and averages
    * `plot_metric_boxplots` - don't output evaluation metric boxplots for each metric comparing algorithm performance

***
### Cleanup Parameters

#### del_time
* **Description:** boolean flag telling STREAMLINE to delete individual runtime files from the output experiment folder
* **Format:** [Command Line Argument] just use flag (i.e. `--do-report`), [Configuration File] (bool) 
* **Values:** `True` or `False`

#### del_old_cv
* **Description:** boolean flag telling STREAMLINE to delete intermediary cross validation datasets (i.e. training and testing datasets prior to completed data processing, imputation, scaling, and feature selection) form the output experiment folder
* **Format:** [Command Line Argument] just use flag (i.e. `--do-report`), [Configuration File] (bool) 
* **Values:** `True` or `False`
* **Tips:** this parameter is only relevant if [overwrite_cv](#overwrite-cv) was set to `False`

***
### Multiprocessing Parameters

#### run_parallel
* **Description:** indicates whether or not to run STREAMLINE in parallel (locally) with CPU core multiprocessing
* **Format:** (bool)
* **Values:** `True` or `False`
* **Tips:** this parameter is only relevant when `run_cluster = False`

#### run_cluster
* **Description:** indicates whether or not to run STREAMLINE on an dask-compatible computing cluster (HPC)
* **Format:**  (bool or str)
* **Values:** `False`, or a string identifying the cluster type from options below:
    * `LSF` - LSFCluster
    * `SLURM` - SLURMCluster
    * `HTCondor` - HTCondorCluster
    * `Moab` - MoabCluster
    * `OAR` - OARCluster
    * `PBS` - PBSCluster
    * `SGE` - SGECluster
    * `UGE` - SGECluster variant used at our institution
    * `Local` - LocalCluster
    * `SLURMOld` - Legacy job submission for SLURMCluster
    * `LSFOld` - Legacy job submission for LSFCluster 
* **Tips:** The default of `"SLURM"` is specific to our institutions HPC hardware/software, and may not be relevant to many users

#### reserved_memory
* **Description:** the memory (in Gigabytes) reserved for STREAMLINE jobs
* **Format:** (int)
* **Values:** an integer generally > `1` or < the maximum memory available for an HPC job on your system (consult your cluster documentation or administrator)

#### queue 
* **Description:** indiates the queue within your HPC where your STREAMLINE jobs will be scheduled to run
* **Format:**  (str)
* **Values:** any viable str name for a queue you have access to at your institution 
* **Tips:** The default of `"defq"` is specific to our institutions HPC hardware/software, and may not be relevant to many users

***
### Logging Parameters

#### verbose
* **Description:** boolean flag telling STREAMLINE to send all print output and warnings to the command line output
* **Format:** [Command Line Argument] just use flag (i.e. `--verbose`), [Configuration File] (bool) 
* **Values:** `True` or `False`

#### logging_level
* **Description:** boolean flag telling STREAMLINE what loggin level to use in the command line output
* **Format:** [Command Line Argument] just use flag (i.e. `--logging-level`), [Configuration File] (bool) 
* **Values:** `True` or `False`

***
## Guidelines for Setting Parameters

### Ensuring Output Reproducibility
STREAMLINE is completely reproducible when the [timeout](#timeout) parameter is set to `None`, and. This also assumes that STREAMLINE is being run on the same datasets, with the same run parameters (including `random_seed`). 

When [timeout](#timeout) is *not* set to `None`, STREAMLINE output can sometimes vary slightly (particularly when parallelized) since Optuna (for hyperparameter optimization) may not complete the same number of optimization trials within the user specified time limit on different
computing resources. 

However, having a [timeout](#timeout) value specified helps ensure STREAMLINE run completion within a reasonable time frame.

### Reducing Runtime
Conducting a more effective ML analysis typically demands a much larger amount of computing power and runtime. However, we provide general guidelines here for limiting overall runtime of a STREAMLINE experiment.
1. Run on a fewer number of [datasets](#dataset_path) at once.
2. Run using fewer ML [algorithms](#algorithms) at once:
    * Naive Bayes, Logistic Regression, and Decision Trees are typically fastest.
    * Genetic Programming, eLCS, XCS, and ExSTraCS often take the longest (however other algorithms such as SVM, KNN, and ANN can take even longer when the number of instances is very large).
3. Run using a smaller number of [cv_partitions](#cv-partitions).
4. Run without generating additional plots (see [exclude_eda_output](#exclude-eda-output), [export_hyper_sweep_plots](#export-hyper-sweep-plots),[exclude_plots](#exclude-plots), [exclude_rep_plots](#exclude-rep-plots)).
5. In large datasets with missing values, set [multi_impute](#multi-impute) to `False`. This will apply simple mean imputation to numerical features instead.
6. Set [use_TURF](#use-turf) as `False`. However we strongly recommend setting this to `True` in feature spaces > 10,000 in order to avoid missing feature interactions during feature selection.
7. Set [TURF_pct](#turf-pct) no lower than 0.5.  Setting at 0.5 is by far the fastest, but it will operate more effectively in very large feature spaces when set lower.
8. Set [instance_subset](#instance-subset) at or below `2000` (speeds up multiSURF feature importance evaluation at potential expense of performance).
9. Set [max_features_to_keep](#max-features-to-keep) at or below `2000` and [filter_poor_features](#filter-poor-features) = `True` (this limits the maximum number of features that can be passed on to ML modeling).
10. Set [training_subsample](#training-subsample) at or below `2000` (this limits the number of sample used to train particularly expensive ML modeling algorithms). However avoid setting this too low, or ML algorithms may not have enough training instances to effectively learn.
11. Set [n_trials](#n-trials) and/or [timeout](#timeout) to lower values (this limits the time spent on hyperparameter optimization).
12. If using eLCS, XCS, or ExSTraCS, set [do_lcs_sweep](#do-lcs-sweep) to `False`, [lcs_iterations](#lcs_iterations) at or below `200000`, and [lcs_n](#lcs-n) at or below `2000`.

### Improving Modeling Performance
* Generally speaking, the more computational time you are willing to spend on ML, the better the results. Doing the opposite of the above tips for reducing runtime, will likely improve performance.
* In certain situations, setting [filter_poor_features](#filter-poor-features) to `False`, and relying on the ML algorithms alone to identify relevant features can possibly yield better performance. However, this may only be computationally practical when the total number of features in an original dataset is smaller (e.g. under 2000).
* Note that eLCS, XCS, and ExSTraCS are newer algorithm implementations developed by our research group.  As such, their algorithm performance may not yet be optimized in contrast to the other well established and widely utilized options. These learning classifier system (LCS) algorithms are unique however, in their ability to model very complex associations in data, while offering a largely interpretable model made up of simple, human readable IF:THEN rules. They have also been demonstrated to be able to tackle both complex feature interactions as well as heterogeneous patterns of association (i.e. different features are predictive in different subsets of the training data).
* In problems with no noise (i.e. datasets where it is possible to achieve 100% testing accuracy), LCS algorithms (i.e. eLCS, XCS, and ExSTraCS) perform better when [lcs_nu](#lcs-nu) is set larger than `1` (i.e. `5` or `10` recommended).  This applies significantly more pressure for individual rules to achieve perfect accuracy.  In noisy problems this may lead to significant overfitting.

### Other Guidelines
* SVM and ANN modeling should only be applied when data scaling is applied by the pipeline.
* Logistic Regression' baseline model feature importance estimation is determined by the exponential of the feature's coefficient. This should only be used if data scaling is applied by the pipeline. Otherwise [use_uniform_fi](#use_uniform_fi) should be `True`.
* While the STREAMLINE includes [impute_data](#impute-data) as an option that can be turned off in phase 2, most algorithm implementations (all those standard in scikit-learn) cannot handle missing data values with the exception of eLCS, XCS, and ExSTraCS. In general, STREAMLINE is expected to fail with an errors if run on data with missing values, while [impute_data](#impute-data) is set to `False`.
