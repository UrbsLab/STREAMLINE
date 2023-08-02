# Run Parameters
Here we review the run parameters available across the 9 phases of STREAMLINE. We begin with a quick guide/summary of all run parameters according to run mode along with their default values (when applicable). Then we provide further descriptions, formatting, valid values, and guidance (as needed) for each run parameter. Lastly, we provide overall guidance on setting STEAMLINE run parameters. 

## Quick Guide
The quick guide below distinguishes essential from non-essential run parameters within streamline, and further breaks down non-essential run paramters by pipeline phase. The name of each parameter is given for the command-line, configuration file, and notebooks (same for both Colab and Jupyter Notebooks), as well as the internal STREAMLINE default value (which ocassionally differ from the default values used in the notebooks for the [demonstration datasets](data.md#demonstration-data)). 
* Run parameters without default values are incidated with 'no default'. 
* Run parameters that are not used in one of the run modes are indicated with 'NA'.

### Essential Parameters (Phase 1-9)

| Command-line Parameter    | Config File Parameter                                   | Notebook Parameter                           | Default    |
|---------------------------|---------------------------------------------------------|----------------------------------------------|------------|
| --data-path               | [dataset_path](#dataset-path)                           | data_path                                    | no default |
| --out-path                | [output_path](#output-path)                             | output_path                                  | no default |
| --exp-name                | [experiment_name](#experiment-name)                     | experiment_name                              | no default |
| --class-label             | [class_label](#class-label)                             | class_label                                  | 'Class'    |
| --inst-label              | [instance_label](#instance-label)                       | instance_label                               | None       |
| --match-label             | [match_label](#match_label)                             | match_label                                  | None       |
| --fi                      | [ignore_features_path](#ignore_features_path)           | ignore_features                              | None       |
| --cf                      | [categorical_feature_path](#categorical_feature_path)   | categorical_feature_headers                  | None       |
| --qf                      | [quantitative_feature_path](#quantitative_feature_path) | quantitiative_feature_headers                | None       |
| --rep-path                | [rep_data_path](#rep_data_path)                         | rep_data_path                                | no default |
| --dataset                 | [dataset_for_rep](#dataset_for_rep)                     | dataset_for_rep                              | no default |
| [--config](#--config)                  | NA                                         | NA                                           | no default |
| --do-till-report or --dtr | [do_till_report](#do_till_report)                       | NA                                           | False      |
| --do-eda                  | [do_eda](#do_eda)                                       | NA                                           | False      |
| --do-dataprep             | [do_dataprep](#do_dataprep)                             | NA                                           | False      |
| --do-feat-imp             | [do_feat_imp](#do_feat_imp)                             | NA                                           | False      |
| --do-feat-sel             | [do_feat_sel](#do_feat_sel)                             | NA                                           | False      |
| --do-model                | [do_model](#do_model)                                   | NA                                           | False      |
| --do-stats                | [do_stats](#do_stats)                                   | NA                                           | False      |
| --do-compare-dataset      | [do_compare_dataset](#do_compare_dataset)               | NA                                           | False      |
| --do-report               | [do_report](#do_report)                                 | NA                                           | False      |
| --do-replicate            | [do_replicate](#do_replicate)                           | NA                                           | False      |
| --do-rep-report           | [do_rep_report](#do_rep_report)                         | NA                                           | False      |
| --do-cleanup              | [do_cleanup](#do_cleanup)                               | NA                                           | False      |
| NA                        | NA                                                      | [applyToReplication](#applyToReplication)    | True       |
| NA                        | NA                                                      | [demo_run](#demo_run)                        | True       |
| NA                        | NA                                                      | [use_data_prompt](#use_data_prompt) (Colab)  | True       |

### Non-Essential Parameters 
#### General Parameters (Phase 1)

| Command-line Parameter    | Config File Parameter                     | Notebook Parameter                | Default      |
|---------------------------|-------------------------------------------|-----------------------------------|--------------|
| --cv                      | [cv_partitions](#cv_partitions)           | n_splits                          | 10           |
| --part                    | [partition_method](#partition_method)     | partition_method                  | 'Stratified' |
| --cat-cutoff              | [categorical_cutoff](#categorical_cutoff) | categorical_cutoff                | 10           |
| --sig                     | [sig_cutoff](#sig_cutoff)                 | sig_cutoff                        | 0.05         |
| --rand-state              | [random_state](#random_state)             | random_state                      | 42           |
| --verbose                 | [verbose](#verbose)                       | NA                                | False        |

#### Data Processing Parameters (Phase 1)

| Command-line Parameter    | Config File Parameter                                            | Notebook Parameter                | Default    |
|---------------------------|------------------------------------------------------------------|-----------------------------------|------------|
| --exclude-eda-output      | [exclude_eda_output](#exclude_eda_output)                        | exclude_eda_output                | None       |
| --top-uni-feature         | [top_uni_features](#top_uni_features)                            | top_uni_features                  | 20         |
| --feat_miss               | [featureeng_missingness](#featureeng_missingness)                | featureeng_missingness            | 0.5        |
| --clean_miss              | [cleaning_missingness](#cleaning_missingness)                    | cleaning_missingness              | 0.5        |
| --corr_thresh             | [correlation_removal_threshold](#correlation_removal_threshold)  | correlation_removal_threshold     | 1.0        |


#### Scaling and Imputation Parameters (Phase 2)

| Command-line Parameter | Config File Parameter | Notebook Parameter          | Default |
|------------------------|-----------------------|-----------------------------|---------|
| --scale                | scale_data            | scale_data                  | True    |
| --impute               | impute_data           | impute_data                 | True    |
| --multi-impute         | multi_impute          | multi_impute                | True    |
| --over-cv              | overwrite_cv          | overwrite_cv                | True    |

#### Feature Importance Estimation Parameters (Phase 3)

| Command-line Parameter | Config File Parameter | Notebook Parameter                            | Default |
|------------------------|-----------------------|-----------------------------------------------|---------|
| --do-mi                | do_mutual_info        | do_mutual_info                                | True    |
| --do-ms                | do_multisurf          | do_multisurf                                  | True    |
| --use-turf             | use_turf              | use_TURF                                      | False   |
| --turf-pct             | turf_pct              | TURF_pct                                      | 0.5     |
| --inst-sub             | instance_subset       | instance_subset                               | 2000    |
| --n-jobs               | n_jobs                | cores                                         | 1       |

#### Feature Selection Parameters (Phase 4)

| Command-line Parameter | Config File Parameter | Notebook Parameter                            | Default |
|------------------------|-----------------------|-----------------------------------------------|---------|
| --filter-feat          | filter_poor_features  | filter_poor_features                          | True    |
| --max-feat             | max_features_to_keep  | max_features_to_keep                          | 2000    |
| --top-features         | top_features          | top_features - UPDATE                         | 40      |
| --export-scores        | export_scores         | export_scores                                 | True    |
| --over-cv-feat         | overwrite_cv_feat     | UPDATE REMOVE                                 | True    |

#### Modeling Parameters (Phase 5)
 Command-line Parameter  | Config File Parameter    | Notebook Parameter                 | Default                   |
|------------------------|--------------------------|------------------------------------|---------------------------|
| --algorithms           | algorithms               | algorithms                         | 'LR,DT,NB'                |
| --exclude              | exclude                  | exclude                            | 'eLCS,XCS'                |
| --metric               | primary_metric           | primary_metric                     | 'balanced_accuracy'       |
| --metric-direction     | metric_direction         | metric_direction                   | 'maximize'                |
| --subsample            | training_subsample       | training_subsample                 | 0                         |
| --use-uniformFI        | use_uniform_fi           | use_uniform_FI                     | True                      |
| --n-trials             | n_trials                 | n_trials                           | 200                       |
| --timeout              | timeout                  | timeout                            | 900                       |    
| --export-hyper-sweep   | export_hyper_sweep_plots | export_hyper_sweep_plots           | False                     |
| --do-LCS-sweep         | do_lcs_sweep             | do_lcs_sweep                       | False                     |
| --nu                   | lcs_nu                   | lcs_nu                             | 1                         |
| --iter                 | lcs_iterations           | lcs_iterations                     | 200000                    |
| --N                    | lcs_n                    | lcs_N                              | 2000                      |
| --lcs-timeout          | lcs_timeout              | lcs_timeout                        | 1200                      |
| --model-resubmit       | model_resubmit           | NA                                 | False                     |

#### Post-Analysis Parameters (Phase 6)
| Command-line Parameter | Config File Parameter | Notebook                                | Default             |
|------------------------|-----------------------|-----------------------------------------|---------------------|
| --plot-ROC             | plot_roc              | plot_ROC                                | True                |
| --plot-PRC             | plot_prc              | plot_PRC                                | True                |
| --plot-FI_box          | plot_fi_box           | plot_metric_boxplots                    | True                |
| --plot-box             | plot_metric_boxplots  | plot_FI_box                             | True                |
| --metric-weight        | metric_weight         | metric_weight                           | 'balanced_accuracy' |
| --top-model-features   | top_model_features    | top_model_features - UPDATE             | 40

#### Compare Datasets Parameters (Phase 7)
There are currently no run parameters to adjust for this phase.

#### Replication Parameters (Phase 8)
| Command-line Parameter | Config File Parameter           | Description     | Default |
|------------------------|---------------------------------|-----------------|---------|
| --rep-export-fc        | rep_export_feature_correlations | NA              | True    |
| --rep-plot-ROC         | rep_plot_roc                    | NA              | True    |
| --rep-plot-PRC         | rep_plot_prc                    | NA              | True    |
| --rep-plot-box         | rep_plot_metric_boxplots        | NA              | True    |

#### Summary Report (Phase 9)
There are currently no run parameters to adjust for this phase.

## Details
This section will go into greater depth for each run parameter, primarily using the configuration file parameter name to identify each. 
* *Parameters identified as (str) format should be entered with single quotation marks within notebooks, or when using a configuration file, but without them when using command line arguments (CLA).* 

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

#### --config
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

### Non-Essential Parameters 
#### General Parameters (Phase 1)

##### cv_partitions
* **Description:** *k*, the number of *k*-fold cross validation training/testing data partitions to create and apply throughout pipeline
* **Format:** (int)
* **Values:** an integer between `3` and `10` is recommended
* **Tips:** smaller values will yield shorter STREAMLINE run times, but training datasets will have a smaller number of instances

##### partition_method 
* **Description:** the cross validation strategy used
* **Format:** (str)
* **Values:** `'Stratified'`, `'Random'`, or `'Group'`
* **Tips:** `'Stratified'` is generally recommended in order to keep class balance as similar as possible within respective partitions, however `'Group'` can be selected when `match_label` has been specified to keep instances with the same match/group ID together within a respective partition

##### categorical_cutoff 
* **Description:** the number of unique values observed for a given feature in a 'target dataset' after which a variable is automatcially considered to be quantitative
* **Format:** (int)
* **Values:** an integer between `3` and `10` is generally recommended, but should be set in a dataset-specific manner
* **Tips:** this parameter will only be used if the user hasn't specifically indicated which features to treat as categorical or quantitative using `categorical_feature_path` and/or `quantiative_feature_path`, respectively. However depending on the specific dataset, users can sometimes conveniently set this parameter to correctly assign variable types, e.g. if all categorical features in the dataset have fewer than 5 unique values, but quantitative ones all have more than 10 unique values, setting `categorical_cutoff = 7` will make correct feature type assignments automatically.

##### sig_cutoff 
* **Description:** the statistical significance cutoff used throughout the pipeline used in deciding whether to run pair-wise non-parametric statistical comparisons following group comparisons, and for identifying significant results in output files with a '*'
* **Format:** (float)
* **Values:** a value <= 0.05 is recommended
* **Tips:** Note: STREAMLINE does not currently automatically account for multiple testing - users should take this into consideration themselves

##### random_state 
* **Description:** sets a specific random seed for the STREAMLINE run (important for pipeline reproducibility)
* **Format:** (int) 
* **Values:** any positive integer value is fine
* **Tips:** make sure to use the same value for `random_state` in a separate run along with the same datasets and run parameters to obtain reproducible pipeline results

##### verbose
* **Description:** boolean flag telling STREAMLINE to send all print output and warnings to the command line output
* **Format:** [Command Line Argument] just use flag (i.e. `--verbose`), [Configuration File] (bool) 
* **Values:** `True` or `False`

#### Data Processing Parameters (Phase 1)

##### exclude_eda_output
* **Description:** allows users to exclude some of the outputs automatically generated by STREAMLINE during phase 1
* **Format:** 
    1. for notebook or config file modes: provide a (list) of (str) feature names that can be found in any of the 'target datasets', e.g. `['describe','univariate_plots','correlation_plots']`
    2. for command line arguments: provide as a list of comma separated values with no spaces, e.g. `describe,univariate_plots,correlation_plots`
* **Values:** `None`, or [`'describe'`, `'univariate_plots'`, or `'correlation_plots'`] - provided in format above
    * `describe` - don't run or output the set of standard pandas functions (i.e. `Describe()`, `Dtypes()`, and `nunique()`) as `.csv` files
    * `univariate_plots` - don't output individual univariate analysis plots illustrating features vs. outcome (by default STREAMLINE outputs these plots for any feature with a significant univariate association based on `sig_cutoff`)
    * `correlation_plots` - don't output feature correlation heatmaps for the 'initial' or 'processed' data EDA

##### top_uni_features
* **Description:** number of most significant features to report in the notebook and PDF summary
* **Format:** (int)
* **Values:** an integer between 10 and 40 is recommended

##### featureeng_missingness
* **Description:** the proportion of missing values within a feature (*above which*) a new binary categorical feature is generated that indicates if the value for an instance was missing or not
* **Format:** (float)
* **Values:** (0.0-1.0)
* **Tips:** this parameter controls automated feature engineering of a new 'missingness' feature, generated for another pre-existing feature in the 'target dataset'. It's useful for identifying the potentially predictive value of any feature who's missingness is not completely at random (NCAR)

##### cleaning_missingness
* **Description:** the proportion of missing values, within a feature or instance, (*at which*) the given feature or instance will be automatically cleaned (i.e. removed) from the processed 'target dataset'
* **Format:** (float)
* **Values:** (0.0-1.0)
* **Tips:** this parameter controls automated data cleaning based on feature or instance 'missingness'. STREAMLINE will first remove features with high missingness, then subsequently remove any instances with missingness over this proportion.

##### correlation_removal_threshold
* **Description:** the (pearson) feature correlation at which one out of a pair of features is randomly removed from the processed 'target dataset'
* **Format:** (float)
* **Values:** (0.0-1.0)
* **Tips:** this parameter controls automated data cleaning based on feature correlation. The safest setting (to avoid missing predictive information) is the default of 1.0 (i.e. perfect correlation between two features). Note: STREAMLINE interprets this parameter as both a positive and negative correlation threshold.

#### Scaling and Imputation Parameters (Phase 2)

##### scale_data
* **Description:** 
* **Format:**
* **Values:** 
* **Tips:** 

##### impute_data
* **Description:** 
* **Format:**
* **Values:** 
* **Tips:** 

##### multi_impute
* **Description:** 
* **Format:**
* **Values:** 
* **Tips:** 


##### m
* **Description:** 
* **Format:**
* **Values:** 
* **Tips:** 

| Command-line Parameter | Config File Parameter | Notebook Parameter          | Default |
|------------------------|-----------------------|-----------------------------|---------|
| --scale                | scale_data            | scale_data                  | True    |
| --impute               | impute_data           | impute_data                 | True    |
| --multi-impute         | multi_impute'         | multi_impute                | True    |
| --over-cv              | overwrite_cv          | overwrite_cv                | True    |

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

### Reducing runtime
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

### Improving Modeling Performance
* Generally speaking, the more computational time you are willing to spend on ML, the better the results. Doing the opposite of the above tips for reducing runtime, will likely improve performance.
* In certain situations, setting `feature_selection` to 'False', and relying on the ML algorithms alone to identify relevant features will yield better performance.  However, this may only be computationally practical when the total number of features in an original dataset is smaller (e.g. under 2000).
* Note that eLCS, XCS, and ExSTraCS are newer algorithm implementations developed by our research group.  As such, their algorithm performance may not yet be optimized in contrast to the other well established and widely utilized options. These learning classifier system (LCS) algorithms are unique however, in their ability to model very complex associations in data, while offering a largely interpretable model made up of simple, human readable IF:THEN rules. They have also been demonstrated to be able to tackle both complex feature interactions as well as heterogeneous patterns of association (i.e. different features are predictive in different subsets of the training data).
* In problems with no noise (i.e. datasets where it is possible to achieve 100% testing accuracy), LCS algorithms (i.e. eLCS, XCS, and ExSTraCS) perform better when `nu` is set larger than 1 (i.e. 5 or 10 recommended).  This applies significantly more pressure for individual rules to achieve perfect accuracy.  In noisy problems this may lead to significant overfitting.

### Other Guidelines
* SVM and ANN modeling should only be applied when data scaling is applied by the pipeline.
* Logistic Regression' baseline model feature importance estimation is determined by the exponential of the feature's coefficient. This should only be used if data scaling is applied by the pipeline.  Otherwise `use_uniform_FI` should be True.
* While the STREAMLINE includes `impute_data` as an option that can be turned off in `DataPreprocessing`, most algorithm implementations (all those standard in scikit-learn) cannot handle missing data values with the exception of eLCS, XCS, and ExSTraCS. In general, STREAMLINE is expected to fail with an errors if run on data with missing values, while `impute_data` is set to 'False'.

## NOTES FOR RECYCLING


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
