# Run Parameters

STREAMLINE parameters can be supplied through notebooks, `.cfg` files, or
phase CLI flags. The `.cfg` names intentionally match the command-line names
where possible.

## Shared Run Parameters

| Parameter | Typical value | Used by | Description |
| --- | --- | --- | --- |
| `output_path` | `out` | all phases | Parent folder for experiment outputs. |
| `experiment_name` | `UCIHCCPipeline` | all phases | Experiment folder name. |
| `outcome_label` | `Class`, `MPG` | P1, P6, P8, P9, P11 | Outcome column. |
| `outcome_type` | `Binary`, `Multiclass`, `Continuous` | P1, P6, P8, P9, P11 | Learning task type. |
| `instance_label` | `InstanceID` | P1, P6-P11 | Optional row identifier column. |
| `n_splits` | `3`, `5`, `10` | CV-aware phases | Number of CV folds. |
| `run_cluster` | `Serial`, `Local`, `Parallel`, `BashSLURM`, `BashLSF` | all phases | Execution mode. `Local` uses a local Dask cluster; `Parallel` uses local joblib parallelism. |
| `random_state` | `42` | stochastic phases | Seed for reproducibility. |

## Phase Toggles

The `[phases]` section controls which phases run:

```ini
[phases]
phase_order = p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11
do_p1 = True
do_p2 = True
do_p3 = True
do_p4 = True
do_p5 = True
do_p6 = True
do_p7 = True
do_p8 = True
do_p9 = True
do_p10 = True
do_p11 = True
```

The runner also accepts old-style broad flags such as `do_till_report`.

## P1 Data Process

| Parameter | Default or example | Description |
| --- | --- | --- |
| `data_path` | `data/UCIBinaryClassification` | Folder containing one or more input datasets. |
| `categorical_features` | `data/UCIFeatureTypes/hcc_survival_categorical_features.csv` | Optional feature-name file. |
| `quantitative_features` | `data/UCIFeatureTypes/hcc_survival_quantitative_features.csv` | Optional feature-name file. |
| `ignore_features` | empty | Optional feature-name file/list to drop. |
| `partition_method` | `Stratified` or `Random` | CV partitioning strategy. |
| `categorical_cutoff` | `10` | Inference threshold when feature type files are absent. |
| `one_hot_encoding` | `True` | Expand categorical features in P1. |
| `force` | `False` | Overwrite existing phase outputs. |

## P2 Impute And Scale

| Parameter | Default or example | Description |
| --- | --- | --- |
| `imputer_id` | phase default | Registry imputer. |
| `scaler_id` | phase default | Registry scaler. |
| `smote` | `False` | Apply training-fold oversampling after imputation/scaling. |
| `smote_method` | `auto` | Use `SMOTENC` when categorical features are present, otherwise `SMOTE`. |

## P3 Feature Learning

| Parameter | Default or example | Description |
| --- | --- | --- |
| `learner_id` | `pca` | Feature learner registry ID. |
| `learner_params` | `{}` | JSON/Python-literal dictionary of learner parameters. |
| `keep_original_features` | `True` | Keep input features alongside learned features. |

## P4 Feature Importance

| Parameter | Default or example | Description |
| --- | --- | --- |
| `models` | all registered methods | Feature-importance methods to run. |
| `models_params` | method dictionary | Per-method parameter dictionary. STREAMLINE injects ReBATE `categorical_features` from saved feature-type artifacts. |
| `instance_subset` | not used unless provided | Optional sampling limit for expensive methods. |

## P5 Feature Selection

| Parameter | Default or example | Description |
| --- | --- | --- |
| `selector_id` | `default` | Feature selector registry ID. |
| `algorithms` | `auto` | Feature-importance methods considered by selector logic. |
| `top_features` | `20` | Number of features to keep when applicable. |

## P6 Modeling

| Parameter | Default or example | Description |
| --- | --- | --- |
| `outcome_type` | `Binary`, `Multiclass`, `Continuous` | Modeling task. `model_type` is still accepted as a backward-compatible alias. |
| `models` | `NB,LR,DT` | Model registry IDs. |
| `scoring_metric` | `balanced_accuracy`, `explained_variance` | Optuna/evaluation metric. |
| `metric_direction` | `maximize` or `minimize` | Optimization direction. |
| `n_trials` | `200` | Optuna trial budget. |
| `timeout` | `900` | Optuna time budget in seconds. |
| `training_subsample` | `0` | Optional training subset size for models that set `subsampling_allowed=True`, including ANN, SVM, KNN, XGB, and HEROS. |
| `calibrate` | `0` or `1` | Classification calibration toggle. |
| `bypass_one_hot_for_native_models` | `True` | Allow native categorical model path. |
| `native_categorical_models` | `CGB,ExSTraCS` | Models allowed when P1 did not one-hot encode. |

P6 records Optuna trial accounting in model outputs so reports can show how
many trials actually ran within the requested budget.

## P7 Ensembles

P7 is classification-only in the current codebase.

| Parameter | Default or example | Description |
| --- | --- | --- |
| `ensembles` | `hard_voting,soft_voting,stack_lr` | Ensemble registry IDs. |
| `base_models` | `NB,LR,DT` | Base model predictions to combine. |
| `meta_train_source` | `train` | Source for stacking meta-training. |

## P8 To P11

| Phase | Key parameters | Notes |
| --- | --- | --- |
| P8 Summary | `scoring_metric`, `metric_weight`, `top_features`, `include_ensembles` | Aggregates model, ensemble, and feature outputs. |
| P9 Compare | `sig_cutoff`, `show_plots` | Compares datasets in an experiment. |
| P10 Replication | `rep_data_path`, `dataset_for_rep`, `show_plots` | Applies trained workflows to external data. |
| P11 Reporting | `report_modes`, `report_mode`, `make_pdf`, `enable_plots`, `reuse_existing_figures` | Builds standard and replication reports. |

## Saved Run Command Controls

All phase CLIs support:

| Flag | Behavior |
| --- | --- |
| `--ignore_saved_run_command` | Ignore `run_commands.pickle` for this run. |
| `--no_update_saved_run_command` | Do not update `run_commands.pickle` after the run. |
