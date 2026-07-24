# Detailed Pipeline Walkthrough

This page explains what STREAMLINE does during a run, why the phases are
separated, what users can customize, and which outputs to expect. A single
STREAMLINE run is called an **experiment**. Each experiment can contain one or
more datasets, and each dataset is processed through cross-validation folds so
that model evaluation stays separated from model training.

The current v1.0.0 pipeline has eleven phases. P1-P8 are the core training and
summary workflow, P9 compares multiple datasets inside an experiment, P10
applies trained workflows to external replication data, and P11 produces PDF
reports.

## Dataset Terms

| Term | Meaning |
| --- | --- |
| Target dataset | The input dataset supplied to P1 for model development. |
| Training fold | The fold-specific subset used to fit preprocessing, feature learning, feature selection, and models. |
| Testing fold | The fold-specific subset held out for model evaluation. |
| Validation split | The internal split made inside model training for Optuna hyperparameter search. |
| Replication dataset | An external or held-out dataset used in P10 after the main training/CV workflow has already produced fitted artifacts. |

Replication data should not be used to tune model choices. It is meant to
evaluate whether the trained workflow generalizes beyond the original CV test
folds.

## Phase Summary

| Phase | Module | Main CLI | Summary |
| --- | --- | --- | --- |
| P1 | `streamline.p1_data_process` | `python -m streamline.p1_data_process.p1_cli` | Load datasets, clean/encode columns, generate EDA outputs, and create CV folds. |
| P2 | `streamline.p2_impute_scale` | `python -m streamline.p2_impute_scale.p2_cli` | Impute, scale, and optionally apply SMOTE/SMOTENC to training folds. |
| P3 | `streamline.p3_feature_learning` | `python -m streamline.p3_feature_learning.p3_cli` | Add learned features such as PCA components and save fitted transformers/manifests. |
| P4 | `streamline.p4_feature_importance` | `python -m streamline.p4_feature_importance.p4_cli` | Score features without mutating shared CV datasets. |
| P5 | `streamline.p5_feature_selection` | `python -m streamline.p5_feature_selection.p5_cli` | Select informative features and persist selected CV datasets. |
| P6 | `streamline.p6_modeling` | `python -m streamline.p6_modeling.p6_cli` | Train base models, tune with Optuna, evaluate CV metrics, and save predictions. |
| P7 | `streamline.p7_ensembles` | `python -m streamline.p7_ensembles.p7_cli` | Build classification ensembles from base model predictions. |
| P8 | `streamline.p8_summary_statistics` | `python -m streamline.p8_summary_statistics.p8_cli` | Aggregate performance, feature importance, and model summaries. |
| P9 | `streamline.p9_compare_datasets` | `python -m streamline.p9_compare_datasets.p9_cli` | Compare dataset-level outputs within an experiment. |
| P10 | `streamline.p10_replication` | `python -m streamline.p10_replication.p10_cli` | Apply trained workflows to replication datasets. |
| P11 | `streamline.p11_reporting` | `python -m streamline.p11_reporting.p11_cli` | Generate standard and replication PDF reports. |

## P1: Data Process

P1 loads one or more tabular datasets, applies initial cleaning and feature
engineering, records exploratory summaries, and creates cross-validation
train/test folds.

Common work in P1 includes:

* removing instances with missing outcomes
* excluding identifier or user-ignored columns
* applying user-provided or inferred feature types
* adding missingness indicator features when requested
* removing invariant, high-missingness, or highly correlated features
* optionally one-hot encoding categorical features
* creating stratified, random, grouped, or provided CV folds

Important settings include `outcome_label`, `outcome_type`, `instance_label`,
`categorical_features`, `quantitative_features`, `ignore_features`,
`partition_method`, `n_splits`, `one_hot_encoding`, and `force`.

Outputs include dataset summaries, feature-type artifacts, EDA tables/figures,
and the initial `CVDatasets/` train/test files used by later phases.

## P2: Impute, Scale, And Balance

P2 learns missing-value imputation and feature scaling from each training fold
and applies the learned transformations to the corresponding test fold. This
prevents leakage from test data into preprocessing.

P2 can also apply SMOTE/SMOTENC to training folds after imputation and scaling.
This is intended for classification tasks with meaningful class imbalance.
When `smote_method = auto`, STREAMLINE uses SMOTENC when categorical features
are present and standard SMOTE otherwise.

Outputs include transformed CV datasets and saved imputation/scaling metadata.

## P3: Feature Learning

P3 applies feature-learning methods such as PCA. Learned transformations are
fit on training folds and applied consistently to test folds and replication
data.

Users can choose whether learned features replace the original feature set or
are added alongside original features. P3 writes manifests so later phases know
which columns were learned and how to reproduce them.

## P4: Feature Importance

P4 runs filter-style feature-importance methods on each training fold. Current
methods include mutual information and ReBATE-based methods such as MultiSURF,
MultiSURF*, MultiSWRFDB, and MultiSWRFDB*.

P4 writes feature scores only. It does not mutate the shared CV datasets,
which keeps model-specific feature-importance runs independent and avoids
race conditions in parallel execution.

Use `instance_subset` when a feature-importance method would be too slow on
all training instances. ReBATE methods receive categorical feature indexes
from STREAMLINE feature-type artifacts.

## P5: Feature Selection

P5 consumes P4 rankings and creates selected-feature CV datasets. The default
selector can combine rankings from every feature-importance method that was
run, not only a fixed pair of methods.

Important settings include `algorithms`, `selector_id`, `top_features`,
`max_features_to_keep`, and `filter_poor_features`.

Outputs include selected train/test folds and feature-selection summary files.

## P6: Modeling

P6 trains and evaluates base models for binary classification, multiclass
classification, or regression. Model IDs are loaded from the registry for the
selected `outcome_type`.

For models with tunable hyperparameters, STREAMLINE uses Optuna with
`n_trials` and `timeout` budgets. P6 records how many trials actually ran, so
the report can distinguish requested budget from completed search.

P6 also supports native categorical handling. If P1 was run with
`one_hot_encoding = False`, P6 defaults to native categorical models such as
CatBoost/CGB and ExSTraCS. Explicitly requesting an unsupported model raises an
error instead of silently changing the data representation.

Outputs include fitted model pickles, predictions, per-fold metrics, feature
importance estimates, and Optuna trial summaries.

## P7: Ensemble Modeling

P7 builds classification ensembles from P6 base model predictions. Current
ensemble methods include hard voting, soft voting, and logistic-regression
stacking.

P7 is classification-only in the current v1.0.0 implementation. Regression
workflows should skip P7 and continue from P6 to P8.

## P8: Summary Statistics

P8 aggregates performance metrics and feature-importance outputs across folds.
It produces model summary tables, curve plots for classification, regression
diagnostic plots, composite feature-importance plots, and statistical
comparison tables where applicable.

For classification reports, no-skill ROC/PR baselines are computed from the
actual evaluation labels, so the baselines remain appropriate for multiclass
or non-stratified/random CV settings.

## P9: Compare Datasets

P9 compares datasets inside the same experiment. It is useful when an
experiment runs multiple related datasets or feature sets and the user wants
the same statistical summaries and visual comparisons across them.

If an experiment contains only one dataset, P9 may still run but has less to
compare.

## P10: Replication

P10 applies trained preprocessing, feature-learning, feature-selection, and
modeling artifacts to external replication datasets. This phase should be used
for data that were not part of training or CV evaluation.

The replication dataset must contain the required feature columns from the
training dataset schema. Replication outputs are written under each target
dataset's `replication/` folder.

## P11: Reporting

P11 builds standard and replication PDF reports from the experiment outputs.
The standard report focuses on P1-P9 training/CV results. The replication
report focuses on P10 external-validation results and uses a filename that
includes `Replication_Report`.

Each report directory also contains `report_data.json`, which is useful for
debugging report content without parsing the PDF.

## Config Runner

`run.py` wraps `streamline.pipeline.pipeline_cli` and runs one or more phases
from a `.cfg` file:

```bash
python run.py -c run_configs/uci_binary_hcc.cfg --dry_run
python run.py -c run_configs/uci_binary_hcc.cfg
```

Useful partial-run controls:

```bash
python run.py -c run_configs/uci_binary_hcc.cfg --start_at p4
python run.py -c run_configs/uci_binary_hcc.cfg --stop_after p8
python run.py -c run_configs/uci_binary_hcc.cfg --only p6,p8,p11
python run.py -c run_configs/uci_binary_hcc.cfg --skip p3,p4
```

## Saved Run Commands

Each phase records resolved arguments in:

```text
<output_path>/<experiment_name>/run_commands.pickle
```

Later runs reuse saved values for omitted options, while explicitly supplied
command-line values override and update the saved entry. Use
`--ignore_saved_run_command` for a fresh parser/default run and
`--no_update_saved_run_command` to avoid modifying the pickle.
