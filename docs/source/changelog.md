# Changelog

This page summarizes the major STREAMLINE release anchors in newest-first order.
STREAMLINE v1.0.0 is the current main release documented by this site. STREAMLINE
v0.3.4 is the previous tested and stable public release. STREAMLINE v0.2.5 is the
legacy release anchor for the original implementation line.

For smaller patch-level and development notes, see the official
[GitHub Releases](https://github.com/UrbsLab/STREAMLINE/releases) page.

## v1.0.0 Main Release

**Tag reference:** `v1.0.0` when this release is tagged on `main`.

STREAMLINE v1.0.0 is a major reorganization, expansion, and modernization of the
pipeline. It keeps STREAMLINE's original goal of transparent end-to-end tabular
AutoML, while broadening the supported tasks, making phases easier to extend,
and refreshing the notebooks, tests, reports, and documentation around the new
workflow.

### Architecture And Run Workflow

* Refactored STREAMLINE into explicit P1-P11 phase modules: data processing,
  imputation/scaling/balancing, feature learning, feature importance, feature
  selection, modeling, classification ensembles, summary statistics, dataset
  comparison, replication, and reporting.
* Added config-driven full-pipeline execution with `.cfg` files, while keeping
  phase-by-phase command-line and notebook workflows available.
* Standardized parameter names across config files, command-line arguments,
  notebooks, and saved run-command metadata.
* Added run-command pickle support so repeated phase calls can reuse the
  parameters that were actually used, while allowing users to override or ignore
  saved values when needed.
* Added `Parallel` local multiprocessing-style execution in addition to
  `Serial`, local Dask through `Local`, and supported cluster submission modes.
* Reorganized code around registries so users can add or swap methods more
  cleanly across phases, including Phase 2 imputers/scalers, Phase 3 learners,
  Phase 4 feature-importance methods, Phase 5 selectors, Phase 6 models, and
  Phase 7 ensembles.

### Binary, Multiclass, And Regression Support

* Extended STREAMLINE beyond the earlier primarily binary-classification
  workflow to support binary classification, multiclass classification, and
  regression as first-class task types.
* Added UCI-based demo datasets and matching held-out replication splits for
  binary classification, multiclass classification, and regression tutorials and
  tests.
* Updated P1 task handling so user-specified `outcome_type` is respected instead
  of being re-inferred only from the number of outcome values.
* Updated P6 model loading and evaluation around task-specific model registries
  for binary, multiclass, and regression runs.
* Added multiclass evaluation support, including macro/micro metrics and
  multiclass ROC/PR curve summaries where applicable.
* Added regression evaluation support, including regression metrics such as
  explained variance, Pearson correlation, MAE, MSE, median absolute error, max
  error, residual outputs, and actual-vs-predicted style reporting.
* Updated reports and summary statistics so metric names, no-skill baselines,
  curves, and plots are task-aware instead of assuming binary classification.
* Clarified that P7 ensemble modeling is currently classification-only, so
  regression workflows proceed from P6 modeling to P8 summary statistics.

### Data Processing, Feature Types, And Preprocessing

* Added clearer feature-type handling for categorical and quantitative features,
  including optional user-supplied feature type files and inferred feature types.
* Added optional SMOTE/SMOTENC balancing after Phase 2 imputation and scaling,
  with SMOTENC used when categorical features are present.
* Added support for bypassing one-hot encoding when using models that can handle
  categorical features natively.
* Added explicit native-categorical model controls so unsupported models are
  rejected when one-hot encoding is disabled instead of silently receiving an
  incompatible representation.
* Kept preprocessing extensible through the Phase 2 registry, so users can add
  scalers such as `MaxAbsScaler` or custom imputers without changing the rest of
  the pipeline.

### Feature Learning, Importance, And Selection

* Added P3 feature learning as a dedicated phase, including PCA-style learned
  feature outputs and manifests.
* Updated replication handling to use the saved training workflow artifacts and
  feature manifests more consistently.
* Updated Phase 4 so feature-importance methods write scores without mutating
  shared CV datasets, avoiding order-dependent outputs and parallel race risks.
* Updated scikit-rebate integration for the current package behavior, including
  passing STREAMLINE's categorical feature indexes to ReBATE methods.
* Added MultiSWRFDB and MultiSWRFDB* feature-importance methods.
* Updated default feature-importance/selection behavior around mutual
  information, MultiSWRFDB, and MultiSWRFDB*.
* Improved Phase 5 feature selection so rankings can be combined across all
  feature-importance methods that were run, not only a fixed pair of methods.
* Added or restored `instance_subset` support for expensive feature-importance
  methods so large runs can be controlled from config/CLI parameters.

### Modeling And Native Categorical Algorithms

* Reworked Phase 6 modeling around clearer dataset/model/CV execution units for
  serial, parallel, Dask, and cluster runs.
* Added Decision Tree support for classification workflows.
* Added or updated HEROS wrappers and optional TabPFN wrappers.
* Added TabPFN token handling: if `TABPFN_TOKEN` is not set, requested TabPFN
  models are skipped with a warning while HEROS and other requested models
  continue.
* Updated ExSTraCS categorical initialization so categorical and continuous
  attributes can be passed through its supported `discrete_attribute_limit` and
  `specified_attributes` parameters.
* Improved native categorical model handling for compatible algorithms such as
  CatBoost/CGB and ExSTraCS.
* Added Optuna trial accounting so reports can show how many trials actually ran
  within the requested `n_trials` and `timeout` budgets.
* Standardized model defaults across config, command-line, and notebook run
  modes, including removal of deprecated/default-only legacy methods where
  appropriate.

### Reporting, Replication, And Outputs

* Added standard and replication PDF report improvements, including clearer
  first-page summaries, experiment names, resolved/default parameter display,
  and report-mode-specific text.
* Added dedicated replication report naming and clearer replication report
  content so replication outputs are not confused with standard CV reports.
* Improved report language around categorical handling, scaling/imputation,
  feature learning, feature selection, and metric interpretation.
* Added feature-learning and feature-selection summary tables modeled after the
  data-processing/feature-engineering summary style.
* Improved feature-importance figure layout, including more square plots and
  clearer ordering emphasis.
* Updated metric highlighting to use shading rather than only bold text.
* Added no-skill ROC/PR legend notes and label-aware baseline handling for
  multiclass or non-stratified/random CV settings.
* Improved output organization and report data JSON so reports are easier to
  debug and regenerate.

### Notebooks, Documentation, Tests, And Release Readiness

* Updated the Google Colab notebook and local Jupyter notebook for the v1.0.0
  workflow, including parameter blocks for binary, multiclass, regression, and
  custom dataset runs.
* Rebuilt the documentation website around the v1.0.0 workflow as the new main
  documentation site.
* Added Sphinx/autodoc documentation build support.
* Added GitHub pytest workflows for Python 3.10, 3.11, and 3.12. Python 3.9 was
  skipped because TabPFN does not support it.
* Added optional TabPFN-specific pytest coverage for no-token skip behavior and
  token-gated wrapper fitting.
* Added macOS installation guidance for conda-forge compiled dependencies such
  as Graphviz, WeasyPrint/Cairo/Pango, LightGBM, XGBoost, and CatBoost.
* Removed or de-emphasized old main-era documentation paths and files that are
  no longer part of the maintained v1.0.0 workflow.

## v0.3.4 Tested Stable Release

**Tag references:** `v0.3.0-beta`, `v0.3.1-beta`, `v0.3.2-beta`,
`v0.3.3-beta`, `v0.3.4-beta`.

The v0.3.x public beta line is the previous tested and stable STREAMLINE
version. Its latest release was `v0.3.4-beta`.

Notable updates across the v0.3.x line:

* Added Dask jobqueue support for multiple HPC cluster systems.
* Expanded the original Phase 1 EDA flow into numerical encoding, automated
  cleaning, missingness feature engineering, one-hot encoding, correlation
  feature cleaning, processed-data EDA, and data-process summaries.
* Added configuration-file support and whole-pipeline command-line execution.
* Modularized modeling algorithms into classes and added Elastic Net.
* Improved Google Colab workflows for easier data selection and output access.
* Added replication processing parity for categorical and missingness handling.
* Added invariant feature removal and matching replication behavior.
* Improved PDF report formatting, first-page summaries, run-parameter display,
  and multi-dataset report layout.
* Fixed command-line, legacy cluster, replication, missingness naming, and
  unseen categorical-value edge cases.

Release highlights:

| Tag | Date | Summary |
| --- | --- | --- |
| `v0.3.4-beta` | 2023-09-28 | PDF formatting improvements and legacy replication/report fixes. |
| `v0.3.3-beta` | 2023-09-23 | Invariant feature removal, replication parity fixes, notebook ordering fixes, and report text updates. |
| `v0.3.2-beta` | 2023-09-13 | Legacy command-line argument fixes, job-status docs, schematic and PDF naming updates. |
| `v0.3.1-beta` | 2023-09-07 | Replication imputation fallback and alphabetized model legends. |
| `v0.3.0-beta` | 2023-08-06 | Major command-line, cluster, Phase 1, replication, config, modeling, Colab, and report updates. |

## v0.2.5 Legacy Release

**Tag references:** `v0.1.0-alpha`, `v0.1.1-alpha`, `v0.1.2-alpha`,
`v0.1.3-alpha`, `v0.2.0-beta`, `v0.2.1-beta`, `v0.2.2-beta`,
`v0.2.3-beta`, `v0.2.4-beta`, `v0.2.5-beta`.

The v0.2.x line covers the first public STREAMLINE implementation and early beta
stabilization work. Its latest release was `v0.2.5-beta`.

Notable updates across the v0.2.x line:

* Introduced the first stable STREAMLINE implementation inherited from
  AutoMLPipe-BC concepts.
* Moved the codebase into the `streamline` package folder.
* Updated default Optuna parameters and documented reproducibility limitations
  when parallel optimization uses timeouts.
* Fixed serial Linux command-line execution and several command-line phase
  issues.
* Improved composite feature-importance behavior, feature-selection options,
  report formatting, and Optuna plotting failure handling.
* Added support for replication input as `.csv` or `.txt`.
* Switched feature-importance summary reporting between mean and median during
  early beta refinements based on collaborator feedback.
* Added small statistical-comparison and no-missing-data imputation fixes.

Release highlights:

| Tag | Date | Summary |
| --- | --- | --- |
| `v0.2.5-beta` | 2022-06-24 | Statistical-comparison edge-case catch and cleanup. |
| `v0.2.4-beta` | 2022-06-15 | No-missing-data imputation fix, replication file support, and FI/report summary updates. |
| `v0.2.3-beta` | 2022-05-20 | Stable Linux serial command-line beta. |
| `v0.2.2-beta` | 2022-05-19 | Composite FI, metric weighting, serial CLI, report formatting, and Optuna fixes. |
| `v0.2.1-beta` | 2022-05-17 | Package layout, Optuna default, reproducibility, and scaled-data rounding updates. |
| `v0.2.0-beta` | 2022-05-14 | First beta for external use. |
| `v0.1.x-alpha` | 2022-05-12 | Initial alpha releases and early Anaconda/scipy compatibility fixes. |
