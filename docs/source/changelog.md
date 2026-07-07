# Changelog

This page summarizes the major STREAMLINE release lines in newest-first order.
The v3 line is the main release line for this documentation site. The v2 line
is the previous tested and stable public version. The v1 line covers the
original legacy implementation.

Release tags are listed as anchors so users can map documentation language back
to GitHub releases.

## v3 Main Release

**Tag reference:** `v3.0.0` when the v3 release is tagged on `main`.

STREAMLINE v3 is the current main release. It refactors STREAMLINE into an
eleven-phase architecture and extends the pipeline beyond the previous
binary-classification-centered workflow.

Major changes:

* Supports binary classification, multiclass classification, and regression.
* Uses explicit P1-P11 phase modules for data processing, imputation/scaling,
  feature learning, feature importance, feature selection, modeling,
  classification ensembles, summary statistics, dataset comparison,
  replication, and reporting.
* Adds config-driven full-pipeline execution with parameter names aligned
  across `.cfg` files, command-line calls, and notebooks.
* Adds updated Google Colab and local Jupyter workflows for binary,
  multiclass, regression, and custom datasets.
* Adds UCI-based demo datasets and matching replication splits for tutorial
  and test runs.
* Adds optional SMOTE and SMOTENC balancing after Phase 2 imputation/scaling.
* Improves categorical feature handling, including native categorical bypass
  controls for compatible models.
* Adds Decision Tree classification, HEROS wrappers, optional TabPFN wrappers,
  and updated ExSTraCS categorical initialization.
* Updates scikit-rebate handling for categorical feature indexes and adds
  MultiSWRFDB and MultiSWRFDB* feature-importance methods.
* Improves Phase 5 feature selection so rankings can be combined across more
  than two feature-importance algorithms.
* Adds `Parallel` local multiprocessing-style run mode alongside serial,
  local Dask, and supported cluster modes.
* Adds saved run-command metadata so reports can show the parameters actually
  used while preserving defaults when a value was not explicitly specified.
* Improves standard and replication PDF report first pages, dataset summaries,
  metric highlighting, categorical handling text, feature-learning/selection
  summaries, and replication naming.
* Adds documentation and optional pytest coverage for TabPFN token setup and
  skip behavior.

## v2 Tested Stable Release

**Tag references:** `v0.3.0-beta`, `v0.3.1-beta`, `v0.3.2-beta`,
`v0.3.3-beta`, `v0.3.4-beta`.

The v2 public beta line is the previous tested and stable STREAMLINE version.
Its latest release was `v0.3.4-beta`.

Notable updates across the v2 line:

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

## v1 Legacy Release

**Tag references:** `v0.1.0-alpha`, `v0.1.1-alpha`, `v0.1.2-alpha`,
`v0.1.3-alpha`, `v0.2.0-beta`, `v0.2.1-beta`, `v0.2.2-beta`,
`v0.2.3-beta`, `v0.2.4-beta`, `v0.2.5-beta`.

The v1 line covers the first public STREAMLINE implementation and early beta
stabilization work.

Notable updates across the v1 line:

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
