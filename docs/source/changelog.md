# Changelog

This changelog summarizes notable STREAMLINE changes in newest-first order.
Older public release notes are based on the
[GitHub Releases](https://github.com/UrbsLab/STREAMLINE/releases) entries, with
minor wording cleanup for readability.

## v1.0.0 - Main Release

STREAMLINE v1.0.0 is a major reorganization and expansion of STREAMLINE into a
broader supervised tabular AutoML pipeline. It keeps the original transparent,
end-to-end design while adding first-class support for more task types, clearer
phase boundaries, registry-based extension points, updated notebooks, Sphinx
documentation, and broader testing.

### Added

* Added explicit P1-P11 phase modules for data processing, imputation/scaling
  and balancing, feature learning, feature importance, feature selection,
  modeling, classification ensembles, summary statistics, dataset comparison,
  replication, and reporting.
* Added run-command pickle support so repeated phase calls can reuse the
  parameters actually used in previous phases, while still allowing user
  overrides.
* Added binary classification, multiclass classification, and regression as
  supported task types across core pipeline phases.
* Added optional probability calibration for classification base models and
  ensembles, with configurable calibration method and CV folds.
* Added UCI demo datasets and held-out replication splits for binary,
  multiclass, and regression examples.
* Added optional SMOTE/SMOTENC balancing after Phase 2 imputation and scaling.
* Added P3 feature learning as a dedicated phase, including PCA-style learned
  feature outputs and manifests.
* Added MultiSWRFDB and MultiSWRFDB* feature-importance methods.
* Added expanded task-specific model registries for binary classification,
  multiclass classification, and regression workflows.
* Added multiclass Decision Tree support.
* Added HEROS and optional TabPFN wrappers, including token-aware TabPFN skip
  behavior when `TABPFN_TOKEN` is unavailable.
* Added Optuna trial accounting so model outputs and reports can distinguish
  requested hyperparameter budgets from completed trials.
* Added local `Parallel` execution in addition to `Serial`, local Dask through
  `Local`, and supported cluster submission modes.
* Added standard and replication report-mode separation, experiment-name-aware
  report titles and filenames, resolved/default parameter display, and
  structured `report_data.json` output for report debugging.
* Added feature-learning and feature-selection summary tables modeled after the
  data-processing and feature-engineering summary table style.
* Added Sphinx/autodoc documentation and GitHub test workflows for Python 3.10,
  3.11, and 3.12.

### Changed

* Reorganized code around registries for Phase 2 preprocessing, Phase 3 feature
  learning, Phase 4 feature importance, Phase 5 selection, Phase 6 models, and
  Phase 7 ensembles.
* Updated the existing configuration-file workflow around `.cfg` files with
  aligned parameter names, UCI demo configs, dry-run support, phase selection
  flags, and matching notebook parameters.
* Standardized parameter names across config files, command-line arguments,
  notebooks, and saved run-command metadata.
* Updated Phase 1 task handling so user-specified `outcome_type` is respected
  instead of being inferred only from the number of unique outcome values.
* Updated Phase 4 so feature-importance methods write scores without mutating
  shared CV datasets.
* Updated scikit-rebate integration to pass STREAMLINE categorical feature
  indexes to ReBATE methods.
* Updated Phase 5 feature selection so rankings can be combined across all
  feature-importance methods that were run.
* Reworked Phase 6 modeling around clearer dataset/model/CV execution units for
  serial, parallel, Dask, and cluster runs.
* Updated ExSTraCS categorical initialization using supported
  `discrete_attribute_limit` and `specified_attributes` parameters.
* Updated native categorical handling for compatible algorithms such as
  CatBoost/CGB and ExSTraCS.
* Updated reports and summary statistics so metric names, no-skill baselines,
  curves, and plots are task-aware rather than binary-only.
* Updated replication and reporting flows to handle binary, multiclass, and
  regression outputs with task-specific metrics and plots.
* Updated Google Colab and local Jupyter notebooks for binary, multiclass,
  regression, and custom dataset workflows.

### Fixed

* Fixed replication behavior that could otherwise zero-fill learned/PCA feature
  columns instead of respecting saved feature-learning artifacts.
* Fixed Phase 4 shared CV file mutation and parallel race risks.
* Fixed `training_subsample` timing so subsampling affects the intended model
  training workflow.
* Fixed multiclass XGB/LGB objective handling and binary-only assumptions.
* Fixed binary ensemble confusion-metric extraction so confusion matrices are
  not dropped before TP/TN/FP/FN-derived metrics are calculated.
* Fixed P1 bash/job submission path issues.
* Fixed report wording where unspecified settings should instead show the
  actual default used or indicate that a phase was not run.
* Fixed no-skill ROC/PR legend notes and label-aware baseline handling for
  multiclass or non-stratified/random CV settings.

## [v0.3.4-beta](https://github.com/UrbsLab/STREAMLINE/releases/tag/v0.3.4-beta) - 2023-09-28

### Changed

* Improved PDF report formatting so first-page content is clearer and reports
  handle larger numbers of analyzed datasets more gracefully.

### Fixed

* Fixed an edge-case failure when running multiple separate replication and
  replication-report phases in legacy mode.

## [v0.3.3-beta](https://github.com/UrbsLab/STREAMLINE/releases/tag/v0.3.3-beta) - 2023-09-23

### Added

* Added invariant feature removal during the C2 cleaning step of data
  processing.
* Added matching replication behavior so features removed during Phase 1
  cleaning are also removed when replication data are processed.

### Changed

* Updated replication PDF report content to simplify the data-processing report.
* Updated first-page PDF report text sizing.

### Fixed

* Fixed algorithm ordering in notebook and Colab figures.
* Fixed unseen binary categorical values during replication by converting them
  to missing values instead of adding incompatible new features.
* Fixed engineered missingness feature naming.
* Fixed legacy cluster mode when categorical or quantitative feature files were
  not specified.

## [v0.3.2-beta](https://github.com/UrbsLab/STREAMLINE/releases/tag/v0.3.2-beta) - 2023-09-13

### Changed

* Updated legacy run mode so submitted jobs can be launched and the script can
  exit instead of waiting for all jobs to complete.
* Updated the STREAMLINE schematic.
* Updated PDF summary file naming.
* Added documentation describing how to check job status.

### Fixed

* Fixed command-line argument passing for legacy run mode.

## [v0.3.1-beta](https://github.com/UrbsLab/STREAMLINE/releases/tag/v0.3.1-beta) - 2023-09-07

### Changed

* Ordered plot legends, including composite feature-importance plot legends,
  alphabetically by full model name.

### Fixed

* Fixed replication imputation when a feature had no missing values during
  training but did have missing values in the replication dataset. The
  replication phase now applies a simple fallback imputation strategy: mean for
  quantitative features and mode for categorical features.

## [v0.3.0-beta](https://github.com/UrbsLab/STREAMLINE/releases/tag/v0.3.0-beta) - 2023-08-06

### Added

* Added Dask jobqueue support for running STREAMLINE across several HPC cluster
  systems.
* Expanded Phase 1 from exploratory analysis into numerical encoding,
  automated cleaning, feature engineering, and a second processed-data EDA pass.
* Added numerical encoding maps for binary text-valued features.
* Added categorical and quantitative feature path parameters, plus output files
  documenting final feature-type handling.
* Added missingness feature engineering with output documentation.
* Added feature and instance cleaning based on missingness.
* Added one-hot encoding for categorical features with three or more values.
* Added highly correlated feature removal with output documentation.
* Added `DataProcessSummary.csv` to track feature, feature-type, instance,
  class, and missing-value counts through data-processing steps.
* Added replication processing parity so replication data are transformed to
  match the target dataset feature space.
* Added command-line support for running the whole pipeline as a single command.
* Added configuration-file support for command-line runs.
* Added class-based modeling algorithm modules to make adding compatible
  scikit-learn-style classifiers easier.
* Added Elastic Net as an included modeling algorithm.
* Added expanded Colab workflows with repository download, easy/manual run
  modes, user data selection, and output download/report display support.
* Added custom HCC demo and replication datasets designed to exercise automatic
  data cleaning, feature engineering, and replication behavior.

### Changed

* Reorganized repository hierarchy, file names, output names, and phase
  groupings.
* Updated the STREAMLINE schematic to match the reorganized phase structure.
* Reverted model feature-importance plot sorting/presentation from median back
  to mean to avoid confusing demo behavior with small CV counts.
* Updated feature correlation heatmap colors, non-redundant triangle display,
  feature-name scaling, and large-feature-count behavior.
* Added `FeatureCorrelations.csv` output.
* Reformatted PDF summaries to reorganize first-page run parameters, include
  version text, and add data-processing/count summaries.
* Added test run and score outputs to univariate analysis files.
* Updated Jupyter and useful notebooks for the reorganized framework.

## [v0.2.5-beta](https://github.com/UrbsLab/STREAMLINE/releases/tag/v0.2.5-beta) - 2022-06-24

### Fixed

* Added a catch to prevent statistical-comparison result failures in specific
  edge cases.
* Cleaned up old commented code.

## [v0.2.4-beta](https://github.com/UrbsLab/STREAMLINE/releases/tag/v0.2.4-beta) - 2022-06-15

### Changed

* Switched feature-importance figure summaries from mean to median at a
  collaborator's recommendation.
* Added median algorithm performance summaries and median performance values to
  PDF summaries.
* Updated statistical significance output to present medians, matching the
  non-parametric statistical comparisons more closely.

### Fixed

* Fixed a no-missing-data edge case where imputation was enabled but no
  imputation file existed.
* Updated model application so replication data can be loaded from `.csv` and
  `.txt` files.

## [v0.2.3-beta](https://github.com/UrbsLab/STREAMLINE/releases/tag/v0.2.3-beta) - 2022-05-20

### Added

* Confirmed stable serial command-line functionality on Linux.

### Changed

* Marked the beta line as stable and fully functional based on testing and user
  feedback after the alpha releases.

## [v0.2.2-beta](https://github.com/UrbsLab/STREAMLINE/releases/tag/v0.2.2-beta) - 2022-05-19

### Changed

* Added composite feature-importance plot weighting by balanced accuracy and
  ROC AUC.
* Removed the `None` option for maximum features in feature selection.
* Updated Logistic Regression Optuna search behavior to avoid invalid
  hyperparameter combinations.
* Enforced Optuna 2.0.0 for hyperparameter-optimization figure generation and
  added error handling so plotting issues do not fail an entire STREAMLINE run.
* Updated notebooks for the beta fixes.

### Fixed

* Fixed composite feature importance when only one algorithm was used.
* Fixed major issues preventing certain phases from running serially from the
  command line.
* Fixed first-page PDF summary formatting.

## [v0.2.1-beta](https://github.com/UrbsLab/STREAMLINE/releases/tag/v0.2.1-beta) - 2022-05-17

### Changed

* Moved the codebase into the `streamline` package folder and updated imports
  accordingly.
* Updated default Optuna run parameters.
* Documented that complete reproducibility is not guaranteed when Optuna is run
  in parallel.
* Rounded scaled CV data to seven decimal places to reduce floating-point
  reproducibility drift after scaling.

## [v0.2.0-beta](https://github.com/UrbsLab/STREAMLINE/releases/tag/v0.2.0-beta) - 2022-05-14

### Added

* Published the first beta release after initial alpha testing across multiple
  platforms and Anaconda installations.

### Changed

* Marked STREAMLINE ready for external use while noting that untested
  configurations could still expose issues.
* Added guidance for users to report run mode, Anaconda version, and errors
  when issues arise.
* Added an early request for users applying STREAMLINE in publications to check
  the repository for the current citation reference.

## [v0.1.3-alpha](https://github.com/UrbsLab/STREAMLINE/releases/tag/v0.1.3-alpha) - 2022-05-12

### Changed

* Updated README installation instructions.
* Updated the default setting for model feature-importance estimation.
* Recorded testing against the then-current Linux Anaconda version.

## [v0.1.2-alpha](https://github.com/UrbsLab/STREAMLINE/releases/tag/v0.1.2-alpha) - 2022-05-12

### Fixed

* Fixed an Anaconda/scipy compatibility issue in exploratory analysis.

## [v0.1.1-alpha](https://github.com/UrbsLab/STREAMLINE/releases/tag/v0.1.1-alpha) - 2022-05-12

### Fixed

* Replaced deprecated `scipy.interp()` usage with `numpy.interp()`.

## [v0.1.0-alpha](https://github.com/UrbsLab/STREAMLINE/releases/tag/v0.1.0-alpha) - 2022-05-12

### Added

* Published the first stable, bug-tested STREAMLINE implementation, inheriting
  much of its underlying code from AutoMLPipe-BC.

### Notes

* This alpha was tested only with the specified Anaconda and package versions,
  and only on Windows and Linux.
