# Changelog

This page maps STREAMLINE's branch history to the public GitHub release log.
It is a user-facing summary rather than a commit-by-commit changelog.

Release log reference:
[GitHub Releases](https://github.com/UrbsLab/STREAMLINE/releases?page=1)

## Version Map

| Documentation name | Branch/reference | Release tag reference | Status |
| --- | --- | --- | --- |
| STREAMLINE v1 | `legacy` | `v0.1.0-alpha` through `v0.2.5-beta` | Historical legacy line |
| STREAMLINE v2 | `main` / `v2` | `v0.3.0-beta` through `v0.3.4-beta` | Current public release line |
| STREAMLINE v3 | current v3 development | planned next major tag, for example `v3.0.0` | Current improvement branch, not yet a GitHub release |

## Release Log Summary

| Release | Date | Release title / note |
| --- | --- | --- |
| `v0.1.0-alpha` | 2022-05-12 | Alpha Release |
| `v0.1.1-alpha` | 2022-05-12 | Alpha 0.1.1 |
| `v0.1.2-alpha` | 2022-05-12 | Alpha 0.1.2 |
| `v0.1.3-alpha` | 2022-05-12 | Alpha 0.1.3 |
| `v0.2.0-beta` | 2022-05-14 | Beta Release 0.2.0 |
| `v0.2.1-beta` | 2022-05-17 | Code rearrange and default updates |
| `v0.2.2-beta` | 2022-05-19 | `.gitignore` update |
| `v0.2.3-beta` | 2022-05-20 | Linux serial command-line runs fixed |
| `v0.2.4-beta` | 2022-06-15 | Command-line cleanup |
| `v0.2.5-beta` | 2022-06-24 | Minor statistical-analysis/code cleanup fix |
| `v0.3.0-beta` | 2023-08-06 | Documentation update and removal of old documentation files |
| `v0.3.1-beta` | 2023-09-07 | Development documentation update |
| `v0.3.2-beta` | 2023-09-13 | Development documentation update |
| `v0.3.3-beta` | 2023-09-23 | Development documentation update |
| `v0.3.4-beta` | 2023-09-28 | Latest public beta release |

## STREAMLINE v1: Legacy Line

Reference: `legacy` branch, with public release history from
`v0.1.0-alpha` through `v0.2.5-beta`.

The legacy line contains the original STREAMLINE implementation. It was
organized around phase-specific scripts and job files for exploratory
analysis, preprocessing, feature importance, feature selection, modeling,
statistics, dataset comparison, replication/application, report generation,
and cleanup.

Major characteristics:

* focused on tabular supervised binary classification
* script/job-file style execution
* demonstration HCC data and replication examples
* Google Colab and Jupyter notebook entry points
* early PDF reports and post-run analysis notebooks
* cluster-oriented job submission patterns

Important release milestones:

* `v0.1.0-alpha`: first public alpha release line
* `v0.2.0-beta`: beta release line begins
* `v0.2.1-beta`: code rearrange and default updates
* `v0.2.3-beta`: Linux serial command-line run fixes
* `v0.2.5-beta`: minor statistical-analysis and old-code cleanup fixes

Use v1 only when reproducing an old analysis that depends on the legacy script
layout.

## STREAMLINE v2: Main Public Line

Reference: `main` / `v2` branch, with public release history from
`v0.3.0-beta` through `v0.3.4-beta`.

v2 is the current public release line represented by the older hosted
documentation. It reorganized STREAMLINE into importable package areas such as
`dataprep`, `featurefns`, `modeling`, `models`, `postanalysis`, `runners`, and
`utils`.

Major improvements over v1:

* clearer Sphinx documentation site
* improved package structure and runner classes
* maintained Google Colab and Jupyter workflows
* command-line and cluster-oriented execution paths
* automated PDF reports and generated plots
* useful post-run notebooks for thresholds, prediction probabilities, ROC/PRC plots, and feature importance visualization

Important release milestones:

* `v0.3.0-beta`: documentation refresh and old documentation cleanup
* `v0.3.1-beta` through `v0.3.4-beta`: development-documentation updates
* `v0.3.4-beta`: latest public GitHub release

Important scope note: v2 documentation describes the historical 9-phase
STREAMLINE workflow and is primarily binary-classification-oriented.

## STREAMLINE v3: Current Improvement Branch

Reference: current v3 development branch. This branch is not yet represented
by a public GitHub release tag. The intended release reference should be the
next major tag, for example `v3.0.0`, once v3 is merged and released.

v3 keeps the original goal of transparent, reproducible tabular AutoML, but
reorganizes the implementation around explicit P1-P11 phase packages and
config-driven orchestration.

Major improvements over v2:

* supports binary classification, multiclass classification, and regression
* replaces older package areas with explicit phase modules:
  `p1_data_process` through `p11_reporting`
* adds a config runner through `run.py` and `run_configs/*.cfg`
* aligns parameter names across config files, CLIs, and notebooks
* saves resolved run commands so repeated phase calls can reuse prior settings
* adds UCI-based demo datasets for binary, multiclass, and regression workflows
* adds deterministic held-out replication demo splits for P10
* adds SMOTE/SMOTENC support in P2
* adds feature learning as its own P3 phase
* updates feature importance/selection around current scikit-rebate behavior
* adds MultiSWRFDB and MultiSWRFDB* feature-importance options
* supports native categorical handling for models such as CatBoost/CGB and ExSTraCS
* records Optuna trial accounting so reports show how many trials actually ran
* supports local joblib parallelism with `run_cluster = Parallel`
* keeps local Dask execution under `run_cluster = Local`
* restructures modeling jobs around dataset/model/CV work units
* adds and updates complete binary, multiclass, and regression pytest smoke tests
* improves standard and replication PDF report summaries, feature-engineering tables, feature-importance figures, and replication naming
* updates the Google Colab and local Jupyter notebooks for demo and custom data modes
* removes old main-era files that are no longer part of the maintained v3 workflow

Use v3 for new work once released or when intentionally working from the v3
development branch.
