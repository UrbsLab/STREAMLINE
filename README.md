![STREAMLINE Logo](https://github.com/UrbsLab/STREAMLINE/blob/main/docs/source/pictures/STREAMLINE_Logo_Full.png?raw=true)

# STREAMLINE

STREAMLINE is an end-to-end automated machine learning pipeline for tabular biomedical and general supervised learning workflows. The current codebase supports binary classification, multiclass classification, and regression, with integrated feature learning, feature importance, feature selection, base-model training, classification ensembles, summary statistics, dataset comparison, replication, and PDF reporting.

This repository is the most up-to-date starting point for the current refactored STREAMLINE pipeline. The older documentation site remains useful for background and historical context, but the README and notebooks in this repository best reflect the current 11-phase implementation.

## What Is Included In This Version

- Unified support for binary classification, multiclass classification, and regression
- Phase-first architecture spanning P1 through P11
- Registry-backed extension points for multiple phases
- Feature learning with PCA and related outputs
- Feature importance methods including mutual information and MultiSURF, with additional methods available through the registry
- Modeling with base learners, calibration support for classification, and composite feature importance from modeling
- Classification ensemble evaluation
- Summary statistics and cross-dataset comparison
- Replication / external validation as a dedicated phase
- Automated reporting for both standard experiment outputs and replication outputs
- Updated Google Colab and Jupyter notebook workflows

## Pipeline Overview

The current pipeline is organized into 11 phases:

| Phase | Name | Purpose |
| --- | --- | --- |
| P1 | Data Process | Load datasets, define CV partitions, run exploratory analysis, and prepare dataset-specific metadata |
| P2 | Impute and Scale | Apply preprocessing, imputation, and scaling |
| P3 | Feature Learning | Learn transformed features such as PCA-derived components and write feature-learning artifacts |
| P4 | Feature Importance | Score features with filter-style feature importance methods |
| P5 | Feature Selection | Select and persist reduced feature sets |
| P6 | Modeling | Train and evaluate base models |
| P7 | Ensembles | Train and evaluate classification ensembles on top of base models |
| P8 | Summary Statistics | Aggregate CV metrics, generate plots, and summarize feature/model behavior |
| P9 | Compare Datasets | Compare results across datasets within an experiment |
| P10 | Replication | Apply the trained workflow to external replication datasets |
| P11 | Reporting | Build publication-style PDF reports for standard or replication outputs |

## Supported Learning Tasks

### Binary Classification

- Standard classification metrics such as balanced accuracy, accuracy, F1, recall, precision, ROC AUC, PRC AUC, PRC APS, and Brier score
- ROC and PR curve outputs
- Calibration-aware outputs and decision-threshold workflows

### Multiclass Classification

- Macro and micro metric support where appropriate
- Multiclass ROC/PR curve summaries
- Multiclass Brier score and one-vs-rest evaluation logic
- Report outputs and summary statistics tailored to multiclass evaluation

### Regression

- Regression metrics such as explained variance, Pearson correlation, MAE, MSE, median absolute error, and max error
- Actual-versus-predicted and residual diagnostics
- Replication and reporting flows adapted for regression outputs
- Phase 7 ensembles are currently classification-only; regression runs should proceed from Phase 6 directly to Phase 8.

## Repository Layout

Top-level items you will use most often:

- [`STREAMLINE_GoogleColab.ipynb`](STREAMLINE_GoogleColab.ipynb): primary Colab-oriented notebook
- [`STREAMLINE_Notebook.ipynb`](STREAMLINE_Notebook.ipynb): local Jupyter notebook workflow
- [`run_configs/`](run_configs): example `.cfg` files for full UCI demo pipelines
- [`data/`](data): demo training and replication datasets
- [`streamline/`](streamline): source code organized by phase
- [`usefulnotebooks/`](usefulnotebooks): focused post hoc analysis and visualization notebooks
- [`requirements.txt`](requirements.txt): local dependency list

Key source directories:

- `streamline/p1_data_process`
- `streamline/p2_impute_scale`
- `streamline/p3_feature_learning`
- `streamline/p4_feature_importance`
- `streamline/p5_feature_selection`
- `streamline/p6_modeling`
- `streamline/p7_ensembles`
- `streamline/p8_summary_statistics`
- `streamline/p9_compare_datasets`
- `streamline/p10_replication`
- `streamline/p11_reporting`
- `streamline/pipeline` for config-driven P1-P11 orchestration

## Run Modes

STREAMLINE can be used in several ways depending on user preference and compute environment:

- Google Colab notebook
- Local Jupyter notebook
- Config-driven full pipeline runs
- Local command line
- Local parallel execution
- HPC / cluster execution using the phase CLIs and job submission helpers

The current codebase includes CLI and runner modules for every phase from P1 through P11.

## Saved Run Commands

Each phase CLI records its resolved arguments in `<output_path>/<experiment_name>/run_commands.pickle` after a successful run. On later runs, the same phase will reuse saved arguments for options you omit, while command-line values you provide override and update the saved entry. Use `--ignore_saved_run_command` for a fresh run or `--no_update_saved_run_command` to avoid updating the pickle.

## Config-Driven Pipeline Runs

STREAMLINE can run multiple phases from one `.cfg` file, matching the config-file workflow used in earlier releases. The config runner reads shared settings from `[run]`, phase toggles from `[phases]`, phase-specific settings from sections such as `[p1]` and `[p6]`, and then calls the same P1-P11 runner classes used by the phase CLIs. The `[phases]` section supports direct flags such as `do_p1 = True` as well as old-style broad flags such as `do_till_report = True`.

Dry-run a config to inspect the resolved phase calls:

```bash
python run.py -c run_configs/uci_binary_hcc.cfg --dry_run
```

Run the configured pipeline:

```bash
python run.py -c run_configs/uci_binary_hcc.cfg
```

Useful controls:

```bash
python run.py -c run_configs/uci_binary_hcc.cfg --start_at p4
python run.py -c run_configs/uci_binary_hcc.cfg --stop_after p8
python run.py -c run_configs/uci_binary_hcc.cfg --only p6,p8,p11
python run.py -c run_configs/uci_binary_hcc.cfg --skip p3,p4
```

Example configs are included for the three UCI demos:

- `run_configs/uci_binary_hcc.cfg`
- `run_configs/uci_multiclass_student.cfg`
- `run_configs/uci_regression_auto_mpg.cfg`

Phase 10 runs only when replication paths are configured, unless it is explicitly enabled. Phase 7 is automatically skipped for continuous/regression runs because the current ensemble registry is classification-only.

## Feature Type Handling

Phase 1 exposes `--one_hot_encoding`. The default keeps historical behavior and expands non-binary categorical features during data processing. Set `--one_hot_encoding 0` when you want Phase 6 to handle raw categorical columns per model.

Phase 6 then uses `--bypass_one_hot_for_native_models` and `--native_categorical_models` to decide how raw categoricals are prepared. If Phase 1 metadata shows `one_hot_encoding=False`, Phase 6 only runs models listed in `--native_categorical_models` by default. Auto-discovered models are filtered to that native-capable list, and explicitly requested unsupported models raise an error instead of being silently one-hot encoded. With the default settings, this means CGB/CatBoost and ExSTraCS are the native categorical models; CatBoost receives `cat_features`, while ExSTraCS receives `discrete_attribute_limit="d"` plus zero-indexed `specified_attributes` for the categorical feature columns.

Phase 6 also writes Optuna trial accounting to `<dataset>/models/optuna_trials/*_optuna_trials*.csv` and includes the same trial summary in each per-CV metrics JSON. This records how many trials actually ran and completed within the requested `--n_trials` and `--timeout` budget.

## Getting Started

### Google Colab

Use the current Colab notebook here:

[Open the STREAMLINE Colab notebook](https://colab.research.google.com/drive/1ByQuU805GzDGAAGzbUYz8wahnOTUuzvg?usp=sharing)

The updated Colab workflow is designed to support:

- classification and regression in the same notebook
- demo runs and custom runs through configuration flags
- richer explanatory markdown and visible phase outputs

### Local Jupyter

For local notebook use, start from:

- [`STREAMLINE_Notebook.ipynb`](STREAMLINE_Notebook.ipynb) for local binary, multiclass, regression, and custom workflows
- [`STREAMLINE_ColabNotebook.ipynb`](STREAMLINE_ColabNotebook.ipynb) for the Colab version of the parameter-driven flow

### Local CLI

Each phase can be run independently with its CLI entry point. For example:

```bash
python -m streamline.p1_data_process.p1_cli --data_path data/UCIBinaryClassification --output_path out --experiment_name DemoBinary --outcome_label Class --outcome_type Binary --instance_label InstanceID --categorical_features data/UCIFeatureTypes/hcc_survival_categorical_features.csv --quantitative_features data/UCIFeatureTypes/hcc_survival_quantitative_features.csv
python -m streamline.p2_impute_scale.p2_cli --output_path out --experiment_name DemoBinary
python -m streamline.p3_feature_learning.p3_cli --output_path out --experiment_name DemoBinary
python -m streamline.p4_feature_importance.p4_cli --output_path out --experiment_name DemoBinary
python -m streamline.p5_feature_selection.p5_cli --output_path out --experiment_name DemoBinary
python -m streamline.p6_modeling.p6_cli --output_path out --experiment_name DemoBinary --outcome_label Class --model_type Binary --instance_label InstanceID
python -m streamline.p7_ensembles.p7_cli --output_path out --experiment_name DemoBinary
python -m streamline.p8_summary_statistics.p8_cli --output_path out --experiment_name DemoBinary --outcome_label Class --outcome_type Binary --instance_label InstanceID
python -m streamline.p9_compare_datasets.p9_cli --output_path out --experiment_name DemoBinary --outcome_label Class --outcome_type Binary --instance_label InstanceID
python -m streamline.p11_reporting.p11_cli --experiment_path out/DemoBinary --report_mode standard
```

For a longer command cookbook covering classification, regression, replication, and reporting examples, see [`sample_runcommands.txt`](sample_runcommands.txt).

Replication and replication reporting are available through:

```bash
python -m streamline.p10_replication.p10_cli \
  --rep_data_path data/UCIRepBinaryClassification \
  --dataset_for_rep data/UCIBinaryClassification/hcc_survival.csv \
  --output_path out \
  --experiment_name DemoBinary

python -m streamline.p11_reporting.p11_cli \
  --experiment_path out/DemoBinary \
  --report_mode replication
```

## Local Installation

For a reproducible local setup, create a dedicated environment and install the repository requirements.

Example with conda:

```bash
git clone --single-branch https://github.com/UrbsLab/STREAMLINE.git
cd STREAMLINE
conda create -n streamline python=3.11 pip
conda activate streamline
pip install -r requirements.txt
```

Example with `venv`:

```
git clone --single-branch https://github.com/UrbsLab/STREAMLINE
cd STREAMLINE
pip install -r requirements.txt
```

Notes:

- The pinned requirements are the best starting point for local reproducibility.
- If you intentionally install the latest unpinned package versions, expect to do some compatibility testing because the upstream scientific Python stack changes frequently.
- Some optional packages depend on compiled libraries or environment-specific binaries.

## Demo Data And Demo Paths

Included UCI demo datasets with missing values and mixed categorical/quantitative features:

- `data/UCIBinaryClassification/hcc_survival.csv` and companion `_copy.csv`
- `data/UCIMulticlassClassification/student_dropout_academic_success.csv` and companion `_copy.csv`
- `data/UCIRegression/auto_mpg.csv` and companion `_copy.csv`
- `data/UCIRepBinaryClassification/hcc_survival_rep.csv`
- `data/UCIRepMulticlassClassification/student_dropout_academic_success_rep.csv`
- `data/UCIRepRegression/auto_mpg_rep.csv`
- `data/UCIFeatureTypes/` for UCI categorical and quantitative feature-type examples
- `data/UCI_DemoDatasets_README.md` for source links, target labels, Auto MPG field handling, and example Phase 1 commands

These datasets are used by the notebooks and test suite to exercise classification, multiclass classification, and regression workflows. The normal UCI folders contain deterministic 80% training splits; the matching `data/UCIRep*` folders contain the held-out 20% replication splits.

## Output Structure

STREAMLINE writes experiment outputs under:

```text
<output_path>/<experiment_name>/
```

Within an experiment, common directories include:

- `<dataset>/exploratory`
- `<dataset>/CVDatasets`
- `<dataset>/feature_learning`
- `<dataset>/feature_importance`
- `<dataset>/feature_selection`
- `<dataset>/models`
- `<dataset>/model_evaluation`
- `<dataset>/ensemble_evaluation`
- `<dataset>/runtime`
- `<dataset>/replication/<rep_dataset>/...`
- `DatasetComparisons/`
- `reporting/`
- `reporting_replication/`
- `jobsCompleted/`

Standard reports are written to:

- `<experiment>/reporting/report.pdf`

Replication reports are written to:

- `<experiment>/reporting_replication/report.pdf`

## Reporting

The reporting phase can:

- discover datasets dynamically
- handle binary, multiclass, and regression outputs
- generate missing figures when possible
- reuse existing figures when available
- generate standard experiment reports
- generate replication-focused reports from replication folders

Reporting entry point:

```bash
python -m streamline.p11_reporting.p11_cli --experiment_path out/DemoRun --report_mode standard
python -m streamline.p11_reporting.p11_cli --experiment_path out/DemoRun --report_mode replication
```

## Useful Notebooks

The `usefulnotebooks/` directory contains focused notebooks for downstream analysis and visualization, including:

- decision-threshold analysis
- ROC and PR curve generation
- composite feature-importance visualization
- feature-importance heatmaps
- model visualization
- test-set probability access
- replication probability analysis

These notebooks are intended for users who want to inspect outputs after a pipeline run rather than drive the entire workflow from a single notebook.

## Extending STREAMLINE

The refactored codebase uses registry-driven discovery across multiple phases. This makes it easier to add new methods without rewriting pipeline orchestration for every new component.

Relevant extension points include:

- `streamline/p2_impute_scale/registry`
- `streamline/p3_feature_learning/registry`
- `streamline/p4_feature_importance/registry`
- `streamline/p5_feature_selection/registry`
- `streamline/p6_modeling/models`
- `streamline/p7_ensembles/registry`

In practice, this means new preprocessing, feature-learning, feature-importance, feature-selection, modeling, and ensemble components can be added in a more modular way than in older versions of STREAMLINE.

## Current Notes And Limitations

- STREAMLINE is designed for supervised learning on tabular data.
- Unstructured data pipelines such as image, text, audio, and raw time-series feature extraction are not automated here.
- Some documentation pages on the public docs site still describe older pipeline versions or older terminology.
- As with any ML pipeline assembled on top of evolving third-party libraries, latest-version local environments may expose compatibility issues that require updates.

## Publications And Citation

The first publication detailing STREAMLINE (release Beta 0.2.4) and applying it to simulated benchmark data is available here:

[Springer chapter](https://link.springer.com/chapter/10.1007/978-981-19-8460-0_9)

The paper is also available as a preprint:

[arXiv preprint](https://arxiv.org/abs/2206.12002)

Additional citation information and related publications:

[STREAMLINE citations](https://urbslab.github.io/STREAMLINE/citation.html)

## Additional Resources

- Historical documentation site: [https://urbslab.github.io/STREAMLINE/index.html](https://urbslab.github.io/STREAMLINE/index.html)
- FAQ / background page: [https://urbslab.github.io/STREAMLINE/about.html](https://urbslab.github.io/STREAMLINE/about.html)

## Contact

We welcome ideas, suggestions, bug reports, code contributions, and collaborations.

- General questions and collaboration inquiries: Ryan Urbanowicz at `ryan.urbanowicz@cshs.org`
- Codebase, installation, running, troubleshooting, and implementation questions: Harsh Bandhey at `harsh.bandhey@cshs.org`

## Acknowledgements

The development of STREAMLINE benefited from feedback across multiple biomedical research collaborators at the University of Pennsylvania, Fox Chase Cancer Center, Cedars Sinai Medical Center, and the University of Kansas Medical Center.

The bulk of the coding was completed by Ryan Urbanowicz, Robert Zhang, and Harsh Bandhey. Special thanks to Yuhan Cui, Pranshu Suri, Patryk Orzechowski, Trang Le, Sy Hwang, Richard Zhang, Wilson Zhang, and Pedro Ribeiro for their code contributions and feedback.

We also thank the following collaborators for their feedback on application of the pipeline during development: Shannon Lynch, Rachael Stolzenberg-Solomon, Ulysses Magalang, Allan Pack, Brendan Keenan, Danielle Mowery, Jason Moore, and Diego Mazzotti.

Funding supporting this work comes from NIH grants: R01 AI173095, U01 AG066833, and P01 HL160471.
