# Output

STREAMLINE writes outputs under:

```text
<output_path>/<experiment_name>/
```

For example:

```text
out/UCIHCCPipeline/
```

## Experiment-Level Files

Common experiment-level outputs include:

| Path | Description |
| --- | --- |
| `metadata.pickle` | Experiment metadata saved by P1, including dataset names, outcome settings, CV settings, and feature-type settings. |
| `run_params.pickle` | Resolved pipeline/config parameters used by the run. |
| `run_commands.pickle` | Saved resolved phase arguments for repeat runs. |
| `jobs/` | Scheduler/job scripts created for BashSLURM or BashLSF runs. |
| `logs/` | Scheduler stdout/stderr logs for BashSLURM or BashLSF runs. |
| `DatasetComparisons/` | P9 cross-dataset comparison outputs. |
| `reporting/<experiment_name>_STREAMLINE_Report.pdf` | Standard P11 report. |
| `reporting_replication/<experiment_name>_STREAMLINE_Replication_Report.pdf` | Replication P11 report. |
| `jobsCompleted/` | Completion markers for orchestration. |
| `runtime/` | Experiment-level runtime files, including P9 and P11 timing when those phases run. |

`jobsCompleted/` is mainly for STREAMLINE orchestration. It is useful when
debugging scheduler runs, but it is not usually the first place to inspect
scientific results.

## How To Check A Run Quickly

After a full demo run, check for these files first:

```text
<output_path>/<experiment_name>/<dataset>/model_evaluation/Summary_performance_mean.csv
<output_path>/<experiment_name>/reporting/<experiment_name>_STREAMLINE_Report.pdf
```

If P10/P11 replication ran, also check:

```text
<output_path>/<experiment_name>/<dataset>/replication/<rep_dataset>/model_evaluation/Summary_performance_mean.csv
<output_path>/<experiment_name>/reporting_replication/<experiment_name>_STREAMLINE_Replication_Report.pdf
```

## Directory Tree

A typical single-dataset experiment looks like this. Some folders appear only
when the corresponding phase is run.

```text
<output_path>/<experiment_name>/
├── metadata.pickle
├── run_params.pickle
├── run_commands.pickle
├── jobs/
├── logs/
├── jobsCompleted/
├── runtime/
├── DatasetComparisons/
├── reporting/
│   ├── <experiment_name>_STREAMLINE_Report.pdf
│   ├── report_data.json
│   └── figures/
├── reporting_replication/
│   ├── <experiment_name>_STREAMLINE_Replication_Report.pdf
│   ├── report_data.json
│   └── figures/
└── <dataset>/
    ├── exploratory/
    ├── CVDatasets/
    ├── impute_scale/
    ├── feature_learning/
    ├── feature_importance/
    ├── feature_selection/
    ├── models/
    ├── model_evaluation/
    ├── ensemble_evaluation/
    ├── runtime/
    └── replication/
```

For multi-dataset experiments, STREAMLINE creates one `<dataset>/` folder for
each input dataset under the same experiment root. P9 writes
`DatasetComparisons/` only when at least two dataset folders are available for
comparison.

## Dataset-Level Folders

Each dataset gets a folder under the experiment directory:

| Folder | Produced by | Description |
| --- | --- | --- |
| `exploratory/` | P1 | DataProcessSummary, missingness, feature typing, class counts, and EDA summaries. |
| `CVDatasets/` | P1-P5 | Train/test CV datasets, including selected feature versions. |
| `impute_scale/` | P2 | Imputation/scaling metadata and artifacts. |
| `feature_learning/` | P3 | Learned feature manifests and feature lists. |
| `feature_importance/` | P4 | Feature score files by method and CV. |
| `feature_selection/` | P5 | Informative feature summaries and selected feature artifacts. |
| `models/` | P6 | Fitted models, predictions, metrics, and Optuna accounting. |
| `model_evaluation/` | P6/P8 | Summary metrics and model plots. |
| `ensemble_evaluation/` | P7/P8 | Ensemble metrics and plots for classification runs. |
| `runtime/` | multiple | Runtime summaries. |
| `replication/` | P10 | Replication predictions, metrics, and plots. |

## Phase Output Details

The most useful files for each phase are:

| Phase | Main location | Useful files |
| --- | --- | --- |
| P1 Data Process | `<dataset>/exploratory/` and `<dataset>/CVDatasets/` | `DataProcessSummary.csv`, `DataCounts.csv`, `ClassCounts.csv`, `DataMissingness.csv`, `FeatureCorrelations.csv`, categorical/quantitative feature headers, and `<dataset>_CV_<k>_Train.csv` / `<dataset>_CV_<k>_Test.csv`. |
| P2 Impute/Scale/Balance | `<dataset>/impute_scale/` and `<dataset>/CVDatasets/` | Saved imputer/scaler metadata and fitted transformers. The CV train/test CSVs are updated with imputed, scaled, and optional SMOTE-balanced training data. |
| P3 Feature Learning | `<dataset>/feature_learning/` and `<dataset>/CVDatasets/` | `feature_manifest_cv<k>.json`, `features_cv<k>.txt`, `input_features_cv<k>.txt`, `learner_cv<k>.pickle`, and `fitted_learner_cv<k>.pickle`. |
| P4 Feature Importance | `<dataset>/feature_importance/<method>/` | `<method>_scores_cv_<k>.csv` score rankings and optional `TopAverageScores.png` plots. |
| P5 Feature Selection | `<dataset>/feature_selection/` and `<dataset>/CVDatasets/` | Informative/uninformative feature summaries plus selected train/test CV files when filtering is enabled. |
| P6 Modeling | `<dataset>/models/` and `<dataset>/model_evaluation/` | `pickledModels/<model>_<k>.pickle`, `<model>_usedparams<k>.csv` or `<model>_bestparams<k>.csv`, `metrics_by_cv/<model>_CV_<k>.json`, `curves_by_cv/`, and `pickled_metrics/` residual payloads for regression. |
| P7 Ensembles | `<dataset>/ensemble_evaluation/` | Ensemble per-CV metrics, curve JSON files, and ensemble summary artifacts for classification runs. |
| P8 Summary Statistics | `<dataset>/model_evaluation/` | `Summary_performance_mean.csv`, `Summary_performance_median.csv`, `Summary_performance_std.csv`, `statistical_comparisons/`, `evalPlots/`, and `feature_importance/` composite outputs. |
| P9 Dataset Compare | `DatasetComparisons/` | Cross-dataset statistical comparison CSVs and boxplots. This phase is skipped when fewer than two comparable dataset folders exist. |
| P10 Replication | `<dataset>/replication/<rep_dataset>/` | Replication processed data, replication CV/test files, model metrics, model curves, ensemble metrics, and replication summaries. |
| P11 Reporting | `reporting/` and `reporting_replication/` | PDF reports, `report_data.json`, generated figure cache, and `runtime_report.txt`. |

The exact file list can vary by task type and phase settings. For example,
ROC/PR curve files are created for classification tasks, while regression runs
write residual and actual-vs-predicted outputs instead.

## Cross-Validation Files

`CVDatasets/` is the handoff point between phases. P1 creates the initial
fold-specific train/test CSVs:

```text
<dataset>/CVDatasets/<dataset>_CV_0_Train.csv
<dataset>/CVDatasets/<dataset>_CV_0_Test.csv
...
```

Later phases update these files as the dataset moves through imputation,
scaling, balancing, feature learning, and feature selection. The train file is
the only split that should be fit or resampled by preprocessing/modeling
steps. The paired test file is transformed using training-fold artifacts and
is preserved for fold-level evaluation.

If a phase is skipped, downstream phases use the most recent CV files produced
by earlier phases.

## Metrics And Model Artifacts

For model-level debugging, start with:

```text
<dataset>/model_evaluation/metrics_by_cv/<model>_CV_<k>.json
<dataset>/models/pickledModels/<model>_<k>.pickle
<dataset>/models/optuna_trials/<model>_optuna_trials<k>.csv
```

The per-CV JSON files store raw metric values, feature importance used by the
report, Optuna trial accounting, and categorical feature handling information.
P8 then aggregates those per-CV files into:

```text
<dataset>/model_evaluation/Summary_performance_mean.csv
<dataset>/model_evaluation/Summary_performance_median.csv
<dataset>/model_evaluation/Summary_performance_std.csv
```

Those summary CSVs are the best starting point when comparing model
performance outside the PDF report.

## Replication Output

Replication outputs are nested under the training dataset that supplied the
trained workflow:

```text
<output_path>/<experiment_name>/<dataset>/replication/<rep_dataset>/
```

Inside that folder, STREAMLINE mirrors the main dataset structure where
possible, including exploratory summaries, processed replication data,
replication model metrics, curve files, and summary performance tables. The
replication report is built from these nested folders rather than from the
training/CV test metrics.

## Reports

P11 can generate two report scopes:

```bash
python -m streamline.p11_reporting.p11_cli \
  --experiment_path out/UCIHCCPipeline \
  --report_mode standard

python -m streamline.p11_reporting.p11_cli \
  --experiment_path out/UCIHCCPipeline \
  --report_mode replication
```

The standard report focuses on training/CV experiment outputs. The replication
report focuses on external validation outputs under the dataset replication
folders.

The first page of each report is intended to answer the practical questions
users ask first: what dataset was run, what phases were run, which settings
were used, what task type was evaluated, and where the strongest or tied metric
results appear.

## Report Data

Each report directory also includes `report_data.json`. This JSON is the
structured input used to build the PDF and is useful for debugging report
content without parsing the PDF.

## Figures

The reporting phase can either reuse existing generated figures or generate
missing figures:

```bash
python -m streamline.p11_reporting.p11_cli \
  --experiment_path out/UCIHCCPipeline \
  --enable_plots 1 \
  --reuse_existing_figures 1
```

Set `--enable_plots 0` when you want a faster report-only smoke test.
