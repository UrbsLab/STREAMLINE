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
| `run_commands.pickle` | Saved resolved phase arguments for repeat runs. |
| `DatasetComparisons/` | P9 cross-dataset comparison outputs. |
| `reporting/report.pdf` | Standard P11 report. |
| `reporting_replication/report.pdf` | Replication P11 report. |
| `jobsCompleted/` | Completion markers for orchestration. |

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
