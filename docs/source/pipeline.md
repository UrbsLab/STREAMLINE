# Pipeline

STREAMLINE is implemented as eleven phase modules plus a config-driven
orchestrator.

| Phase | Module | Main CLI | Summary |
| --- | --- | --- | --- |
| P1 | `streamline.p1_data_process` | `python -m streamline.p1_data_process.p1_cli` | Load datasets, clean/encode columns, generate EDA outputs, and create CV folds. |
| P2 | `streamline.p2_impute_scale` | `python -m streamline.p2_impute_scale.p2_cli` | Impute, scale, and optionally apply SMOTE/SMOTENC to training folds. |
| P3 | `streamline.p3_feature_learning` | `python -m streamline.p3_feature_learning.p3_cli` | Add learned features such as PCA components and write feature manifests. |
| P4 | `streamline.p4_feature_importance` | `python -m streamline.p4_feature_importance.p4_cli` | Score features without mutating shared CV datasets. |
| P5 | `streamline.p5_feature_selection` | `python -m streamline.p5_feature_selection.p5_cli` | Select informative features and persist selected CV datasets. |
| P6 | `streamline.p6_modeling` | `python -m streamline.p6_modeling.p6_cli` | Train base models, tune with Optuna, evaluate CV metrics, and save predictions. |
| P7 | `streamline.p7_ensembles` | `python -m streamline.p7_ensembles.p7_cli` | Build classification ensembles from base model predictions. |
| P8 | `streamline.p8_summary_statistics` | `python -m streamline.p8_summary_statistics.p8_cli` | Aggregate performance, feature importance, and model summaries. |
| P9 | `streamline.p9_compare_datasets` | `python -m streamline.p9_compare_datasets.p9_cli` | Compare dataset-level outputs within an experiment. |
| P10 | `streamline.p10_replication` | `python -m streamline.p10_replication.p10_cli` | Apply trained workflows to replication datasets. |
| P11 | `streamline.p11_reporting` | `python -m streamline.p11_reporting.p11_cli` | Generate standard and replication PDF reports. |

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
