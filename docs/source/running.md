# Running STREAMLINE

STREAMLINE can be run through:

* Google Colab
* Local Jupyter notebook
* Config-driven full pipeline runs
* Phase-by-phase command-line calls

For most command-line work, start with the `.cfg` runner. It keeps shared
settings, phase toggles, and phase-specific parameters in one editable file.

## Google Colab

Open the Colab notebook:

[Open the STREAMLINE Colab notebook](https://colab.research.google.com/drive/1ByQuU805GzDGAAGzbUYz8wahnOTUuzvg?usp=sharing)

At the top of the notebook, set the demo/run parameters. The notebook supports
binary, multiclass, regression, and custom data modes.

## Local Jupyter

From the repository root:

```bash
conda activate streamline
jupyter notebook
```

Open `STREAMLINE_Notebook.ipynb`. The top parameter block controls which demo
or custom dataset is run.

## Config-Driven Runs

Dry-run a config first:

```bash
python run.py -c run_configs/uci_binary_hcc.cfg --dry_run
```

Run a full binary demo:

```bash
python run.py -c run_configs/uci_binary_hcc.cfg
```

Run the multiclass and regression demos:

```bash
python run.py -c run_configs/uci_multiclass_student.cfg
python run.py -c run_configs/uci_regression_auto_mpg.cfg
```

Partial-run examples:

```bash
python run.py -c run_configs/uci_binary_hcc.cfg --start_at p4
python run.py -c run_configs/uci_binary_hcc.cfg --stop_after p8
python run.py -c run_configs/uci_binary_hcc.cfg --only p6,p8,p11
python run.py -c run_configs/uci_binary_hcc.cfg --skip p3,p4
```

## Config File Layout

Each config uses sections like:

```ini
[run]
output_path = out
experiment_name = UCIHCCPipeline
outcome_label = Class
outcome_type = Binary
instance_label = InstanceID
n_splits = 3
run_cluster = Serial
random_state = 42

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

[p6]
outcome_type = Binary
models = NB,LR,DT
scoring_metric = balanced_accuracy
metric_direction = maximize
n_trials = 200
timeout = 900
```

Use the included configs as templates:

* `run_configs/uci_binary_hcc.cfg`
* `run_configs/uci_multiclass_student.cfg`
* `run_configs/uci_regression_auto_mpg.cfg`

## Phase CLI Commands

Each phase can also be run independently. A binary classification example:

```bash
python -m streamline.p1_data_process.p1_cli \
  --data_path data/UCIBinaryClassification \
  --output_path out \
  --experiment_name DemoBinary \
  --outcome_label Class \
  --outcome_type Binary \
  --instance_label InstanceID \
  --categorical_features data/UCIFeatureTypes/hcc_survival_categorical_features.csv \
  --quantitative_features data/UCIFeatureTypes/hcc_survival_quantitative_features.csv \
  --n_splits 3 \
  --force true

python -m streamline.p2_impute_scale.p2_cli --output_path out --experiment_name DemoBinary
python -m streamline.p3_feature_learning.p3_cli --output_path out --experiment_name DemoBinary
python -m streamline.p4_feature_importance.p4_cli --output_path out --experiment_name DemoBinary
python -m streamline.p5_feature_selection.p5_cli --output_path out --experiment_name DemoBinary

python -m streamline.p6_modeling.p6_cli \
  --output_path out \
  --experiment_name DemoBinary \
  --outcome_label Class \
  --outcome_type Binary \
  --instance_label InstanceID \
  --models NB,LR,DT \
  --scoring_metric balanced_accuracy \
  --metric_direction maximize

python -m streamline.p7_ensembles.p7_cli --output_path out --experiment_name DemoBinary
python -m streamline.p8_summary_statistics.p8_cli --output_path out --experiment_name DemoBinary
python -m streamline.p9_compare_datasets.p9_cli --output_path out --experiment_name DemoBinary

python -m streamline.p10_replication.p10_cli \
  --rep_data_path data/UCIRepBinaryClassification \
  --dataset_for_rep data/UCIBinaryClassification/hcc_survival.csv \
  --output_path out \
  --experiment_name DemoBinary

python -m streamline.p11_reporting.p11_cli \
  --experiment_path out/DemoBinary \
  --report_mode standard

python -m streamline.p11_reporting.p11_cli \
  --experiment_path out/DemoBinary \
  --report_mode replication
```

More examples are available in `sample_runcommands.txt`.

## Discovery Commands

Several phases can list registry options:

```bash
python -m streamline.p2_impute_scale.p2_cli --output_path out --experiment_name DemoBinary --list-imputers
python -m streamline.p2_impute_scale.p2_cli --output_path out --experiment_name DemoBinary --list-scalers
python -m streamline.p3_feature_learning.p3_cli --output_path out --experiment_name DemoBinary --list-learners
python -m streamline.p4_feature_importance.p4_cli --output_path out --experiment_name DemoBinary --list-models
python -m streamline.p6_modeling.p6_cli --output_path out --experiment_name DemoBinary --outcome_type Binary --list_models
python -m streamline.p7_ensembles.p7_cli --output_path out --experiment_name DemoBinary --list_ensembles
```
