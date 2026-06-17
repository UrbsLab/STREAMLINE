# Development

The refactored codebase is organized by pipeline phase. Each phase generally
contains:

* a CLI module, for example `p6_cli.py`
* a runner module, for example `p6_runner.py`
* implementation modules and registries
* optional job-submission helpers for cluster modes

## Source Layout

```text
streamline/
  p1_data_process/
  p2_impute_scale/
  p3_feature_learning/
  p4_feature_importance/
  p5_feature_selection/
  p6_modeling/
  p7_ensembles/
  p8_summary_statistics/
  p9_compare_datasets/
  p10_replication/
  p11_reporting/
  pipeline/
  utils/
```

## Adding Methods

Prefer the existing registry patterns when adding new components:

| Area | Location |
| --- | --- |
| Imputers/scalers | `streamline/p2_impute_scale/registry` |
| Feature learners | `streamline/p3_feature_learning/registry` |
| Feature importance methods | `streamline/p4_feature_importance/registry` |
| Feature selectors | `streamline/p5_feature_selection/registry` |
| Models | `streamline/p6_modeling/models` |
| Ensembles | `streamline/p7_ensembles/registry` |

## Tests

Run focused tests while developing:

```bash
pytest streamline/tests/test_pipeline_runner.py
pytest streamline/tests/test_reporting_run_commands.py
pytest streamline/tests/test_uci_demo_datasets.py
```

Run the full suite when changing shared phase behavior:

```bash
pytest
```

## Documentation

Build the docs locally with:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/build/html
```

Keep examples synchronized with `run_configs/`, `sample_runcommands.txt`, and
the notebook parameter names.
