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

The default pytest configuration collects only the current main end-to-end
tests. Legacy and phase-level subtests were removed from the maintained v1.0.0
test path so routine testing stays focused on the binary, multiclass, and
regression demo pipelines.

Run one main test while developing:

```bash
pytest streamline/tests/test_complete_binary.py
pytest streamline/tests/test_complete_multiclass.py
pytest streamline/tests/test_complete_regression.py
```

Run the default main suite when changing shared phase behavior:

```bash
pytest
```

Run the optional TabPFN smoke test directly when changing TabPFN handling:

```bash
pytest streamline/tests/subtests/tabpfn_smoke.py -q -rs
```

STREAMLINE supports Python 3.10 and newer. The CI pytest workflow exercises the
main suite on Python 3.10, 3.11, 3.12, and 3.13.

## Documentation

Build the docs locally with:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/build/html
```

Keep examples synchronized with `run_configs/`, `sample_runcommands.txt`, and
the notebook parameter names.
