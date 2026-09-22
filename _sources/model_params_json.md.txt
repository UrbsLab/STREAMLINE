# Model Parameter JSON

`model_params_json` lets Phase 6 pass model-specific settings into model
wrappers. Use it when most STREAMLINE run settings can stay shared, but one
model needs a specific constructor value.

The parameter is named `model_params_json` in `.cfg` files, notebooks, and the
Phase 6 CLI.

## How It Works

`model_params_json` is a mapping from model ID to parameter overrides:

```json
{
  "MODEL_ID": {
    "parameter_name": "parameter_value"
  }
}
```

Model IDs are matched case-insensitively against the model wrapper
`small_name` or full `model_name`. For example, `HEROS`, `heros`, and
`ExSTraCS` are accepted model keys.

Use exact model wrapper parameter names. For constructor-controlled settings,
such as HEROS and ExSTraCS hyperparameters, values must be present before the
wrapper builds its Phase 6 parameter grid.

## Expert Knowledge

Phase 6 automatically passes Phase 4 feature-importance scores into model
wrappers that expose an expert-knowledge constructor parameter. ExSTraCS uses
this through its `expert_knowledge` parameter.

The score vector is loaded separately for each CV split and ordered to match
the features in that CV training file after feature selection. STREAMLINE uses
Relief-style scores first when available (`MultiSWRFDB`, `MultiSWRFDB*`,
`MultiSURF`, `MultiSURF*`) and falls back to mutual information.

You usually do not need to set `expert_knowledge` yourself. If you do supply
it through `model_params_json`, your value takes precedence over the automatic
Phase 4 score loading.

## Config File Examples

In a `.cfg` file, place `model_params_json` in the `[p6]` section:

```ini
[p6]
outcome_type = Binary
models = HEROS,ExSTraCS
model_params_json = {"HEROS": {"iterations": 200000, "pop_size": 2000, "model_iterations": 1000, "model_pop_size": 200, "nu": 1}, "ExSTraCS": {"iterations": 200000, "N": 2000, "nu": 1, "rule_compaction": "QRF"}}
```

This runs HEROS and ExSTraCS with fixed values. It does not create a broad
hyperparameter sweep because every supplied value is a single numeric or string
value.

## CLI Examples

For a direct Phase 6 CLI run, quote the parameter string:

```bash
python -m streamline.p6_modeling.p6_cli \
  --output_path out \
  --experiment_name DemoBinary \
  --outcome_type Binary \
  --n_splits 3 \
  --models HEROS,ExSTraCS \
  --model_params_json '{"HEROS": {"pop_size": 2000, "model_pop_size": 200}, "ExSTraCS": {"N": 5000, "nu": 1}}'
```

Use `None` when you want to restore a built-in candidate list for one
parameter. This example searches only HEROS `pop_size`:

```bash
python -m streamline.p6_modeling.p6_cli \
  --output_path out \
  --experiment_name DemoBinary \
  --outcome_type Binary \
  --n_splits 3 \
  --models HEROS \
  --model_params_json '{"HEROS": {"pop_size": None}}'
```

You can combine fixed values and `None` in the same CLI value. This keeps
HEROS mostly fixed while searching `pop_size`, and keeps ExSTraCS mostly fixed
while searching `N`:

```bash
python -m streamline.p6_modeling.p6_cli \
  --output_path out \
  --experiment_name DemoBinary \
  --outcome_type Binary \
  --n_splits 3 \
  --models HEROS,ExSTraCS \
  --model_params_json '{"HEROS": {"iterations": 100000, "pop_size": None, "model_iterations": 500, "model_pop_size": 100, "nu": 1}, "ExSTraCS": {"iterations": 200000, "N": None, "nu": 1, "rule_compaction": "QRF"}}'
```

To run the built-in candidate sweep for every HEROS and ExSTraCS wrapper
parameter, set each tunable parameter to `None`:

```bash
python -m streamline.p6_modeling.p6_cli \
  --output_path out \
  --experiment_name DemoBinary \
  --outcome_type Binary \
  --n_splits 3 \
  --models HEROS,ExSTraCS \
  --n_trials 200 \
  --timeout 900 \
  --model_params_json '{"HEROS": {"iterations": None, "pop_size": None, "model_iterations": None, "model_pop_size": None, "nu": None}, "ExSTraCS": {"iterations": None, "N": None, "nu": None, "rule_compaction": None}}'
```

## Notebook Examples

In a notebook parameter block, use a Python dictionary:

```python
P6_MODEL_PARAMS_JSON = {
    "HEROS": {
        "iterations": 200000,
        "pop_size": 2000,
        "model_iterations": 1000,
        "model_pop_size": 200,
        "nu": 1,
    },
    "ExSTraCS": {
        "iterations": 200000,
        "N": 2000,
        "nu": 1,
        "rule_compaction": "QRF",
    },
}
```

Use `None` in notebooks when you want the built-in candidate list for a
specific parameter:

```python
P6_MODEL_PARAMS_JSON = {
    "HEROS": {"pop_size": None},
    "ExSTraCS": {"iterations": None},
}
```

## HEROS

HEROS uses these Phase 6 wrapper parameters:

| Parameter | Fixed default | Candidate list when set to `None` |
| --- | ---: | --- |
| `iterations` | `100000` | `[100000, 200000, 500000]` |
| `pop_size` | `1000` | `[1000, 2000, 5000]` |
| `model_iterations` | `500` | `[500, 1000]` |
| `model_pop_size` | `100` | `[100, 200]` |
| `nu` | `1` | `[1, 10]` |

Use `pop_size` for the top-level HEROS population size. Use
`model_pop_size` for the internal model population size. HEROS does not use an
`N` alias in STREAMLINE.

Example fixed HEROS settings:

```json
{
  "HEROS": {
    "iterations": 200000,
    "pop_size": 2000,
    "model_iterations": 1000,
    "model_pop_size": 200,
    "nu": 1
  }
}
```

Example HEROS run where only `pop_size` is searched:

```python
{
  "HEROS": {
    "iterations": 100000,
    "pop_size": None,
    "model_iterations": 500,
    "model_pop_size": 100,
    "nu": 1
  }
}
```

## ExSTraCS

ExSTraCS uses these Phase 6 wrapper parameters:

| Parameter | Fixed default | Candidate list when set to `None` |
| --- | ---: | --- |
| `iterations` | `200000` | `[100000, 200000, 500000]` |
| `N` | `2000` | `[1000, 2000, 5000]` |
| `nu` | `1` | `[1, 10]` |
| `rule_compaction` | `"QRF"` | `[None, "QRF"]` |

ExSTraCS uses `N` for population size. HEROS uses `pop_size`.

When Phase 4 feature-importance results exist, STREAMLINE automatically passes
CV-specific expert knowledge into ExSTraCS. Manual `expert_knowledge` values in
`model_params_json` override the automatic Phase 4 scores.

Example fixed ExSTraCS settings:

```python
{
  "ExSTraCS": {
    "iterations": 200000,
    "N": 2000,
    "nu": 1,
    "rule_compaction": "QRF"
  }
}
```

Example ExSTraCS run where only `N` is searched:

```python
{
  "ExSTraCS": {
    "iterations": 200000,
    "N": None,
    "nu": 1,
    "rule_compaction": "QRF"
  }
}
```

## Using `None`

By default, HEROS and ExSTraCS use fixed single-value settings. This protects
runs from accidentally launching expensive RBML hyperparameter searches.

Passing `None` restores the built-in candidate list only for that parameter.
If any model parameter has a candidate list with more than one value, Phase 6
uses Optuna for that model within the configured `n_trials` and `timeout`
budget.

For `.cfg` files:

```ini
model_params_json = {"HEROS": {"pop_size": None}}
```

To sweep all built-in HEROS and ExSTraCS wrapper candidate lists from a
`.cfg` file:

```ini
model_params_json = {"HEROS": {"iterations": None, "pop_size": None, "model_iterations": None, "model_pop_size": None, "nu": None}, "ExSTraCS": {"iterations": None, "N": None, "nu": None, "rule_compaction": None}}
```

```bash
python -m streamline.p6_modeling.p6_cli \
  --output_path out \
  --experiment_name DemoBinary \
  --outcome_type Binary \
  --n_splits 3 \
  --models HEROS \
  --model_params_json '{"HEROS": {"pop_size": None}}'
```

For notebook dictionaries:

```python
P6_MODEL_PARAMS_JSON = {"HEROS": {"pop_size": None}}
```

To sweep all built-in HEROS and ExSTraCS wrapper candidate lists from a
notebook:

```python
P6_MODEL_PARAMS_JSON = {
    "HEROS": {
        "iterations": None,
        "pop_size": None,
        "model_iterations": None,
        "model_pop_size": None,
        "nu": None,
    },
    "ExSTraCS": {
        "iterations": None,
        "N": None,
        "nu": None,
        "rule_compaction": None,
    },
}
```

## Common Mistakes

Use `model_params_json`, not `model_json_param`.

Use HEROS `pop_size`, not `N`.

Use ExSTraCS `N`, not `pop_size`.

Use Python values such as `None`, `True`, and `False` in CLI strings, cfg
files, and notebook dictionaries.
