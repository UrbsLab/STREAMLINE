# Tips

## Start With A Dry Run

Before launching a full config, inspect the resolved phase calls:

```bash
python run.py -c run_configs/uci_binary_hcc.cfg --dry_run
```

This catches most path, phase toggle, and parameter-name mistakes early.

## Keep Parameter Names Consistent

The current notebooks, `.cfg` files, and CLI arguments are aligned around the
same parameter names where possible. Prefer copying one of the included UCI
configs and editing values rather than starting from an empty file.

## Choose Metrics By Task

Good defaults:

| Task | Metric |
| --- | --- |
| Binary classification | `balanced_accuracy` |
| Multiclass classification | `balanced_accuracy` |
| Regression | `explained_variance` or `pearson_correlation` |

Use `metric_direction = maximize` for these defaults. Use `minimize` for error
metrics such as mean absolute error.

## Use Replication As External Validation

P10 should be used for data that were not part of training/CV. The included UCI
replication folders are deterministic held-out splits for demonstration.

## SMOTE Guidance

Enable P2 SMOTE only for classification tasks and only when class imbalance is
large enough to justify oversampling:

```ini
[p2]
smote = True
smote_method = auto
```

`auto` chooses SMOTENC when categorical features are present.

## Native Categorical Handling

If P1 uses `one_hot_encoding = False`, P6 should run only native categorical
models unless you explicitly add support for another model. The default native
categorical list includes CatBoost/CGB and ExSTraCS.

## Faster Test Runs

For quick smoke tests:

* Use `n_splits = 3`.
* Use a small model list such as `NB,LR,DT`.
* Lower `n_trials`.
* Set report `enable_plots = False` when checking report logic only.
* Use `--only` or `--stop_after` for targeted config runs.
