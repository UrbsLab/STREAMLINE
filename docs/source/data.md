# Datasets

## Input Data Requirements

STREAMLINE expects tabular supervised learning datasets.

1. Use `.csv`, `.tsv`, or `.txt` files with a header row.
2. Rows are instances and columns are variables.
3. Include one outcome column specified by `outcome_label`.
4. Include an optional instance identifier column specified by `instance_label`.
5. Include optional feature-type files listing categorical and quantitative feature names.
6. Encode missing values as blank cells, `NA`, `NaN`, or `?`; avoid numeric placeholders such as `-99` unless they are real values.
7. Keep replication datasets aligned with the original dataset's feature names and outcome/instance labels.

Recommended preparation before P1:

* Remove free-text, image, raw time-series, or other unstructured columns unless you have already converted them to tabular features.
* Decide whether high-cardinality identifier-like fields should be `instance_label`, ignored, or removed before modeling.
* Check that the outcome column has the interpretation you expect. Binary labels should be documented so downstream metrics are not ambiguous.
* Keep a copy of the raw source data outside the STREAMLINE output directory.

The current code supports:

| Task | Example `outcome_type` | Notes |
| --- | --- | --- |
| Binary classification | `Binary` | Standard classification metrics, ROC, PRC, and calibration outputs. |
| Multiclass classification | `Multiclass` | Macro/micro summaries where applicable. |
| Regression | `Continuous` | Regression metrics and residual-style outputs. |

## Feature Types

You can provide feature type files through:

```text
categorical_features = path/to/categorical_features.csv
quantitative_features = path/to/quantitative_features.csv
```

Each file should contain feature names, one per row or in a single column. If
feature type files are omitted, P1 can infer categorical features using
`categorical_cutoff`.

Supplying feature-type files is recommended for real analyses because numeric
codes are common in biomedical and administrative data. A coded feature such as
`1`, `2`, `3` may be categorical even though it looks numeric to Python.

### One-Hot Encoding And Native Categorical Models

By default, P1 one-hot encodes non-binary categorical features. Set
`one_hot_encoding = False` when you want later models to handle raw categorical
columns directly.

When `one_hot_encoding` is false, P6 only runs models listed in
`native_categorical_models` by default. Unsupported explicitly requested models
raise an error instead of silently changing the feature representation.

## Included UCI Demo Datasets

The repository includes deterministic 80:20 train/replication splits for three
UCI-derived demos:

| Task | Training folder | Replication folder | Outcome |
| --- | --- | --- | --- |
| Binary classification | `data/UCIBinaryClassification` | `data/UCIRepBinaryClassification` | `Class` |
| Multiclass classification | `data/UCIMulticlassClassification` | `data/UCIRepMulticlassClassification` | `Class` |
| Regression | `data/UCIRegression` | `data/UCIRepRegression` | `MPG` |

The demo sources are:

* HCC survival: [UCI HCC Survival](https://archive.ics.uci.edu/dataset/423/hcc+survival)
* Student dropout and academic success: [UCI Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
* Auto MPG: [UCI Auto MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg)

Feature type files for the demos live in `data/UCIFeatureTypes/`.

The normal demo folders contain training/development splits. The matching
`UCIRep*` folders contain held-out replication splits created for STREAMLINE
demonstration and testing.

## Replication Data

Replication datasets are external validation inputs for P10. They are not
assumed to be official test splits from UCI. The included demo replication
folders are deterministic held-out 20% splits made to exercise P10 and
replication reporting.

## Ignored Columns

Identifier-like fields should be supplied as `instance_label` or ignored before
modeling. For Auto MPG, the car name is treated as an identifier-like field in
the demo preparation rather than a normal predictive feature.
