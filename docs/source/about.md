# About STREAMLINE

STREAMLINE was built to make reproducible supervised machine learning workflows
more accessible for biomedical and general tabular datasets. The current
refactor keeps the original end-to-end spirit while separating the pipeline
into explicit P1-P11 phase modules.

STREAMLINE is best thought of as a transparent analysis framework rather than
a black-box model picker. It applies a consistent pipeline, runs multiple
modeling perspectives, saves intermediate artifacts, and generates reports so
users can inspect what happened at every stage.

## What STREAMLINE Is Useful For

* running a structured supervised-learning analysis on tabular data
* comparing multiple algorithms under the same CV and preprocessing design
* producing baseline models and summary reports for a new dataset
* evaluating feature importance and selected features across folds
* applying trained workflows to replication data
* teaching or demonstrating an end-to-end AutoML-style workflow

## What STREAMLINE Automates

* Dataset loading and exploratory summaries
* Cross-validation partitioning
* Missing-value imputation and scaling
* Optional SMOTE/SMOTENC oversampling
* Feature learning
* Feature importance scoring
* Feature selection
* Base model training and evaluation
* Classification ensembles
* Summary statistics and dataset comparison
* External replication
* Standard and replication PDF reporting

## What Users Still Control

Users should still make scientific and domain-specific decisions about:

* Outcome definition
* Feature inclusion and exclusion
* Feature type declarations
* Leakage checks
* Metric choice
* Class-label interpretation
* Replication dataset suitability

STREAMLINE does not replace domain review. Users should still check for data
leakage, unclear outcome definitions, inappropriate feature encodings,
collection bias, and metrics that do not match the scientific question.

## Current Learning Tasks

STREAMLINE supports binary classification, multiclass classification, and
regression. P7 ensembles are currently classification-only.

## What STREAMLINE Does Not Automate

STREAMLINE does not perform feature extraction from unstructured text, images,
audio, video, or raw time-series streams. It also does not decide whether a
feature is scientifically valid, whether a dataset is biased, or whether a
replication cohort is comparable to the training cohort.
