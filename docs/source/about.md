# About STREAMLINE

STREAMLINE was built to make reproducible supervised machine learning workflows
more accessible for biomedical and general tabular datasets. The current
refactor keeps the original end-to-end spirit while separating the pipeline
into explicit P1-P11 phase modules.

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

## Current Learning Tasks

STREAMLINE supports binary classification, multiclass classification, and
regression. P7 ensembles are currently classification-only.
