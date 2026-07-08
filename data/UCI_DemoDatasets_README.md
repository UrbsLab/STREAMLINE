# UCI Demo Datasets for STREAMLINE

These datasets are normalized from official UCI Machine Learning Repository sources into the same style used by the built-in demo data: one task-specific data folder with CSV files, plus optional feature-type CSVs. Each dataset is split deterministically with seed 42: the normal demo folder contains the 80% training split and the matching `UCIRep*` folder contains the held-out 20% replication split. Classification splits are stratified by `Class`; the regression split is random.

## Selected datasets

| STREAMLINE task | Local CSV | UCI source | Outcome | Notes |
| --- | --- | --- | --- | --- |
| Binary classification | `data/UCIBinaryClassification/hcc_survival.csv` and `_copy.csv` | HCC Survival | `Class` | 132 training rows and 33 held-out replication rows. The target is one-year survival, encoded as `0=dies` and `1=lives`. Missing `?` values are stored as `NA`. |
| Multiclass classification | `data/UCIMulticlassClassification/student_dropout_academic_success.csv` and `_copy.csv` | Predict Students' Dropout and Academic Success | `Class` | 3539 training rows and 885 held-out replication rows. The original labels are normalized as `0=Dropout`, `1=Enrolled`, and `2=Graduate`. UCI reports no missing values after preprocessing; this demo adds deterministic synthetic missingness with seed 42 to selected categorical and quantitative features and stores those cells as `NA`. |
| Regression | `data/UCIRegression/auto_mpg.csv` and `_copy.csv` | Auto MPG | `MPG` | 318 training rows and 80 held-out replication rows. Predicts miles per gallon from mixed vehicle attributes. The high-cardinality UCI `car_name` field is dropped from the modeling CSV. Missing horsepower values are stored as `NA`. |

## Source URLs

- HCC Survival: https://archive.ics.uci.edu/dataset/423/hcc+survival
- HCC Survival raw zip file: https://archive.ics.uci.edu/static/public/423/hcc+survival.zip
- Predict Students' Dropout and Academic Success: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- Predict Students' Dropout and Academic Success raw zip file: https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip
- Auto MPG: https://archive.ics.uci.edu/dataset/9/auto+mpg
- Auto MPG raw data file: https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data


## Replication demo folders

The following folders mirror the cleaned schemas above and contain held-out 20% replication splits for Phase 10 examples:

- `data/UCIRepBinaryClassification/hcc_survival_rep.csv`
- `data/UCIRepMulticlassClassification/student_dropout_academic_success_rep.csv`
- `data/UCIRepRegression/auto_mpg_rep.csv`

## Feature-type files

The files in `data/UCIFeatureTypes/` are one-column CSVs with header `Feature`, which matches the Phase 1 CLI path-based `--categorical_features` and `--quantitative_features` loaders.

## Example Phase 1 commands

```bash
python -m streamline.p1_data_process.p1_cli --data_path data/UCIBinaryClassification --output_path out --experiment_name UCIHCCSurvival --outcome_label Class --outcome_type Binary --instance_label InstanceID --categorical_features data/UCIFeatureTypes/hcc_survival_categorical_features.csv --quantitative_features data/UCIFeatureTypes/hcc_survival_quantitative_features.csv

python -m streamline.p1_data_process.p1_cli --data_path data/UCIMulticlassClassification --output_path out --experiment_name UCIStudentDropout --outcome_label Class --outcome_type Multiclass --instance_label InstanceID --categorical_features data/UCIFeatureTypes/student_dropout_categorical_features.csv --quantitative_features data/UCIFeatureTypes/student_dropout_quantitative_features.csv

python -m streamline.p1_data_process.p1_cli --data_path data/UCIRegression --output_path out --experiment_name UCIAutoMPG --outcome_label MPG --outcome_type Continuous --instance_label InstanceID --categorical_features data/UCIFeatureTypes/auto_mpg_categorical_features.csv --quantitative_features data/UCIFeatureTypes/auto_mpg_quantitative_features.csv
```
