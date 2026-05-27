# UCI Demo Datasets for STREAMLINE

These datasets are normalized from official UCI Machine Learning Repository sources into the same style used by the built-in demo data: one task-specific data folder with CSV files, plus optional feature-type CSVs.

## Selected datasets

| STREAMLINE task | Local CSV | UCI source | Outcome | Notes |
| --- | --- | --- | --- | --- |
| Binary classification | `data/UCIBinaryClassification/heart_disease_cleveland.csv` and `_copy.csv` | Heart Disease, processed Cleveland data | `Class` | The original `num` target is binarized as `0` for no disease and `1` for disease presence. Missing `?` values are stored as `NA`. |
| Multiclass classification | `data/UCIMulticlassClassification/dermatology.csv` and `_copy.csv` | Dermatology | `Class` | The original 1-6 disease codes are normalized to 0-5 for model compatibility. Missing `?` values are stored as `NA`. |
| Regression | `data/UCIRegression/auto_mpg.csv` and `_copy.csv` | Auto MPG | `MPG` | Predicts miles per gallon from mixed vehicle attributes. The high-cardinality UCI `car_name` field is dropped from the modeling CSV. Missing horsepower values are stored as `NA`. |

## Source URLs

- Heart Disease: https://archive.ics.uci.edu/dataset/45/heart+disease
- Heart Disease raw processed Cleveland file: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
- Dermatology: https://archive.ics.uci.edu/dataset/33/dermatology
- Dermatology raw data file: https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data
- Auto MPG: https://archive.ics.uci.edu/dataset/9/auto+mpg
- Auto MPG raw data file: https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data


## Replication demo folders

The following folders mirror the cleaned schemas above and can be used by Phase 10 replication examples:

- `data/UCIRepBinaryClassification/heart_disease_cleveland_rep.csv`
- `data/UCIRepMulticlassClassification/dermatology_rep.csv`
- `data/UCIRepRegression/auto_mpg_rep.csv`

## Feature-type files

The files in `data/UCIFeatureTypes/` are one-column CSVs with header `Feature`, which matches the Phase 1 CLI path-based `--categorical_features` and `--quantitative_features` loaders.

## Example Phase 1 commands

```bash
python -m streamline.p1_data_process.p1_cli --data_path data/UCIBinaryClassification --output_path out --experiment_name UCIHeartDisease --outcome_label Class --outcome_type Binary --instance_label InstanceID --categorical_features data/UCIFeatureTypes/heart_disease_categorical_features.csv --quantitative_features data/UCIFeatureTypes/heart_disease_quantitative_features.csv

python -m streamline.p1_data_process.p1_cli --data_path data/UCIMulticlassClassification --output_path out --experiment_name UCIDermatology --outcome_label Class --outcome_type Multiclass --instance_label InstanceID --categorical_features data/UCIFeatureTypes/dermatology_categorical_features.csv --quantitative_features data/UCIFeatureTypes/dermatology_quantitative_features.csv

python -m streamline.p1_data_process.p1_cli --data_path data/UCIRegression --output_path out --experiment_name UCIAutoMPG --outcome_label MPG --outcome_type Continuous --instance_label InstanceID --categorical_features data/UCIFeatureTypes/auto_mpg_categorical_features.csv --quantitative_features data/UCIFeatureTypes/auto_mpg_quantitative_features.csv
```
