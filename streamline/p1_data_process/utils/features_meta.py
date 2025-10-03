import os
import json
import pickle

def build_feature_meta(dataset, categorical_features, quantitative_features,
                       one_hot, one_hot_features, engineered_features):
    # Build masks in final column order of dataset.data
    cols = list(dataset.data.columns)
    outcome = dataset.outcome_label
    instance = dataset.instance_label
    feature_cols = [c for c in cols if c not in [outcome, instance]]

    cat_set = set(categorical_features)
    quant_set = set(quantitative_features)

    meta = {
        "feature_names": feature_cols,
        "categorical_mask": [c in cat_set for c in feature_cols],
        "quantitative_mask": [c in quant_set for c in feature_cols],
        "original_dtypes": {c: str(dataset.data[c].dtype) for c in cols},
        "one_hot": bool(one_hot),
        "one_hot_features": list(one_hot_features),
        "engineered_features": list(engineered_features),
        "outcome_label": outcome,
        "instance_label": instance,
    }
    return meta

def save_feature_meta(experiment_path, dataset_name, feature_meta):
    exp_ds = os.path.join(experiment_path, dataset_name, "exploratory")
    if not os.path.exists(exp_ds):
        os.makedirs(exp_ds)
    with open(os.path.join(exp_ds, "feature_meta.pickle"), "wb") as f:
        pickle.dump(feature_meta, f)
    with open(os.path.join(exp_ds, "feature_meta.json"), "w") as f:
        json.dump(feature_meta, f)
