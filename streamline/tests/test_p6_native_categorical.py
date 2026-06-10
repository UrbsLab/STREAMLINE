from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from streamline.p6_modeling.modeling import ModelingPhaseJob
from streamline.p6_modeling.utils.categorical import (
    NATIVE_CATEGORICAL_MODELS_DEFAULT,
    parse_model_id_csv,
)
from streamline.p6_modeling.utils.modeljob import ModelJob


class DummyExstracsEstimator:
    def __init__(self):
        self.discrete_attribute_limit = None
        self.specified_attributes = None

    def get_params(self, deep=True):
        return {
            "discrete_attribute_limit": self.discrete_attribute_limit,
            "specified_attributes": self.specified_attributes,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class DummyExstracsModel:
    small_name = "ExSTraCS"
    model_name = "ExSTraCS"

    def __init__(self):
        self.model = DummyExstracsEstimator()


def write_native_categorical_dataset(dataset_dir: Path) -> None:
    cv_dir = dataset_dir / "CVDatasets"
    exploratory_dir = dataset_dir / "exploratory"
    cv_dir.mkdir(parents=True, exist_ok=True)
    exploratory_dir.mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(
        {
            "Class": [0, 1, 0, 1],
            "age": [50, 61, 45, 70],
            "sex": ["M", "F", "F", "M"],
            "stage": ["I", "II", "I", "III"],
            "lab": [1.1, 2.2, 1.4, 3.0],
        }
    )
    test = pd.DataFrame(
        {
            "Class": [0, 1],
            "age": [55, 68],
            "sex": ["F", "M"],
            "stage": ["II", "III"],
            "lab": [1.8, 2.9],
        }
    )
    train.to_csv(cv_dir / f"{dataset_dir.name}_CV_0_Train.csv", index=False)
    test.to_csv(cv_dir / f"{dataset_dir.name}_CV_0_Test.csv", index=False)

    feature_meta = {
        "feature_names": ["age", "sex", "stage", "lab"],
        "categorical_mask": [False, True, True, False],
        "quantitative_mask": [True, False, False, True],
        "one_hot": False,
        "one_hot_features": [],
        "categorical_features": ["sex", "stage"],
    }
    with (exploratory_dir / "feature_meta.pickle").open("wb") as handle:
        pickle.dump(feature_meta, handle)
    (exploratory_dir / "feature_meta.json").write_text(json.dumps(feature_meta))


def make_model_job(tmp_path: Path) -> ModelJob:
    exp_root = tmp_path / "out" / "Exp"
    dataset_dir = exp_root / "ToyData"
    write_native_categorical_dataset(dataset_dir)
    return ModelJob(
        full_path=str(dataset_dir),
        output_path=str(tmp_path / "out"),
        experiment_name="Exp",
        cv_count=0,
        outcome_label="Class",
        scoring_metric="balanced_accuracy",
        metric_direction="maximize",
        native_categorical_models=NATIVE_CATEGORICAL_MODELS_DEFAULT,
    )


def test_default_native_categorical_models_include_exstracs():
    parsed = parse_model_id_csv(NATIVE_CATEGORICAL_MODELS_DEFAULT)
    assert "cgb" in parsed
    assert "exstracs" in parsed

    job = ModelingPhaseJob(
        dataset_dir="/tmp/not-used",
        output_path="/tmp",
        experiment_name="not-used",
    )
    assert {"cgb", "exstracs"}.issubset(job.native_categorical_model_ids)


def test_exstracs_native_categorical_data_prep_and_params(tmp_path: Path):
    model_job = make_model_job(tmp_path)
    model = DummyExstracsModel()

    x_train, y_train, x_test, y_test = model_job.data_prep(model)
    model_job._configure_native_categorical_model(model)

    assert model_job.categorical_encoding_mode == "native"
    assert list(x_train.columns) == ["age", "sex", "stage", "lab"]
    assert list(x_test.columns) == ["age", "sex", "stage", "lab"]
    assert y_train.tolist() == [0, 1, 0, 1]
    assert y_test.tolist() == [0, 1]
    assert model.model.discrete_attribute_limit == "d"
    np.testing.assert_array_equal(model.model.specified_attributes, np.asarray([1, 2], dtype=int))
    report = model_job._categorical_report()
    assert report["native_categorical_features"] == ["sex", "stage"]
    assert report["native_categorical_indices"] == [1, 2]


def test_exstracs_params_are_not_forced_outside_native_mode(tmp_path: Path):
    model_job = make_model_job(tmp_path)
    model_job.categorical_encoding_mode = "none"
    model_job.categorical_feature_names = []
    model = DummyExstracsModel()

    model_job._configure_native_categorical_model(model)

    assert model.model.discrete_attribute_limit is None
    assert model.model.specified_attributes is None
