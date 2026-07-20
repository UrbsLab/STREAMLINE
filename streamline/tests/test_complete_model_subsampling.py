from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from streamline.p6_modeling.models.binary_classification.heros import HEROSClassifier
from streamline.p6_modeling.models.multiclass_classification.heros import HEROSMulticlassClassifier
from streamline.p6_modeling.utils.modeljob import ModelJob


class RecordingEstimator:
    def __init__(self):
        self.feature_importances_ = np.asarray([0.6, 0.4])

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, x_train, y_train):
        return self

    def predict(self, x_test):
        return np.zeros(len(x_test), dtype=int)


class RecordingModel:
    model_type = "Binary"
    model_name = "Recording Model"
    color = "black"
    is_single = True
    optuna_report = {}

    def __init__(self, small_name: str, subsampling_allowed: bool):
        self.small_name = small_name
        self.subsampling_allowed = subsampling_allowed
        self.model = RecordingEstimator()
        self.params = {"constant": 1}
        self.fit_rows = None
        self.fit_class_counts = None

    def fit(self, x_train, y_train, n_trials, timeout, feature_names=None):
        self.fit_rows = len(y_train)
        values, counts = np.unique(np.asarray(y_train), return_counts=True)
        self.fit_class_counts = {str(value): int(count) for value, count in zip(values, counts)}
        self.model.fit(x_train, y_train)

    def model_evaluation(self, x_test, y_test):
        return {"balanced_accuracy": 1.0}, {"roc": {}, "prc": {}}


def make_modeljob_dataset(
    tmp_path: Path,
    train_classes: list[int] | None = None,
) -> tuple[Path, Path]:
    output_path = tmp_path / "out"
    experiment_path = output_path / "SubsampleExp"
    dataset_path = experiment_path / "ToyData"
    cv_path = dataset_path / "CVDatasets"
    cv_path.mkdir(parents=True)

    if train_classes is None:
        train_classes = ([0] * 10) + ([1] * 3)
    row_count = len(train_classes)
    train = pd.DataFrame({
        "Class": train_classes,
        "f0": np.arange(row_count),
        "f1": np.arange(row_count, row_count * 2),
    })
    test = pd.DataFrame({
        "Class": [0, 1, 0, 1],
        "f0": np.arange(4),
        "f1": np.arange(4, 8),
    })
    train.to_csv(cv_path / "ToyData_CV_0_Train.csv", index=False)
    test.to_csv(cv_path / "ToyData_CV_0_Test.csv", index=False)
    return output_path, dataset_path


def run_recording_model(
    output_path: Path,
    dataset_path: Path,
    model: RecordingModel,
    training_subsample: int = 6,
):
    job = ModelJob(
        full_path=str(dataset_path),
        output_path=str(output_path),
        experiment_name="SubsampleExp",
        cv_count=0,
        outcome_label="Class",
        training_subsample=training_subsample,
        n_trials=1,
        timeout=1,
        random_state=7,
    )
    job.run_model(model)


def test_training_subsample_uses_model_opt_in(tmp_path: Path):
    output_path, dataset_path = make_modeljob_dataset(tmp_path)

    allowed_model = RecordingModel("Allowed", subsampling_allowed=True)
    run_recording_model(output_path, dataset_path, allowed_model)
    assert allowed_model.fit_rows == 6
    assert allowed_model.fit_class_counts == {"0": 3, "1": 3}

    blocked_model = RecordingModel("Blocked", subsampling_allowed=False)
    run_recording_model(output_path, dataset_path, blocked_model)
    assert blocked_model.fit_rows == 13
    assert blocked_model.fit_class_counts == {"0": 10, "1": 3}


def test_training_subsample_balances_to_minority_class(tmp_path: Path):
    output_path, dataset_path = make_modeljob_dataset(tmp_path, train_classes=([0] * 18) + ([1] * 2))

    balanced_model = RecordingModel("Balanced", subsampling_allowed=True)
    run_recording_model(output_path, dataset_path, balanced_model, training_subsample=10)
    assert balanced_model.fit_rows == 4
    assert balanced_model.fit_class_counts == {"0": 2, "1": 2}


def test_training_subsample_can_use_internal_stratified_strategy(tmp_path: Path):
    output_path, dataset_path = make_modeljob_dataset(tmp_path, train_classes=([0] * 18) + ([1] * 2))

    stratified_model = RecordingModel("Stratified", subsampling_allowed=True)
    stratified_model.subsampling_strategy = "stratified"
    run_recording_model(output_path, dataset_path, stratified_model, training_subsample=10)
    assert stratified_model.fit_rows == 10
    assert stratified_model.fit_class_counts == {"0": 9, "1": 1}


def test_heros_models_allow_training_subsample():
    assert HEROSClassifier.subsampling_allowed is True
    assert HEROSClassifier.subsampling_strategy == "balanced"
    assert HEROSClassifier.undersampling_strategy == "auto"
    assert HEROSMulticlassClassifier.subsampling_allowed is True
    assert HEROSMulticlassClassifier.subsampling_strategy == "balanced"
    assert HEROSMulticlassClassifier.undersampling_strategy == "auto"
