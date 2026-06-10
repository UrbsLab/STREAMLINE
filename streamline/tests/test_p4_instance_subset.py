from __future__ import annotations

from pathlib import Path

import pandas as pd

import streamline.p4_feature_importance.importance as importance_module
from streamline.p4_feature_importance.importance import FeatureImportance
from streamline.p4_feature_importance.p4_runner import P4Runner


class RecordingImportanceModel:
    path_name = "recording"
    model_name = "Recording"
    small_name = "REC"

    def __init__(self, *, uses_instance_subset: bool):
        self.uses_instance_subset = uses_instance_subset
        self.fit_rows = None
        self.fit_index = None
        self.columns = []

    def fit(self, X, y):
        self.fit_rows = len(X)
        self.fit_index = list(X.index)
        self.columns = list(X.columns)
        return self

    def get_scores(self):
        return {column: float(index) for index, column in enumerate(self.columns)}

    def get_params(self):
        return {"uses_instance_subset": self.uses_instance_subset}


def make_cv_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    exp = tmp_path / "out" / "Exp"
    ds = exp / "Toy"
    cv = ds / "CVDatasets"
    cv.mkdir(parents=True)
    train = pd.DataFrame(
        {
            "Class": [0, 1] * 5,
            "x1": range(10),
            "x2": range(10, 20),
        }
    )
    test = pd.DataFrame({"Class": [0, 1], "x1": [100, 101], "x2": [200, 201]})
    train_path = cv / "Toy_CV_0_Train.csv"
    test_path = cv / "Toy_CV_0_Test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return exp, train_path, test_path


def test_p4_runner_default_instance_subset_matches_legacy_default(tmp_path: Path):
    exp = tmp_path / "out" / "Exp"
    exp.mkdir(parents=True)

    runner = P4Runner(output_path=str(tmp_path / "out"), experiment_name="Exp")

    assert runner.instance_subset == 2000


def test_p4_instance_subset_only_samples_subset_aware_models(monkeypatch, tmp_path: Path):
    exp, train_path, test_path = make_cv_dataset(tmp_path)
    models: list[RecordingImportanceModel] = []

    def fake_load_importance(model_id, **params):
        model = RecordingImportanceModel(uses_instance_subset=True)
        models.append(model)
        return model

    monkeypatch.setattr(importance_module, "load_importance", fake_load_importance)

    FeatureImportance(
        cv_train_path=str(train_path),
        cv_test_path=str(test_path),
        experiment_path=str(exp),
        model_id="recording",
        outcome_label="Class",
        random_state=1,
        instance_subset=3,
    ).run()

    assert models[0].fit_rows == 3


def test_p4_instance_subset_does_not_sample_other_models(monkeypatch, tmp_path: Path):
    exp, train_path, test_path = make_cv_dataset(tmp_path)
    models: list[RecordingImportanceModel] = []

    def fake_load_importance(model_id, **params):
        model = RecordingImportanceModel(uses_instance_subset=False)
        models.append(model)
        return model

    monkeypatch.setattr(importance_module, "load_importance", fake_load_importance)

    FeatureImportance(
        cv_train_path=str(train_path),
        cv_test_path=str(test_path),
        experiment_path=str(exp),
        model_id="recording",
        outcome_label="Class",
        random_state=1,
        instance_subset=3,
    ).run()

    assert models[0].fit_rows == 10
