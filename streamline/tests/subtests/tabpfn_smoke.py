from pathlib import Path

import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from streamline.p6_modeling.modeling import ModelingPhaseJob


def make_binary_cv_dataset(root: Path) -> Path:
    dataset_dir = root / "out" / "TabPFNExp" / "ToyBinary"
    cv_dir = dataset_dir / "CVDatasets"
    cv_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "models").mkdir(exist_ok=True)
    (dataset_dir / "model_evaluation").mkdir(exist_ok=True)
    (dataset_dir.parent / "jobsCompleted").mkdir(exist_ok=True)

    x_values, y_values = make_classification(
        n_samples=72,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        class_sep=1.2,
        random_state=17,
    )
    data = pd.DataFrame(x_values, columns=[f"f{i}" for i in range(x_values.shape[1])])
    data.insert(0, "Class", y_values)

    indices = np.arange(len(data))
    rng = np.random.default_rng(17)
    rng.shuffle(indices)
    train_indices = indices[:50]
    test_indices = indices[50:]

    data.iloc[train_indices].to_csv(cv_dir / "ToyBinary_CV_0_Train.csv", index=False)
    data.iloc[test_indices].to_csv(cv_dir / "ToyBinary_CV_0_Test.csv", index=False)
    return dataset_dir


def test_phase6_skips_tabpfn_without_token_and_runs_other_models(tmp_path, monkeypatch):
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    dataset_dir = make_binary_cv_dataset(tmp_path)

    with pytest.warns(RuntimeWarning, match="TABPFN_TOKEN is not set"):
        ModelingPhaseJob(
            dataset_dir=str(dataset_dir),
            outcome_label="Class",
            model_type="Binary",
            n_splits=1,
            models="TabPFN,NB",
            output_path=str(tmp_path / "out"),
            experiment_name="TabPFNExp",
            scoring_metric="balanced_accuracy",
            metric_direction="maximize",
            n_trials=1,
            timeout=20,
            random_state=17,
        ).run_all_model_cv_jobs()

    nb_metrics = dataset_dir / "model_evaluation" / "metrics_by_cv" / "NB_CV_0.json"
    nb_pickle = dataset_dir / "models" / "pickledModels" / "NB_0.pickle"
    tabpfn_pickle = dataset_dir / "models" / "pickledModels" / "TabPFN_0.pickle"
    phase_flag = dataset_dir.parent / "jobsCompleted" / "job_modeling_ToyBinary.txt"

    assert nb_metrics.exists()
    assert nb_pickle.exists()
    assert not tabpfn_pickle.exists()
    assert phase_flag.exists()

    with open(nb_metrics, "r") as metrics_file:
        payload = json.load(metrics_file)
    assert "balanced_accuracy" in payload["metrics"]


@pytest.mark.skipif(
    not os.environ.get("TABPFN_TOKEN"),
    reason="TABPFN_TOKEN is required for the optional TabPFN fit smoke test.",
)
def test_tabpfn_binary_wrapper_fit_with_token():
    pytest.importorskip("tabpfn")
    from streamline.p6_modeling.models.binary_classification.tabpfn import TabPFNClassifier

    x_values, y_values = make_classification(
        n_samples=48,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        class_sep=1.0,
        random_state=23,
    )
    x_frame = pd.DataFrame(x_values, columns=[f"f{i}" for i in range(x_values.shape[1])])
    y_series = pd.Series(y_values)

    model = TabPFNClassifier(
        cv_folds=2,
        scoring_metric="balanced_accuracy",
        metric_direction="maximize",
        random_state=23,
        n_ensemble_configurations=8,
    )
    model.fit(x_frame, y_series, n_trails=1, timeout=60)

    predictions = model.predict(x_frame.head(5))
    assert len(predictions) == 5
