# tests/test_p6_modeling.py
import os
import pickle
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.calibration import CalibratedClassifierCV

# Phase 6 entrypoint
from streamline.p6_modeling.modeling import ModelingPhaseJob


# -------------------------
# Helpers
# -------------------------
def _make_cv_dataset(root: Path, dataset_name: str = "ToyData", n_features: int = 8, n_samples: int = 300):
    """
    Creates <root>/<dataset_name>/CVDatasets with a single CV split (0) and writes Train/Test CSVs.
    Columns: [Class, f0..f{n_features-1}]
    """
    ds_dir = root / dataset_name
    cv_dir = ds_dir / "CVDatasets"
    (cv_dir).mkdir(parents=True, exist_ok=True)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 3),
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=2,
        class_sep=1.2,
        random_state=42,
    )

    # simple split
    idx = np.arange(n_samples)
    rng = np.random.default_rng(7)
    rng.shuffle(idx)
    split = int(0.7 * n_samples)
    tr, te = idx[:split], idx[split:]

    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "Class", y)

    train = df.iloc[tr].copy()
    test = df.iloc[te].copy()

    train_path = cv_dir / f"{dataset_name}_CV_0_Train.csv"
    test_path = cv_dir / f"{dataset_name}_CV_0_Test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    return ds_dir


def _run_p6_for_models(exp_root: Path, dataset_dir: Path, models_csv: str, calibrate=True):
    """
    Runs ModelingPhaseJob directly for a single dataset with n_splits=1.
    """
    ModelingPhaseJob(
        dataset_dir=str(dataset_dir),
        outcome_label="Class",
        model_type="BinaryClassification",
        instance_label=None,
        n_splits=1,
        models=models_csv,              # comma-separated: e.g., "LR,SVM,NB"
        # calibration flows into BaseModel via construction
        calibrate=calibrate,
        calibrate_method="sigmoid",
        calibrate_cv=3,
        # ModelJob settings
        output_path=str(exp_root.parent),         # <output_path>
        experiment_name=str(exp_root.name),       # <experiment_name>
        scoring_metric="balanced_accuracy",
        metric_direction="maximize",
        n_trials=2,                 # keep the test fast
        timeout=30,                 # seconds budget for optuna
        training_subsample=0,
        uniform_fi=False,
        save_plot=False,
        random_state=123,
    ).run()


def _assert_artifacts(dataset_dir: Path, small_name: str):
    """
    Verifies pickled model + metrics and a sensible probability interface post-calibration.
    """
    # pickled model
    model_pkl = dataset_dir / "models" / "pickledModels" / f"{small_name}_0.pickle"
    assert model_pkl.exists(), f"Missing model pickle for {small_name}"

    # metrics
    metrics_pkl = dataset_dir / "model_evaluation" / "pickled_metrics" / f"{small_name}_CV_0_metrics.pickle"
    assert metrics_pkl.exists(), f"Missing metrics pickle for {small_name}"

    # load and inspect calibration / proba
    with open(model_pkl, "rb") as f:
        est = pickle.load(f)

    # The calibrated estimator is either a CalibratedClassifierCV, or exposes predict_proba
    has_proba = hasattr(est, "predict_proba")
    is_calibrated = isinstance(est, CalibratedClassifierCV)

    # We accept either (SVM may need calibration to expose calibrated_classifiers_)
    assert has_proba or is_calibrated, f"Expected calibrated or proba-capable model for {small_name}"


# -------------------------
# Tests
# -------------------------
@pytest.fixture
def phase6_layout(tmp_path: Path):
    """
    Creates the Phase-1 style layout:
      <tmp>/out/<ExpName>/<Dataset>/CVDatasets/*.csv
    and returns (output_path, experiment_path, dataset_dir).
    """
    out = tmp_path / "out"
    exp = out / "P6Exp"
    exp.mkdir(parents=True, exist_ok=True)
    ds_dir = _make_cv_dataset(exp, dataset_name="ToyData", n_features=10, n_samples=260)
    # phase folders that Phase 6 expects to exist or will create on demand
    (ds_dir / "models").mkdir(exist_ok=True)
    (ds_dir / "model_evaluation").mkdir(exist_ok=True)
    (ds_dir / "runtime").mkdir(exist_ok=True)
    (exp / "jobsCompleted").mkdir(exist_ok=True)
    (exp / "jobs").mkdir(exist_ok=True)
    (exp / "logs").mkdir(exist_ok=True)
    return out, exp, ds_dir


@pytest.mark.parametrize(
    "models_csv, expected_small_names",
    [
        ("LR", ["LR"]),
        ("NB", ["NB"]),
        ("SVM", ["SVM"]),
        ("LR,NB,SVM", ["LR", "NB", "SVM"]),
    ],
)
def test_p6_modeling_smoke(phase6_layout, models_csv, expected_small_names):
    """
    Smoke test Phase 6 modeling for LR, NB, SVM (individually and together).
    Verifies model/metrics artifacts and that the estimator is calibrated or proba-capable.
    """
    out, exp, ds_dir = phase6_layout

    # Run once per models set
    _run_p6_for_models(exp_root=exp, dataset_dir=ds_dir, models_csv=models_csv, calibrate=True)

    # Check artifacts per expected small_name
    for sn in expected_small_names:
        _assert_artifacts(ds_dir, small_name=sn)


def test_p6_modeling_without_calibration(phase6_layout):
    """
    Also run once without calibration to ensure the pipeline still writes artifacts.
    """
    out, exp, ds_dir = phase6_layout
    _run_p6_for_models(exp_root=exp, dataset_dir=ds_dir, models_csv="LR,SVM,NB", calibrate=False)

    for sn in ["LR", "SVM", "NB"]:
        model_pkl = ds_dir / "models" / "pickledModels" / f"{sn}_0.pickle"
        metrics_pkl = ds_dir / "model_evaluation" / "pickled_metrics" / f"{sn}_CV_0_metrics.pickle"
        assert model_pkl.exists()
        assert metrics_pkl.exists()
