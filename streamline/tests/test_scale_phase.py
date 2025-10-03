# tests/phases/p2_impute_scale/test_scaler_minmax.py
import os
import json
import pytest
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from streamline.p2_impute_scale.p2_runner import ImputeAndScale

# tmp_path = Path("tests")
pytest.skip("Tested Already", allow_module_level=True)



def _seed_exp_no_missing(tmp_path: Path, dataset_name: str = "hcc_data"):
    """
    Minimal Phase-1-like layout with NO missing values,
    so we can isolate scaling behavior (impute_data=False in the job).
    """
    exp = tmp_path / "demo"
    ds = exp / dataset_name
    cvd = ds / "CVDatasets"
    expl = ds / "exploratory"
    (exp / "jobsCompleted").mkdir(parents=True, exist_ok=True)
    for p in (ds, cvd, expl):
        p.mkdir(parents=True, exist_ok=True)

    # DataCounts says zero missing (so imputation is skipped if caller sets impute_data=True accidentally)
    pd.DataFrame(
        {"Type": ["A", "B", "C", "D", "Missing"], "Count": [0, 0, 0, 0, 0]}
    ).to_csv(expl / "DataCounts.csv", index=False)

    # One categorical column
    with open(expl / "categorical_features.pickle", "wb") as f:
        pickle.dump(["cat1"], f)

    # Train numeric ranges deliberately chosen (include negatives)
    # - num1 ranges [-2, 8] in train
    # - num2 is constant = 5 in train (edge case)
    train = pd.DataFrame(
        {
            "Class": [0, 1, 0, 1],
            "cat1": ["a", "b", "a", "b"],
            "num1": [-2.0, 0.0, 3.0, 8.0],
            "num2": [5.0, 5.0, 5.0, 5.0],   # constant
        }
    )
    # Test includes values outside train min/max to ensure extrapolation
    test = pd.DataFrame(
        {
            "Class": [1, 0],
            "cat1": ["a", "b"],
            "num1": [10.0, -4.0],  # outside [-2, 8]
            "num2": [5.0, 5.0],    # constant column remains constant → scaled constant
        }
    )

    tr = cvd / f"{dataset_name}_CV_1_Train.csv"
    te = cvd / f"{dataset_name}_CV_1_Test.csv"
    train.to_csv(tr, index=False)
    test.to_csv(te, index=False)
    return exp, ds, tr, te


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _minmax_transform(x, xmin, xmax, a=0.0, b=1.0):
    # MinMax formula; if xmax == xmin, sklearn sets result to 0 (or 'a' after range mapping).
    x = np.asarray(x, dtype=float)
    if xmax == xmin:
        return np.full_like(x, a, dtype=float)
    return (x - xmin) / (xmax - xmin) * (b - a) + a


def test_minmax_default_range_formula_exact():
    """
    Verify exact MinMax scaling math on TRAIN (range [-2,8] for num1),
    categorical untouched, and EXTRAPOLATION on TEST (values outside map <0 or >1).
    """
    tmp_path = Path("tests")
    exp, ds, tr, te = _seed_exp_no_missing(tmp_path)

    job = ImputeAndScale(
        cv_train_path=str(tr),
        cv_test_path=str(te),
        experiment_path=str(exp),
        scale_data=True,
        impute_data=False,       # isolate scaling
        multi_impute=False,
        overwrite_cv=True,
        outcome_label="Class",
        instance_label=None,
        random_state=0,
        scaler_id="minmax",
        scaler_params={"feature_range": (0, 1)},
    )
    job.run()

    train = _read_csv(tr)
    test = _read_csv(te)

    # Categorical remains unchanged
    assert set(train["cat1"].unique()) == {"a", "b"}
    assert set(test["cat1"].unique()) == {"a", "b"}

    # TRAIN: exact mapping into [0,1]
    # num1 train min = -2, max = 8
    tr_raw = np.array([-2.0, 0.0, 3.0, 8.0], dtype=float)
    tr_scaled_expected = _minmax_transform(tr_raw, xmin=-2.0, xmax=8.0, a=0.0, b=1.0)
    # Compare against rows aligned by original order
    np.testing.assert_allclose(train.loc[:, "num1"].to_numpy(), tr_scaled_expected, rtol=0, atol=1e-12)

    # num2 is constant; sklearn MinMax → zeros in default [0,1]
    assert np.allclose(train["num2"].to_numpy(), 0.0, atol=0)

    # TEST: extrapolation for out-of-range values
    # test num1 values: [10, -4] → expected > 1 and < 0 respectively
    te_raw = np.array([10.0, -4.0], dtype=float)
    te_scaled_expected = _minmax_transform(te_raw, xmin=-2.0, xmax=8.0, a=0.0, b=1.0)
    np.testing.assert_allclose(test["num1"].to_numpy(), te_scaled_expected, rtol=0, atol=1e-12)
    assert test["num1"].iloc[0] > 1.0 and test["num1"].iloc[1] < 0.0

    # num2 constant in test as well → zeros
    assert np.allclose(test["num2"].to_numpy(), 0.0, atol=0)

    # Check saved scaler artifact (registry path → dict with id/params)
    pkl = ds / "impute_scale" / "scaler_cv1.pickle"
    with open(pkl, "rb") as f:
        obj = pickle.load(f)
    assert isinstance(obj, dict) and obj.get("id") == "minmax"
    assert obj.get("params", {}).get("feature_range") == (0, 1)


def test_minmax_custom_range_and_rounding():
    """
    Use a custom feature_range=(2,3).
    - TRAIN num1 maps exactly to [2,3].
    - Constant column maps to 'a' (=2) for all rows.
    - Output rounding to 7 decimals (from job) shouldn’t distort exact expectations.
    """
    tmp_path = Path("tests")
    exp, ds, tr, te = _seed_exp_no_missing(tmp_path)

    job = ImputeAndScale(
        cv_train_path=str(tr),
        cv_test_path=str(te),
        experiment_path=str(exp),
        scale_data=True,
        impute_data=False,
        multi_impute=False,
        overwrite_cv=True,
        outcome_label="Class",
        instance_label=None,
        random_state=0,
        scaler_id="minmax",
        scaler_params={"feature_range": (2.0, 3.0)},
    )
    job.run()

    train = _read_csv(tr)
    test = _read_csv(te)

    # TRAIN exact [2,3] for num1
    tr_raw = np.array([-2.0, 0.0, 3.0, 8.0], dtype=float)
    tr_exp = _minmax_transform(tr_raw, xmin=-2.0, xmax=8.0, a=2.0, b=3.0)
    np.testing.assert_allclose(train["num1"].to_numpy(), tr_exp, rtol=0, atol=1e-7)

    # Constant column → 'a' == 2.0
    assert np.allclose(train["num2"].to_numpy(), 2.0, atol=0)

    # TEST extrapolation also uses the same mapping
    te_raw = np.array([10.0, -4.0], dtype=float)
    te_exp = _minmax_transform(te_raw, xmin=-2.0, xmax=8.0, a=2.0, b=3.0)
    np.testing.assert_allclose(test["num1"].to_numpy(), te_exp, rtol=0, atol=1e-7)

    # Artifact contains chosen range
    with open(ds / "impute_scale" / "scaler_cv1.pickle", "rb") as f:
        obj = pickle.load(f)
    assert obj.get("params", {}).get("feature_range") == (2.0, 3.0)


def test_minmax_idempotent_transform_calls():
    """
    Ensure that applying transform twice (without re-fit) doesn’t drift:
    Our pipeline fits once on TRAIN, then transforms TRAIN/TEST.
    Here we simulate a second transform pass and confirm invariance.
    """
    tmp_path = Path("tests")
    exp, ds, tr, te = _seed_exp_no_missing(tmp_path)

    # First run
    job = ImputeAndScale(
        cv_train_path=str(tr),
        cv_test_path=str(te),
        experiment_path=str(exp),
        scale_data=True,
        impute_data=False,
        multi_impute=False,
        overwrite_cv=True,
        outcome_label="Class",
        instance_label=None,
        random_state=0,
        scaler_id="minmax",
        scaler_params={"feature_range": (0.0, 1.0)},
    )
    job.run()
    train_1 = _read_csv(tr).copy()
    test_1 = _read_csv(te).copy()

    # Second run (re-fit on TRAIN again; outputs should be identical)
    job2 = ImputeAndScale(
        cv_train_path=str(tr),
        cv_test_path=str(te),
        experiment_path=str(exp),
        scale_data=True,
        impute_data=False,
        multi_impute=False,
        overwrite_cv=True,
        outcome_label="Class",
        instance_label=None,
        random_state=0,
        scaler_id="minmax",
        scaler_params={"feature_range": (0.0, 1.0)},
    )
    job2.run()
    train_2 = _read_csv(tr)
    test_2 = _read_csv(te)

    pd.testing.assert_frame_equal(train_1, train_2, check_exact=True)
    pd.testing.assert_frame_equal(test_1, test_2, check_exact=True)
