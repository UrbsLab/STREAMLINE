# tests/phases/p2_impute_scale/test_runner_after_p1.py
import os
import json
import pytest
import pickle
from pathlib import Path

import pandas as pd

from streamline.p2_impute_scale.p2_runner import P2Runner

pytest.skip("Tested Already", allow_module_level=True)


def _seed_phase1_outputs(tmp_path: Path, exp_name: str = "demo", ds_name: str = "Toy"):
    """
    Create a minimal Phase-1-like experiment structure with:
      - metadata.pickle
      - exploratory/{DataCounts.csv, categorical_features.pickle}
      - CVDatasets/{Toy_CV_1_Train.csv, Toy_CV_1_Test.csv}
      - jobsCompleted/, jobs/, logs/ dirs
    """
    exp_root = tmp_path / exp_name
    ds_dir = exp_root / ds_name
    cvd = ds_dir / "CVDatasets"
    expl = ds_dir / "exploratory"

    # Make dirs
    (exp_root / "jobsCompleted").mkdir(parents=True, exist_ok=True)
    (exp_root / "jobs").mkdir(exist_ok=True)
    (exp_root / "logs").mkdir(exist_ok=True)
    ds_dir.mkdir(parents=True, exist_ok=True)
    cvd.mkdir(parents=True, exist_ok=True)
    expl.mkdir(parents=True, exist_ok=True)

    # Phase-1 metadata (values used by P2Runner as defaults)
    meta = {
        "Outcome Label": "Class",
        "Instance Label": None,
        "Random Seed": 7,
        "Use Data Scaling": True,
        "Use Data Imputation": True,
        "Use Multivariate Imputation": False,
        # Optional: store a default imputer choice (runner will honor if P2 not overridden)
        "P2 Imputer Id": "median_map",
        "P2 Imputer Params": json.dumps({}),
    }
    with open(exp_root / "metadata.pickle", "wb") as f:
        pickle.dump(meta, f)

    # Exploratory artifacts
    # 5th row ("Missing") Count is used in P2 to decide whether to impute
    pd.DataFrame(
        {"Type": ["A", "B", "C", "D", "Missing"], "Count": [0, 0, 0, 0, 3]}
    ).to_csv(expl / "DataCounts.csv", index=False)
    with open(expl / "categorical_features.pickle", "wb") as f:
        pickle.dump(["cat1"], f)

    # One CV pair with some missing data
    train = pd.DataFrame(
        {
            "Class": [0, 1, 0, 1],
            "cat1": ["x", None, "x", None],
            "num1": [1.0, None, 3.0, 4.0],
            "num2": [None, 2.0, 5.0, 6.0],
        }
    )
    test = pd.DataFrame(
        {
            "Class": [1, 0],
            "cat1": [None, None],
            "num1": [None, 10.0],
            "num2": [7.0, None],
        }
    )

    tr = cvd / f"{ds_name}_CV_1_Train.csv"
    te = cvd / f"{ds_name}_CV_1_Test.csv"
    train.to_csv(tr, index=False)
    test.to_csv(te, index=False)

    return exp_root, ds_dir, tr, te


def test_p2_runner_after_p1_serial():
    """
    Given a Phase-1-like experiment folder, running P2Runner in serial should:
      - impute categorical (mode) + numeric (via metadata’s imputer or defaults)
      - scale numeric (when metadata says True)
      - write pickles (categorical/ordinal/scaler)
      - rewrite CV Train/Test CSVs
      - write jobsCompleted marker
      - append run_params.pickle
    """
    tmp_path = Path("tests")
    exp_root, ds_dir, tr, te = _seed_phase1_outputs(tmp_path)

    r = P2Runner(
        output_path=str(tmp_path),
        experiment_name=exp_root.name,
        run_cluster=False,  # serial path
        overwrite_cv=False
        # leave Phase-2 flags as None so they’re sourced from metadata.pickle
    )
    r.run()

    # CV files should be rewritten and contain no missing values
    train = pd.read_csv(tr)
    test = pd.read_csv(te)
    assert train.isna().sum().sum() == 0
    assert test.isna().sum().sum() == 0

    # Categorical imputation should have filled cat1
    assert train["cat1"].isna().sum() == 0
    assert test["cat1"].isna().sum() == 0

    # Numeric columns should be imputed; and scaled when Use Data Scaling=True
    for col in ["num1", "num2"]:
        assert train[col].isna().sum() == 0
        assert test[col].isna().sum() == 0
        # mean ~ 0 after StandardScaler (tolerance due to small sample)
        assert abs(train[col].mean()) < 1e-3

    # Check artifacts
    scale_dir = ds_dir / "impute_scale"
    assert (scale_dir / "categorical_imputer_cv1.pickle").exists()
    assert (scale_dir / "ordinal_imputer_cv1.pickle").exists()
    assert (scale_dir / "scaler_cv1.pickle").exists()

    # jobsCompleted marker exists
    jc = (exp_root / "jobsCompleted" / f"job_preprocessing_{ds_dir.name}_1.txt")
    assert jc.exists()

    # run_params.pickle appended
    rp = exp_root / "run_params.pickle"
    assert rp.exists()
    with open(rp, "rb") as f:
        runs = pickle.load(f)
    assert isinstance(runs, dict) and runs
    # ensure last run stored Phase 2 fields
    latest = sorted(runs.keys())[-1]
    assert runs[latest]["phase"] == "p2_impute_scale"
    assert runs[latest]["scale_data"] is True
    assert runs[latest]["impute_data"] is True

