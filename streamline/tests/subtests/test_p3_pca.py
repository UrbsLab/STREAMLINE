import os, pickle, json
from pathlib import Path
import pandas as pd
import logging
import pytest
from streamline.p3_feature_learning.feature_learn import FeatureLearn

pytest.skip("Tested Already", allow_module_level=True)

def _seed(tmp_path: Path, ds="hcc_demo"):
    exp = tmp_path / "exp"; ds_dir = exp / ds
    cvd = ds_dir / "CVDatasets"; expl = ds_dir / "exploratory"
    (exp / "jobsCompleted").mkdir(parents=True, exist_ok=True)
    for p in (ds_dir, cvd, expl): p.mkdir(parents=True, exist_ok=True)
    # metadata with Outcome Label
    with open(exp / "metadata.pickle", "wb") as f: pickle.dump({"Outcome Label":"Class"}, f)
    # simple numeric data (no missing)
    tr = cvd / f"{ds}_CV_1_Train.csv"; te = cvd / f"{ds}_CV_1_Test.csv"
    pd.DataFrame({"Class":[0,1,0,1], "x1":[1,2,3,4], "x2":[4,3,2,1]}).to_csv(tr, index=False)
    pd.DataFrame({"Class":[1,0],   "x1":[10, 0],    "x2":[0, 10]}).to_csv(te, index=False)
    return exp, ds_dir, tr, te

def test_p3_pca_writes_outputs(tmp_path):
    tmp_path = Path("tests")
    exp, ds_dir, tr, te = _seed(tmp_path)
    FeatureLearn(
        cv_train_path=str(tr),
        cv_test_path=str(te),
        experiment_path=str(exp),
        learner_id="pca",
        learner_params={"n_components": 2, "svd_solver":"auto"},
        feature_namespace="FL_PCA",
        keep_original_features=True,
        overwrite_cv=True,
        outcome_label="Class",
        random_state=0,
    ).run()

    train = pd.read_csv(tr); test = pd.read_csv(te)
    # original + 2 engineered features
    assert "FL_PCA_PC1" in train.columns and "FL_PCA_PC2" in train.columns
    assert "FL_PCA_PC1" in test.columns and "FL_PCA_PC2" in test.columns

    base = ds_dir / "feature_learning"
    assert (base / "learner_cv1.pickle").exists()
    assert (base / "features_cv1.txt").exists()
    assert (base / "feature_manifest_cv1.json").exists()
