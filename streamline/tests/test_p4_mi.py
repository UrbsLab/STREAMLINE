import os, pickle
from pathlib import Path
import pandas as pd

from streamline.p4_feature_importance.importance import FeatureImportance

def _seed(tmp: Path, ds="Toy"):
    exp = tmp / "exp"; ds_dir = exp / ds
    cvd = ds_dir / "CVDatasets"
    (exp / "jobsCompleted").mkdir(parents=True, exist_ok=True)
    for p in (ds_dir, cvd): p.mkdir(parents=True, exist_ok=True)
    tr = cvd / f"{ds}_CV_1_Train.csv"; te = cvd / f"{ds}_CV_1_Test.csv"
    # Simple signal in x1; x2 is noise
    pd.DataFrame({"Class":[0,0,1,1,1,0,1,0],
                  "x1":[0,0,1,1,1,0,1,0],
                  "x2":[1,2,3,4,5,6,7,8]}).to_csv(tr, index=False)
    pd.DataFrame({"Class":[1,0], "x1":[1,0], "x2":[9,10]}).to_csv(te, index=False)
    with open(exp/"metadata.pickle", "wb") as f: pickle.dump({"Outcome Label":"Class","Outcome Type":"Binary"}, f)
    return exp, ds_dir, tr, te

def test_p4_mutual_info():
    tmp_path = Path("tests")
    exp, ds_dir, tr, te = _seed(tmp_path)
    FeatureImportance(
        cv_train_path=str(tr),
        cv_test_path=str(te),
        experiment_path=str(exp),
        model_id="mutualinformation",
        model_params={"outcome_type":"Binary", "n_neighbors":3},
        top_k=1,
        threshold=None,
        keep_original_features=False,
        overwrite_cv=True,
        outcome_label="Class",
        outcome_type="Binary",
        random_state=0,
    ).run()

    train = pd.read_csv(tr)
    test = pd.read_csv(te)
    # only one selected feature; should favor x1
    assert list(train.columns) == ["Class","x1"]
    assert list(test.columns)  == ["Class","x1"]

    base = ds_dir / "feature_importance"
    assert (base / "scores_cv1.csv").exists()
    assert (base / "importance_cv1.pickle").exists()
    assert (base / "selected_features_cv1.txt").exists()
