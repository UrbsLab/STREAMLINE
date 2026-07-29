import csv
import os
import pickle

import pandas as pd

from streamline.p5_feature_selection.p5_runner import P5Runner
from streamline.p6_modeling.models.regression.random_forest import RFRegressor


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def test_p5_uses_metadata_outcome_label_when_cli_omits_label(tmp_path):
    out = tmp_path / "out"
    exp = out / "Exp"
    dataset = exp / "D1"
    cv_dir = dataset / "CVDatasets"
    fi_dir = dataset / "feature_importance" / "mutualinformation"
    cv_dir.mkdir(parents=True)
    fi_dir.mkdir(parents=True)

    with open(exp / "metadata.pickle", "wb") as f:
        pickle.dump({"Outcome Label": "MPG", "Instance Label": "InstanceID"}, f)

    write_csv(
        cv_dir / "D1_CV_0_Train.csv",
        [
            ["MPG", "f1", "f2", "noise0"],
            [18.0, 1.0, 2.0, 9.0],
            [22.0, 2.0, 3.0, 8.0],
        ],
    )
    write_csv(
        cv_dir / "D1_CV_0_Test.csv",
        [
            ["MPG", "f1", "f2", "noise0"],
            [20.0, 1.5, 2.5, 7.0],
        ],
    )
    write_csv(
        fi_dir / "mutualinformation_scores_cv_0.csv",
        [
            ["feature", "score"],
            ["f1", 0.9],
            ["f2", 0.4],
            ["noise0", 0.0],
        ],
    )

    P5Runner(
        output_path=str(out),
        experiment_name="Exp",
        algorithms="auto",
        n_splits=1,
        outcome_label=None,
        instance_label=None,
        max_features_to_keep=5,
        filter_poor_features=True,
        overwrite_cv=False,
        export_scores=False,
        show_plots=False,
    ).run()

    selected = pd.read_csv(cv_dir / "D1_CV_0_Train.csv")
    assert "MPG" in selected.columns
    assert "Class" not in selected.columns
    assert "noise0" not in selected.columns


def test_regression_random_forest_search_space_uses_valid_sklearn_max_features():
    model = RFRegressor()

    assert "auto" not in model.param_grid["max_features"]
    assert "sqrt" in model.param_grid["max_features"]
