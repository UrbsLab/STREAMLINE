from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


UCI_DATASETS = [
    {
        "name": "heart_disease_cleveland",
        "csv": REPO_ROOT / "data" / "UCIBinaryClassification" / "heart_disease_cleveland.csv",
        "copy_csv": REPO_ROOT / "data" / "UCIBinaryClassification" / "heart_disease_cleveland_copy.csv",
        "rep_csv": REPO_ROOT / "data" / "UCIRepBinaryClassification" / "heart_disease_cleveland_rep.csv",
        "categorical": REPO_ROOT / "data" / "UCIFeatureTypes" / "heart_disease_categorical_features.csv",
        "quantitative": REPO_ROOT / "data" / "UCIFeatureTypes" / "heart_disease_quantitative_features.csv",
        "outcome": "Class",
        "expected_classes": {"0", "1"},
    },
    {
        "name": "dermatology",
        "csv": REPO_ROOT / "data" / "UCIMulticlassClassification" / "dermatology.csv",
        "copy_csv": REPO_ROOT / "data" / "UCIMulticlassClassification" / "dermatology_copy.csv",
        "rep_csv": REPO_ROOT / "data" / "UCIRepMulticlassClassification" / "dermatology_rep.csv",
        "categorical": REPO_ROOT / "data" / "UCIFeatureTypes" / "dermatology_categorical_features.csv",
        "quantitative": REPO_ROOT / "data" / "UCIFeatureTypes" / "dermatology_quantitative_features.csv",
        "outcome": "Class",
        "expected_classes": {"0", "1", "2", "3", "4", "5"},
    },
    {
        "name": "auto_mpg",
        "csv": REPO_ROOT / "data" / "UCIRegression" / "auto_mpg.csv",
        "copy_csv": REPO_ROOT / "data" / "UCIRegression" / "auto_mpg_copy.csv",
        "rep_csv": REPO_ROOT / "data" / "UCIRepRegression" / "auto_mpg_rep.csv",
        "categorical": REPO_ROOT / "data" / "UCIFeatureTypes" / "auto_mpg_categorical_features.csv",
        "quantitative": REPO_ROOT / "data" / "UCIFeatureTypes" / "auto_mpg_quantitative_features.csv",
        "outcome": "MPG",
        "expected_classes": None,
    },
]


def read_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_features(path: Path) -> list[str]:
    with path.open(newline="") as f:
        return [row["Feature"] for row in csv.DictReader(f)]


def test_uci_demo_datasets_have_expected_shapes_and_missing_values():
    for spec in UCI_DATASETS:
        rows = read_rows(spec["csv"])
        assert rows, f"{spec['name']} should have rows"
        assert "InstanceID" in rows[0], f"{spec['name']} should include InstanceID"
        assert spec["outcome"] in rows[0], f"{spec['name']} should include outcome"
        assert sum(value == "NA" for row in rows for value in row.values()) > 0, f"{spec['name']} should contain missing values"

        categorical = read_features(spec["categorical"])
        quantitative = read_features(spec["quantitative"])
        headers = set(rows[0])
        assert categorical, f"{spec['name']} should declare categorical features"
        assert quantitative, f"{spec['name']} should declare quantitative features"
        assert not (set(categorical) & set(quantitative)), f"{spec['name']} feature types should not overlap"
        assert set(categorical).issubset(headers), f"{spec['name']} categorical features should exist in CSV"
        assert set(quantitative).issubset(headers), f"{spec['name']} quantitative features should exist in CSV"
        assert "InstanceID" not in categorical + quantitative
        assert spec["outcome"] not in categorical + quantitative

        if spec["expected_classes"] is not None:
            observed = {row[spec["outcome"]] for row in rows}
            assert observed == spec["expected_classes"], f"{spec['name']} class labels changed"
        else:
            assert all(row[spec["outcome"]] != "NA" for row in rows), f"{spec['name']} target should be complete"


def test_uci_companion_and_replication_datasets_match_training_schema():
    for spec in UCI_DATASETS:
        train_rows = read_rows(spec["csv"])
        copy_rows = read_rows(spec["copy_csv"])
        rep_rows = read_rows(spec["rep_csv"])
        assert copy_rows, f"{spec['name']} companion copy data should have rows"
        assert rep_rows, f"{spec['name']} replication data should have rows"
        assert list(train_rows[0]) == list(copy_rows[0]), f"{spec['name']} companion copy schema should match training schema"
        assert list(train_rows[0]) == list(rep_rows[0]), f"{spec['name']} replication schema should match training schema"
