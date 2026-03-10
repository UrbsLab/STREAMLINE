from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from streamline.p10_replication import p10_cli
from streamline.p10_replication.p10_runner import P10Runner
from streamline.p10_replication.replication import (
    ReplicationJob,
    _normalize_outcome_type,
    _read_table,
)


class DummyEnsembleModel:
    """Simple pickle-safe binary classifier with deterministic probabilities."""

    def __init__(self, positive_prob: float = 0.7):
        self.positive_prob = float(positive_prob)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, x):
        n = len(x)
        p1 = np.clip(np.full(n, self.positive_prob), 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def _make_train_layout(
    base: Path,
    outcome_label: str = "Class",
    instance_label: str = "InstanceID",
    outcome_type: str = "Binary",
    cv_indices: tuple[int, ...] = (0,),
) -> tuple[Path, Path, Path]:
    """
    Build a minimal train dataset output tree required by ReplicationJob.

    Returns:
      (experiment_root, train_root, dataset_for_rep_file)
    """
    exp_root = base / "out" / "exp"
    train_name = "train_data"
    train_root = exp_root / train_name
    cv_dir = train_root / "CVDatasets"
    model_dir = train_root / "models" / "pickledModels"
    cv_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Training raw table used to recover column ordering.
    dataset_for_rep = base / f"{train_name}.csv"
    train_raw = pd.DataFrame(
        {
            outcome_label: [0, 1, 0, 1],
            instance_label: [10, 11, 12, 13],
            "f1": [0.1, 0.2, 0.3, 0.4],
            "f2": [1.0, 2.0, 3.0, 4.0],
        }
    )
    train_raw.to_csv(dataset_for_rep, index=False)

    for cv_idx in cv_indices:
        cv_df = train_raw.copy()
        cv_df.to_csv(cv_dir / f"{train_name}_CV_{cv_idx}_Train.csv", index=False)
        cv_df.to_csv(cv_dir / f"{train_name}_CV_{cv_idx}_Test.csv", index=False)

    # Minimal metadata + algorithm registry.
    metadata = {
        "Outcome Type": outcome_type,
        "Outcome Label": outcome_label,
        "Instance Label": instance_label,
        "Primary Metric": "balanced_accuracy",
    }
    with (exp_root / "metadata.pickle").open("wb") as f:
        pickle.dump(metadata, f)

    alg_info = {
        "AlgA": [True, "A"],
        "AlgB": [True, "B"],
    }
    with (exp_root / "algInfo.pickle").open("wb") as f:
        pickle.dump(alg_info, f)

    return exp_root, train_root, dataset_for_rep


def _make_replication_job(
    base: Path,
    outcome_type: str = "Binary",
    outcome_label: str = "Class",
    instance_label: str | None = "InstanceID",
    cv_partitions: int = 3,
) -> ReplicationJob:
    _, train_root, dataset_for_rep = _make_train_layout(
        base=base,
        outcome_type=outcome_type,
        outcome_label=outcome_label,
        instance_label=instance_label or "InstanceID",
        cv_indices=(0, 1, 2),
    )

    rep_file = base / "rep.csv"
    pd.DataFrame(
        {
            outcome_label: [0, 1, 1, 0],
            "InstanceID": [100, 101, 102, 103],
            "f1": [0.5, 0.6, np.nan, 0.8],
            "f2": [5.0, 6.0, 7.0, np.nan],
        }
    ).to_csv(rep_file, index=False)

    return ReplicationJob(
        dataset_filename=str(rep_file),
        dataset_for_rep=str(dataset_for_rep),
        full_path=str(train_root),
        outcome_label=outcome_label,
        outcome_type=outcome_type,
        instance_label=instance_label,
        match_label=None,
        cv_partitions=cv_partitions,
        show_plots=False,
    )


def test_normalize_outcome_type_aliases():
    assert _normalize_outcome_type("binary") == "Binary"
    assert _normalize_outcome_type("classification_multiclass") == "Multiclass"
    assert _normalize_outcome_type("regression") == "Continuous"
    assert _normalize_outcome_type("Binary") == "Binary"


def test_read_table_supports_csv_tsv_txt(tmp_path):
    csv_path = tmp_path / "a.csv"
    tsv_path = tmp_path / "b.tsv"
    txt_path = tmp_path / "c.txt"
    bad_path = tmp_path / "d.json"

    frame = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    frame.to_csv(csv_path, index=False)
    frame.to_csv(tsv_path, index=False, sep="\t")
    txt_path.write_text("x y\n1 3\n2 4\n", encoding="utf-8")
    bad_path.write_text("{}", encoding="utf-8")

    assert _read_table(str(csv_path)).shape == (2, 2)
    assert _read_table(str(tsv_path)).shape == (2, 2)
    assert _read_table(str(txt_path)).shape == (2, 2)
    with pytest.raises(ValueError):
        _read_table(str(bad_path))


def test_resolve_fold_map_uses_common_model_folds(tmp_path):
    _, train_root, dataset_for_rep = _make_train_layout(base=tmp_path, cv_indices=(0, 1, 2))
    rep_file = tmp_path / "rep.csv"
    pd.DataFrame(
        {
            "Class": [0, 1, 0],
            "InstanceID": [1, 2, 3],
            "f1": [0.1, 0.2, 0.3],
            "f2": [1.0, 2.0, 3.0],
        }
    ).to_csv(rep_file, index=False)

    # A has models for folds 0 and 1; B has models for folds 1 and 2.
    model_dir = train_root / "models" / "pickledModels"
    for name in ("A_0.pickle", "A_1.pickle", "B_1.pickle", "B_2.pickle"):
        with (model_dir / name).open("wb") as f:
            pickle.dump({"ok": True}, f)

    job = ReplicationJob(
        dataset_filename=str(rep_file),
        dataset_for_rep=str(dataset_for_rep),
        full_path=str(train_root),
        outcome_label="Class",
        outcome_type="Binary",
        instance_label="InstanceID",
        match_label=None,
        cv_partitions=5,
    )

    # Only fold 1 is common across all active algorithms.
    assert job._resolve_fold_map() == [(0, 1)]


def test_auto_correct_labels_and_metric_for_regression(tmp_path):
    _, train_root, dataset_for_rep = _make_train_layout(
        base=tmp_path,
        outcome_label="Target",
        instance_label="RID",
        outcome_type="Continuous",
        cv_indices=(0,),
    )
    rep_file = tmp_path / "rep.csv"
    pd.DataFrame({"Target": [1.1, 2.2], "RID": [9, 10], "f1": [0.1, 0.2], "f2": [1, 2]}).to_csv(
        rep_file, index=False
    )

    job = ReplicationJob(
        dataset_filename=str(rep_file),
        dataset_for_rep=str(dataset_for_rep),
        full_path=str(train_root),
        outcome_label="WrongLabel",
        outcome_type="Continuous",
        instance_label="WrongLabel",
        match_label="missing_match",
        scoring_metric="balanced_accuracy",
    )
    job._auto_correct_labels_from_training_cv()

    assert job.outcome_label == "Target"
    assert job.instance_label is None
    assert job.match_label is None
    assert job.scoring_metric == "explained_variance"


def test_write_base_outputs_regression_creates_residual_pickle(tmp_path):
    job = _make_replication_job(tmp_path, outcome_type="Continuous", outcome_label="Target", instance_label=None)
    job._prepare_dirs()

    y_true = np.array([1.0, 2.0, 3.0], dtype=float)
    y_pred = np.array([1.5, 1.8, 3.2], dtype=float)
    residual = y_true - y_pred

    job._write_base_outputs(
        rep_cv_idx=0,
        small_name="A",
        metrics_dict={"Explained Variance": 0.9},
        curves_dict=None,
        fi_list=[0.2, 0.3],
        residual_test=residual,
        y_pred=y_pred,
        y_true=y_true,
    )

    metrics_path = job.model_metrics_dir / "A_CV_0.json"
    residual_path = job.model_pickled_metrics_dir / "A_CV_0_residuals.pickle"

    assert metrics_path.exists()
    assert residual_path.exists()

    with metrics_path.open("r") as f:
        payload = json.load(f)
    assert "metrics" in payload
    assert "feature_importance" in payload
    assert payload["feature_importance"] == [0.2, 0.3]

    with residual_path.open("rb") as f:
        residual_payload = pickle.load(f)
    assert len(residual_payload) == 6
    assert np.allclose(residual_payload[1], residual)
    assert np.allclose(residual_payload[3], y_pred)
    assert np.allclose(residual_payload[5], y_true)


def test_evaluate_ensembles_writes_all_ensemble_algorithms(tmp_path):
    job = _make_replication_job(tmp_path, outcome_type="Binary")
    job._prepare_dirs()

    rep_test = pd.DataFrame(
        {
            "Class": [0, 1, 0, 1],
            "InstanceID": [1, 2, 3, 4],
            "f1": [0.2, 0.3, 0.4, 0.5],
            "f2": [1.0, 1.1, 1.2, 1.3],
        }
    )
    rep_test.to_csv(job.cv_dir / f"{job.apply_name}_CV_0_Test.csv", index=False)

    src_pickled = job.train_root / "ensemble_evaluation" / "pickled_ensembles"
    src_pickled.mkdir(parents=True, exist_ok=True)
    with (src_pickled / "HEV_0.pickle").open("wb") as f:
        pickle.dump(DummyEnsembleModel(0.8), f)
    with (src_pickled / "SEV_0.pickle").open("wb") as f:
        pickle.dump(DummyEnsembleModel(0.6), f)

    job._evaluate_ensembles(rep_test, fold_map=[(0, 0)])

    metric_files = sorted(job.ensemble_metrics_dir.glob("*_CV_0.json"))
    roc_files = sorted(job.ensemble_curves_dir.glob("*_CV_0_roc.json"))
    prc_files = sorted(job.ensemble_curves_dir.glob("*_CV_0_prc.json"))

    assert [f.name for f in metric_files] == ["HEV_CV_0.json", "SEV_CV_0.json"]
    assert [f.name for f in roc_files] == ["HEV_CV_0_roc.json", "SEV_CV_0_roc.json"]
    assert [f.name for f in prc_files] == ["HEV_CV_0_prc.json", "SEV_CV_0_prc.json"]


def test_p10_runner_serial_runs_for_supported_extensions(tmp_path, monkeypatch):
    rep_dir = tmp_path / "rep_data"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "a.csv").write_text("Class,f1\n0,1\n", encoding="utf-8")
    (rep_dir / "b.tsv").write_text("Class\tf1\n1\t2\n", encoding="utf-8")
    (rep_dir / "ignore.md").write_text("x", encoding="utf-8")

    output_path = tmp_path / "out"
    exp_root = output_path / "exp"
    train_file = tmp_path / "train_data.csv"
    train_file.write_text("Class,InstanceID,f1\n0,1,2\n", encoding="utf-8")

    (exp_root / "train_data").mkdir(parents=True, exist_ok=True)
    with (exp_root / "metadata.pickle").open("wb") as f:
        pickle.dump(
            {
                "Outcome Type": "Binary",
                "Outcome Label": "Class",
                "Instance Label": "InstanceID",
                "CV Partitions": 3,
                "Use Data Scaling": True,
                "Use Data Imputation": True,
                "Use Multivariate Imputation": False,
            },
            f,
        )

    calls: list[tuple[str, str]] = []

    class StubReplicationJob:
        def __init__(self, **kwargs):
            self.dataset_filename = kwargs["dataset_filename"]
            calls.append(("init", Path(self.dataset_filename).name))

        def run(self):
            calls.append(("run", Path(self.dataset_filename).name))

    monkeypatch.setattr("streamline.p10_replication.p10_runner.ReplicationJob", StubReplicationJob)

    runner = P10Runner(
        rep_data_path=str(rep_dir),
        dataset_for_rep=str(train_file),
        output_path=str(output_path),
        experiment_name="exp",
        run_cluster="Serial",
    )
    runner.run()

    assert ("run", "a.csv") in calls
    assert ("run", "b.tsv") in calls
    assert all(name != "ignore.md" for _, name in calls)


def test_p10_runner_raises_when_no_supported_datasets(tmp_path):
    rep_dir = tmp_path / "rep_data"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "notes.md").write_text("ignore", encoding="utf-8")

    output_path = tmp_path / "out"
    exp_root = output_path / "exp"
    train_file = tmp_path / "train_data.csv"
    train_file.write_text("Class,InstanceID,f1\n0,1,2\n", encoding="utf-8")

    (exp_root / "train_data").mkdir(parents=True, exist_ok=True)
    with (exp_root / "metadata.pickle").open("wb") as f:
        pickle.dump(
            {"Outcome Type": "Binary", "Outcome Label": "Class", "Instance Label": "InstanceID"},
            f,
        )

    runner = P10Runner(
        rep_data_path=str(rep_dir),
        dataset_for_rep=str(train_file),
        output_path=str(output_path),
        experiment_name="exp",
        run_cluster="Serial",
    )

    with pytest.raises(Exception, match="There must be at least one"):
        runner.run()


def test_p10_cli_parses_args_and_invokes_runner(monkeypatch, tmp_path):
    rep_dir = tmp_path / "rep"
    rep_dir.mkdir(parents=True, exist_ok=True)
    train_file = tmp_path / "train.csv"
    train_file.write_text("Class,InstanceID,f1\n0,1,2\n", encoding="utf-8")

    captured: dict[str, object] = {}

    class StubRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(p10_cli, "P10Runner", StubRunner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p10_cli.py",
            "--rep_data_path",
            str(rep_dir),
            "--dataset_for_rep",
            str(train_file),
            "--output_path",
            str(tmp_path / "out"),
            "--experiment_name",
            "exp",
            "--exclude_plots",
            "plot_ROC,plot_PRC",
            "--show_plots",
            "1",
        ],
    )

    p10_cli.main()

    assert captured["exclude_plots"] == ["plot_ROC", "plot_PRC"]
    assert captured["show_plots"] is True
    assert captured["ran"] is True
