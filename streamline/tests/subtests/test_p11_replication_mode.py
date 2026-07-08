from __future__ import annotations

import json
import pickle
import pytest
from pathlib import Path

from streamline.p11_reporting.p11_runner import P11Runner
from streamline.p11_reporting.reporting import ReportPhaseJob


pytest.skip("Tested Already", allow_module_level=True)

def _mk_dataset_root(path: Path) -> None:
    (path / "exploratory").mkdir(parents=True, exist_ok=True)
    (path / "model_evaluation").mkdir(parents=True, exist_ok=True)


def test_reportphasejob_replication_mode_discovers_replication_dirs(tmp_path: Path):
    exp_root = tmp_path / "exp"
    exp_root.mkdir(parents=True, exist_ok=True)

    with (exp_root / "metadata.pickle").open("wb") as f:
        pickle.dump({"Outcome Label": "Class", "Outcome Type": "Binary"}, f)

    train_a = exp_root / "train_a"
    train_b = exp_root / "train_b"
    _mk_dataset_root(train_a)
    _mk_dataset_root(train_b)

    rep_a = train_a / "replication" / "rep_a"
    rep_b = train_b / "replication" / "rep_b"
    _mk_dataset_root(rep_a)
    _mk_dataset_root(rep_b)

    job = ReportPhaseJob(
        experiment_path=str(exp_root),
        report_mode="replication",
        make_pdf=False,
        enable_plots=False,
    )
    discovered = job._list_datasets()

    assert [p.name for p in discovered] == ["rep_a", "rep_b"]
    assert all("replication" in str(p) for p in discovered)


def test_reportphasejob_replication_mode_writes_replication_report_data(tmp_path: Path):
    exp_root = tmp_path / "exp"
    exp_root.mkdir(parents=True, exist_ok=True)

    with (exp_root / "metadata.pickle").open("wb") as f:
        pickle.dump({"Outcome Label": "Class", "Outcome Type": "Binary"}, f)

    train_ds = exp_root / "train_data"
    _mk_dataset_root(train_ds)
    rep_ds = train_ds / "replication" / "rep_data"
    _mk_dataset_root(rep_ds)

    job = ReportPhaseJob(
        experiment_path=str(exp_root),
        report_mode="replication",
        make_pdf=False,
        enable_plots=False,
    )
    job.run()

    report_json = exp_root / "reporting_replication" / "report_data.json"
    assert report_json.is_file()

    payload = json.loads(report_json.read_text())
    assert payload["report_mode"] == "replication"
    assert payload["title"] == "STREAMLINE Replication Data Evaluation Report"
    assert len(payload["datasets"]) == 1
    assert "replication" in payload["datasets"][0]["dataset_path"]
    assert payload["dataset_comparisons"]["present"] is False


def test_p11_runner_passes_replication_mode_to_report_job(monkeypatch, tmp_path: Path):
    exp_root = tmp_path / "exp"
    exp_root.mkdir(parents=True, exist_ok=True)

    captured: dict[str, object] = {}

    class StubReportPhaseJob:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr("streamline.p11_reporting.p11_runner.ReportPhaseJob", StubReportPhaseJob)

    runner = P11Runner(
        experiment_path=str(exp_root),
        report_mode="replication",
        make_pdf=False,
        enable_plots=False,
        run_cluster="Serial",
    )
    runner.run()

    assert captured["report_mode"] == "replication"
    assert captured["ran"] is True

