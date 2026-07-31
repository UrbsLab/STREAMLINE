from pathlib import Path

from streamline.p6_modeling.p6_runner import P6Runner


def test_p6_skip_completed_models_runs_only_missing_model_cv_jobs(tmp_path: Path):
    output_path = tmp_path / "out"
    exp_root = output_path / "DemoExp"
    dataset_dir = exp_root / "DemoDataset"
    (dataset_dir / "CVDatasets").mkdir(parents=True)
    jobs_completed = exp_root / "jobsCompleted"
    jobs_completed.mkdir()
    (jobs_completed / "job_model_DemoDataset_0_NB.txt").write_text("complete")

    runner = P6Runner(
        output_path=str(output_path),
        experiment_name="DemoExp",
        outcome_type="Binary",
        models="NB,LR",
        n_splits=2,
    )

    jobs, _ = runner.collect_model_cv_jobs([str(dataset_dir)])
    assert len(jobs) == 4
    assert ("NB", 0) in [
        (getattr(ModelCls, "small_name"), cv_idx)
        for _, ModelCls, cv_idx in jobs
    ]

    skip_completed_runner = P6Runner(
        output_path=str(output_path),
        experiment_name="DemoExp",
        outcome_type="Binary",
        models="NB,LR",
        n_splits=2,
        skip_completed_models=True,
    )

    skip_completed_jobs, _ = skip_completed_runner.collect_model_cv_jobs([str(dataset_dir)])
    assert len(skip_completed_jobs) == 3
    assert ("NB", 0) not in [
        (getattr(ModelCls, "small_name"), cv_idx)
        for _, ModelCls, cv_idx in skip_completed_jobs
    ]


def test_p6_bash_skip_completed_models_marks_phase_complete_when_no_jobs(monkeypatch, tmp_path: Path):
    output_path = tmp_path / "out"
    exp_root = output_path / "DemoExp"
    dataset_dir = exp_root / "DemoDataset"
    (dataset_dir / "CVDatasets").mkdir(parents=True)
    jobs_completed = exp_root / "jobsCompleted"
    jobs_completed.mkdir()
    for model_id in ("NB", "LR"):
        for cv_idx in range(2):
            (jobs_completed / f"job_model_DemoDataset_{cv_idx}_{model_id}.txt").write_text("complete")

    submitted = []
    monkeypatch.setattr("streamline.p6_modeling.p6_runner.os.system", lambda cmd: submitted.append(cmd) or 0)

    runner = P6Runner(
        output_path=str(output_path),
        experiment_name="DemoExp",
        outcome_type="Binary",
        models="NB,LR",
        n_splits=2,
        run_cluster="BashSLURM",
        skip_completed_models=True,
    )
    runner.run()

    assert submitted == []
    assert not list((exp_root / "jobs").glob("P6_*.sh"))
    assert (jobs_completed / "job_modeling_DemoDataset.txt").read_text() == "complete"
