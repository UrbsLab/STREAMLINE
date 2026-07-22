import shlex
from pathlib import Path

from streamline.pipeline.pipeline_runner import PHASE_RUNNERS, PipelineRunner
from streamline.p1_data_process.p1_runner import P1Runner
from streamline.utils.runners import quote_command_parts


def test_bash_command_parts_quote_empty_and_spaced_values():
    command = quote_command_parts(
        [
            "python",
            "/tmp/run script.py",
            "--dataset_name",
            "",
            "--output_path",
            "/tmp/output root",
        ]
    )

    assert "'/tmp/run script.py'" in command
    assert "--dataset_name ''" in command
    assert "'/tmp/output root'" in command


def test_p1_bash_command_quotes_empty_optional_values(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_dir = tmp_path / "out root"

    runner = P1Runner(
        data_path=str(data_dir),
        output_path=str(output_dir),
        experiment_name="DemoExp",
        outcome_label="Class",
        run_cluster="BashSLURM",
    )

    command = runner._bash_submit_command(str(tmp_path / "student data.csv"))
    parts = shlex.split(command)

    assert "--dataset_path" in command
    assert parts[parts.index("--dataset_path") + 1].endswith("student data.csv")
    assert parts[parts.index("--dataset_name") + 1] == ""
    assert parts[parts.index("--instance_label") + 1] == ""
    assert parts[parts.index("--match_label") + 1] == ""
    assert "--dataset_name ''" in command
    assert "--instance_label ''" in command
    assert "--match_label ''" in command


def test_pipeline_waits_for_bash_phase_completion_marker(monkeypatch, tmp_path):
    class FakeBashRunner:
        def __init__(self, output_path, experiment_name, run_cluster="Serial"):
            self.output_path = output_path
            self.experiment_name = experiment_name
            self.run_cluster = run_cluster

        def run(self):
            exp_root = Path(self.output_path) / self.experiment_name
            jobs = exp_root / "jobs"
            jobs_completed = exp_root / "jobsCompleted"
            jobs.mkdir(parents=True, exist_ok=True)
            jobs_completed.mkdir(parents=True, exist_ok=True)
            (jobs / "P1_demo_run.sh").write_text("#!/bin/bash\n")
            (jobs_completed / "job_data_process_Demo.txt").write_text("complete")

    monkeypatch.setitem(PHASE_RUNNERS, "p1_data_process", FakeBashRunner)

    runner = PipelineRunner(
        config={
            "run": {
                "output_path": str(tmp_path / "out"),
                "experiment_name": "DemoExp",
                "phases": ["p1"],
                "run_cluster": "BashSLURM",
                "cluster_phase_timeout": 1,
                "cluster_phase_poll_interval": 1,
            }
        }
    )

    assert runner.run() == ["p1_data_process"]


def test_pipeline_cluster_wait_setting_accepts_false_string(tmp_path):
    runner = PipelineRunner(
        config={
            "run": {
                "output_path": str(tmp_path / "out"),
                "experiment_name": "DemoExp",
                "phases": ["p1"],
                "run_cluster": "BashSLURM",
                "wait_for_cluster_completion": "False",
            }
        },
        dry_run=True,
    )

    assert not runner.should_wait_for_cluster_phase({"run_cluster": "BashSLURM"}, {})
