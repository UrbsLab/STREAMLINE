from pathlib import Path

from streamline.pipeline.pipeline_runner import PipelineRunner
from streamline.p6_modeling.utils.loader import load_default_model_classes
from streamline.utils.run_commands import load_phase_run_command, snapshot_effective_args


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_pipeline_controls_use_phase_aliases():
    config = {
        "run": {
            "phases": ["p1", "p2", "p3", "p4"],
        }
    }

    runner = PipelineRunner(
        config=config,
        dry_run=True,
        start_at="p2",
        stop_after="p3",
        skip="p2",
    )

    assert runner.resolve_phase_order() == ["p3_feature_learning"]


def test_do_till_report_matches_old_cfg_style():
    config = {
        "run": {
            "phases": ["p1", "p10", "p11"],
        },
        "phase_controls": {
            "do_till_report": True,
        },
        "phases": {
            "p10": {
                "rep_data_path": "data/UCIRepBinaryClassification",
                "dataset_for_rep": "data/UCIBinaryClassification/hcc_survival.csv",
            }
        },
    }

    runner = PipelineRunner(config=config, dry_run=True)

    assert runner.phase_is_enabled("p1_data_process")
    assert not runner.phase_is_enabled("p10_replication")
    assert runner.phase_is_enabled("p11_reporting")


def test_elcs_is_not_a_default_model():
    for model_type in ("Binary", "Multiclass"):
        default_ids = {
            getattr(model_class, "small_name", "").lower()
            for model_class in load_default_model_classes(model_type)
        }

        assert "elcs" not in default_ids


def test_example_configs_dry_run_expected_phases():
    cases = {
        "uci_binary_hcc.cfg": [
            "p1_data_process",
            "p2_impute_scale",
            "p3_feature_learning",
            "p4_feature_importance",
            "p5_feature_selection",
            "p6_modeling",
            "p7_ensembles",
            "p8_summary_statistics",
            "p9_compare_datasets",
            "p10_replication",
            "p11_reporting",
        ],
        "uci_multiclass_student.cfg": [
            "p1_data_process",
            "p2_impute_scale",
            "p3_feature_learning",
            "p4_feature_importance",
            "p5_feature_selection",
            "p6_modeling",
            "p7_ensembles",
            "p8_summary_statistics",
            "p9_compare_datasets",
            "p10_replication",
            "p11_reporting",
        ],
        "uci_regression_auto_mpg.cfg": [
            "p1_data_process",
            "p2_impute_scale",
            "p3_feature_learning",
            "p4_feature_importance",
            "p5_feature_selection",
            "p6_modeling",
            "p8_summary_statistics",
            "p9_compare_datasets",
            "p10_replication",
            "p11_reporting",
        ],
    }

    for config_name, expected_phases in cases.items():
        runner = PipelineRunner(
            config_path=PROJECT_ROOT / "run_configs" / config_name,
            dry_run=True,
        )

        assert runner.run() == expected_phases


def test_config_runner_records_phase_args_in_run_command_pickle(monkeypatch, tmp_path):
    class FakeRunner:
        def __init__(self, output_path, experiment_name, foo="default", optional=None, run_cluster="Serial"):
            self.output_path = output_path
            self.experiment_name = experiment_name
            self.foo = foo
            self.optional = "resolved-default" if optional is None else optional
            self.run_cluster = run_cluster

        def run(self):
            Path(self.output_path, self.experiment_name).mkdir(parents=True, exist_ok=True)

    monkeypatch.setitem(
        __import__("streamline.pipeline.pipeline_runner", fromlist=["PHASE_RUNNERS"]).PHASE_RUNNERS,
        "p1_data_process",
        FakeRunner,
    )

    output_path = tmp_path / "out"
    runner = PipelineRunner(
        config={
            "run": {
                "output_path": str(output_path),
                "experiment_name": "DemoExp",
                "phases": ["p1"],
            },
            "phases": {
                "p1": {
                    "foo": "bar",
                }
            },
        }
    )

    assert runner.run() == ["p1_data_process"]

    record = load_phase_run_command(output_path / "DemoExp", "p1_data_process")
    assert record["args"]["foo"] == "bar"
    assert record["args"]["optional"] == "resolved-default"
    assert record["args"]["run_cluster"] == "Serial"
    assert record["args"]["experiment_name"] == "DemoExp"


def test_effective_args_snapshot_uses_runner_kw_dict():
    class FakeRunner:
        def __init__(self):
            self.kw = {
                "outcome_type": "Binary",
                "show_plots": False,
            }
            self.queue = "defq"
            self.runtime_only = "not saved"

    effective = snapshot_effective_args(
        {
            "outcome_type": None,
            "show_plots": True,
            "queue": "oldq",
        },
        FakeRunner(),
    )

    assert effective == {
        "outcome_type": "Binary",
        "show_plots": False,
        "queue": "defq",
    }
