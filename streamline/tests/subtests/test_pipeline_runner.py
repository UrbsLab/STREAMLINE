from pathlib import Path

from streamline.pipeline.pipeline_runner import PipelineRunner, load_config
import streamline.p6_modeling.p6_runner as p6_runner_module
from streamline.p6_modeling.p6_runner import P6Runner
from streamline.p6_modeling.utils.loader import load_default_model_classes
from streamline.utils.run_commands import load_phase_run_command, snapshot_effective_args


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def csv_values(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [item.strip() for item in str(value).split(",") if item.strip()]


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


def test_example_configs_include_explicit_non_json_phase_parameters():
    required_run_keys = {
        "output_path",
        "experiment_name",
        "outcome_label",
        "outcome_type",
        "instance_label",
        "n_splits",
        "run_cluster",
        "queue",
        "reserved_memory",
        "random_state",
    }
    required_phase_keys = {
        "p1": {
            "data_path",
            "exclude_eda_output",
            "match_label",
            "ignore_features",
            "categorical_features",
            "quantitative_features",
            "top_features",
            "categorical_cutoff",
            "sig_cutoff",
            "featureeng_missingness",
            "cleaning_missingness",
            "correlation_removal_threshold",
            "partition_method",
            "show_plots",
            "one_hot_encoding",
            "cv_provided",
            "cv_input_root",
            "enable_plots",
            "plot_missingness",
            "plot_class_counts",
            "plot_correlation",
            "correlation_plot_max_features",
            "plot_univariate",
            "univariate_top_k",
            "plot_anomalies",
            "force",
        },
        "p2": {
            "scale_data",
            "impute_data",
            "multi_impute",
            "overwrite_cv",
            "imputer_id",
            "scaler_id",
            "smote",
            "smote_method",
            "smote_sampling_strategy",
            "smote_k_neighbors",
        },
        "p3": {
            "learner_id",
            "feature_namespace",
            "keep_original_features",
            "overwrite_cv",
        },
        "p4": {
            "models",
            "top_k",
            "threshold",
            "keep_original_features",
            "overwrite_cv",
            "instance_subset",
        },
        "p5": {
            "algorithms",
            "n_splits",
            "max_features_to_keep",
            "filter_poor_features",
            "overwrite_cv",
            "selector_id",
            "export_scores",
            "top_features",
            "show_plots",
            "strict_discovery",
        },
        "p6": {
            "outcome_type",
            "model_type",
            "models",
            "calibrate",
            "calibrate_method",
            "calibrate_cv",
            "scoring_metric",
            "metric_direction",
            "n_trials",
            "timeout",
            "training_subsample",
            "uniform_fi",
            "save_plot",
            "bypass_one_hot_for_native_models",
            "native_categorical_models",
        },
        "p7": {
            "ensembles",
            "base_models",
            "meta_train_source",
            "calibrate",
            "calibrate_method",
            "calibrate_cv",
        },
        "p8": {
            "scoring_metric",
            "metric_weight",
            "top_features",
            "sig_cutoff",
            "scale_data",
            "exclude_plots",
            "show_plots",
            "include_ensembles",
            "multiclass_average",
        },
        "p9": {"sig_cutoff", "show_plots"},
        "p10": {
            "rep_data_path",
            "dataset_for_rep",
            "match_label",
            "exclude_plots",
            "show_plots",
        },
        "p11": {
            "report_modes",
            "reporting_dir",
            "outcome_label",
            "outcome_type",
            "instance_label",
            "make_pdf",
            "enable_plots",
            "reuse_existing_figures",
        },
    }

    for config_path in sorted((PROJECT_ROOT / "run_configs").glob("uci_*.cfg")):
        config = load_config(config_path)
        missing_run = required_run_keys.difference(config["run"])
        assert not missing_run, f"{config_path.name} missing [run] keys: {sorted(missing_run)}"
        for phase, keys in required_phase_keys.items():
            missing = keys.difference(config["phases"].get(phase, {}))
            assert not missing, f"{config_path.name} missing [{phase}] keys: {sorted(missing)}"


def test_example_configs_keep_current_fi_and_demo_model_sets():
    cases = {
        "uci_binary_hcc.cfg": {
            "model_type": "Binary",
            "expected_p6_models": {"NB", "LR", "DT"},
        },
        "uci_multiclass_student.cfg": {
            "model_type": "Multiclass",
            "expected_p6_models": {"NB", "LR", "DT"},
        },
        "uci_regression_auto_mpg.cfg": {
            "model_type": "Regression",
            "expected_p6_models": {"LR", "RF"},
        },
    }

    for config_name, expected in cases.items():
        config = load_config(PROJECT_ROOT / "run_configs" / config_name)
        p4_config = config["phases"]["p4"]
        p6_config = config["phases"]["p6"]

        assert csv_values(p4_config["models"]) == ["mutualinformation", "multiswrfdb"]
        assert set(p4_config["models_params"]) == {"mutualinformation", "multiswrfdb"}

        p6_models = set(csv_values(p6_config["models"]))
        assert p6_models == expected["expected_p6_models"]
        assert p6_config["model_params_json"] in (None, {})


def test_p6_runner_uses_outcome_type_as_public_task_parameter(tmp_path):
    output_path = tmp_path / "out"
    exp_root = output_path / "DemoExp"
    exp_root.mkdir(parents=True)

    runner = P6Runner(
        output_path=str(output_path),
        experiment_name="DemoExp",
        outcome_type="Continuous",
    )

    assert runner.outcome_type == "Continuous"
    assert runner.model_type == "Regression"

    legacy_runner = P6Runner(
        output_path=str(output_path),
        experiment_name="DemoExp",
        model_type="Regression",
    )

    assert legacy_runner.outcome_type == "Continuous"
    assert legacy_runner.model_type == "Regression"


def test_parallel_mode_name_and_p6_dispatch(monkeypatch, tmp_path):
    output_path = tmp_path / "out"
    dataset_dir = output_path / "DemoExp" / "DemoDataset"
    (dataset_dir / "CVDatasets").mkdir(parents=True)
    captured = {}

    def fake_parallel_jobs(function, jobs, **kwargs):
        captured["jobs"] = list(jobs)
        captured["function"] = function
        captured["label"] = kwargs.get("label")

    def fail_get_cluster(*args, **kwargs):
        raise AssertionError("Parallel mode should not use Dask cluster lookup")

    monkeypatch.setattr(p6_runner_module, "run_parallel_jobs", fake_parallel_jobs)
    monkeypatch.setattr(p6_runner_module, "get_cluster", fail_get_cluster)

    runner = P6Runner(
        output_path=str(output_path),
        experiment_name="DemoExp",
        outcome_type="Binary",
        models="NB,LR",
        n_splits=2,
        run_cluster="Parallel",
    )
    runner.run()

    assert len(captured["jobs"]) == 4
    assert {job[0] for job in captured["jobs"]} == {str(dataset_dir)}
    assert [getattr(job[1], "small_name") for job in captured["jobs"]] == ["NB", "NB", "LR", "LR"]
    assert [job[2] for job in captured["jobs"]] == [0, 1, 0, 1]
    assert "4 model/CV jobs" in captured["label"]
    assert callable(captured["function"])


def test_p6_local_dask_dispatch_reports_model_cv_job_count(monkeypatch, tmp_path):
    output_path = tmp_path / "out"
    dataset_dir = output_path / "DemoExp" / "DemoDataset"
    (dataset_dir / "CVDatasets").mkdir(parents=True)
    captured = {}

    class DummyContext:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_run_dask_tasks(tasks, client, label=None):
        captured["task_count"] = len(list(tasks))
        captured["label"] = label

    monkeypatch.setattr(p6_runner_module, "LocalCluster", DummyContext)
    monkeypatch.setattr(p6_runner_module, "Client", DummyContext)
    monkeypatch.setattr(p6_runner_module, "run_dask_tasks", fake_run_dask_tasks)

    runner = P6Runner(
        output_path=str(output_path),
        experiment_name="DemoExp",
        outcome_type="Binary",
        models="NB,LR",
        n_splits=2,
        run_cluster="Local",
    )
    runner.run()

    assert captured["task_count"] == 4
    assert "4 model/CV jobs" in captured["label"]
    assert "2 model(s)" in captured["label"]


def test_p6_bash_submission_writes_one_script_per_model_cv(monkeypatch, tmp_path):
    output_path = tmp_path / "out"
    dataset_dir = output_path / "DemoExp" / "DemoDataset"
    (dataset_dir / "CVDatasets").mkdir(parents=True)
    submitted = []

    monkeypatch.setattr(p6_runner_module.os, "system", lambda cmd: submitted.append(cmd) or 0)

    runner = P6Runner(
        output_path=str(output_path),
        experiment_name="DemoExp",
        outcome_type="Binary",
        models="NB,LR",
        n_splits=2,
        run_cluster="BashSLURM",
    )
    runner.run()

    scripts = sorted((output_path / "DemoExp" / "jobs").glob("P6_DemoDataset_*_run.sh"))
    assert len(scripts) == 4
    assert len(submitted) == 4

    script_text = "\n".join(path.read_text() for path in scripts)
    assert script_text.count("--model_id NB") == 2
    assert script_text.count("--model_id LR") == 2
    assert script_text.count("--cv_idx 0") == 2
    assert script_text.count("--cv_idx 1") == 2


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
