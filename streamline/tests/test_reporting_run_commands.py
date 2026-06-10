from __future__ import annotations

import json
import pickle
from pathlib import Path

from streamline.p11_reporting.reporting import ReportPhaseJob
from streamline.utils.run_commands import save_phase_run_command


def make_dataset(path: Path) -> None:
    (path / "exploratory").mkdir(parents=True, exist_ok=True)
    (path / "model_evaluation").mkdir(parents=True, exist_ok=True)


def save_demo_commands(exp_root: Path) -> None:
    save_phase_run_command(
        exp_root,
        "p1_data_process",
        {
            "data_path": "data/UCIBinaryClassification",
            "output_path": str(exp_root.parent),
            "experiment_name": exp_root.name,
            "outcome_label": "Class",
            "outcome_type": "Binary",
            "instance_label": "InstanceID",
            "n_splits": 3,
            "partition_method": "Stratified",
            "one_hot_encoding": True,
            "categorical_features": "data/UCIFeatureTypes/hcc_survival_categorical_features.csv",
            "quantitative_features": "data/UCIFeatureTypes/hcc_survival_quantitative_features.csv",
        },
        argv=["python", "-m", "streamline.p1_data_process.p1_cli"],
    )
    save_phase_run_command(
        exp_root,
        "p2_impute_scale",
        {
            "output_path": str(exp_root.parent),
            "experiment_name": exp_root.name,
            "smote": False,
            "smote_method": "auto",
        },
        argv=["python", "-m", "streamline.p2_impute_scale.p2_cli"],
    )
    save_phase_run_command(
        exp_root,
        "p6_modeling",
        {
            "output_path": str(exp_root.parent),
            "experiment_name": exp_root.name,
            "model_type": "Binary",
            "models": "NB,LR,DT",
            "scoring_metric": "balanced_accuracy",
            "metric_direction": "maximize",
            "n_trials": 200,
            "timeout": 900,
            "bypass_one_hot_for_native_models": 1,
            "native_categorical_models": "CGB",
        },
        argv=["python", "-m", "streamline.p6_modeling.p6_cli"],
    )
    save_phase_run_command(
        exp_root,
        "p10_replication",
        {
            "output_path": str(exp_root.parent),
            "experiment_name": exp_root.name,
            "rep_data_path": "data/UCIRepBinaryClassification",
            "dataset_for_rep": "data/UCIBinaryClassification/hcc_survival.csv",
            "show_plots": 0,
        },
        argv=["python", "-m", "streamline.p10_replication.p10_cli"],
    )


def summary_lines(payload: dict) -> list[str]:
    lines = []
    for section in payload["run_command_summary"]["sections"]:
        lines.extend(section["lines"])
    return lines


def summary_titles(payload: dict) -> list[str]:
    return [
        section["title"]
        for section in payload["run_command_summary"]["sections"]
    ]


def test_report_data_includes_saved_command_summary(tmp_path: Path):
    exp_root = tmp_path / "out" / "DemoExp"
    exp_root.mkdir(parents=True)
    with (exp_root / "metadata.pickle").open("wb") as handle:
        pickle.dump(
            {
                "Outcome Label": "Class",
                "Outcome Type": "Binary",
                "Instance Label": "InstanceID",
            },
            handle,
        )
    make_dataset(exp_root / "hcc_survival")
    save_demo_commands(exp_root)

    ReportPhaseJob(
        experiment_path=str(exp_root),
        report_mode="standard",
        make_pdf=False,
        enable_plots=False,
    ).run()

    payload = json.loads((exp_root / "reporting" / "report_data.json").read_text())
    summary = payload["run_command_summary"]
    sections = {section["title"]: section["lines"] for section in summary["sections"]}

    assert summary["present"] is True
    assert "Saved Command Pickle" not in summary_titles(payload)
    assert "P10 Replication Settings" not in sections
    assert any("CV Splits: 3" in line for line in sections["P1 Data Processing and CV"])
    assert any("SMOTE: False" in line for line in sections["P1-P2 EDA, Scaling, Imputation, and SMOTE"])
    assert any("Models: NB,LR,DT" in line for line in sections["P6-P8 Modeling, Ensembles, and Metrics"])
    assert any("Bypass One-Hot for Native Categorical Models: 1" in line for line in sections["P6-P8 Modeling, Ensembles, and Metrics"])
    assert not any(line.startswith("Selector:") for line in summary_lines(payload))
    assert not any("Task Type:" in line for line in sections["Target Dataset(s)"])
    assert not any("Not specified" in line for line in summary_lines(payload))


def test_replication_report_first_page_summary_uses_replication_settings(tmp_path: Path):
    exp_root = tmp_path / "out" / "DemoExp"
    exp_root.mkdir(parents=True)
    with (exp_root / "metadata.pickle").open("wb") as handle:
        pickle.dump({"Outcome Label": "Class", "Outcome Type": "Binary"}, handle)

    train_ds = exp_root / "hcc_survival"
    make_dataset(train_ds)
    make_dataset(train_ds / "replication" / "hcc_survival_rep")
    save_demo_commands(exp_root)
    save_phase_run_command(
        exp_root,
        "p11_reporting",
        {
            "experiment_path": str(exp_root),
            "report_mode": "standard",
            "make_pdf": True,
            "enable_plots": True,
            "reuse_existing_figures": True,
        },
        argv=["python", "-m", "streamline.p11_reporting.p11_cli", "--report_mode", "standard"],
    )

    ReportPhaseJob(
        experiment_path=str(exp_root),
        report_mode="replication",
        make_pdf=False,
        enable_plots=False,
    ).run()

    payload = json.loads((exp_root / "reporting_replication" / "report_data.json").read_text())
    sections = {section["title"]: section["lines"] for section in payload["run_command_summary"]["sections"]}

    assert payload["report_mode"] == "replication"
    assert any("Rep Report Focus: Held-out/external replication folders only" in line for line in sections["P10 Replication Settings"])
    assert any("Report Mode: replication" in line for line in sections["P11 Reporting Settings"])
    assert any("hcc_survival_rep from hcc_survival" in line for line in sections["Target Dataset(s)"])
    assert not any("Task Type:" in line for line in sections["Target Dataset(s)"])
    assert not any("Not specified" in line for line in summary_lines(payload))


def test_legacy_report_summary_uses_run_params_and_artifacts(tmp_path: Path):
    exp_root = tmp_path / "out" / "LegacyExp"
    exp_root.mkdir(parents=True)
    with (exp_root / "metadata.pickle").open("wb") as handle:
        pickle.dump(
            {
                "Data Path": "data/Legacy",
                "Outcome Label": "Class",
                "Outcome Type": "Binary",
                "Instance Label": "InstanceID",
                "CV Partitions": 3,
                "Partition Method": "Stratified",
                "Specified Categorical Features": "categorical.csv",
                "Specified Quantitative Features": "quantitative.csv",
            },
            handle,
        )
    with (exp_root / "run_params.pickle").open("wb") as handle:
        pickle.dump(
            {
                "2026-01-01T00:00:00": {
                    "data_path": "data/Legacy",
                    "n_splits": 3,
                    "partition_method": "Stratified",
                    "one_hot_encoding": True,
                },
                "2026-01-01T00:00:01": {
                    "phase": "p2_impute_scale",
                    "scale_data": True,
                    "impute_data": True,
                    "multi_impute": False,
                    "overwrite_cv": True,
                    "smote": False,
                    "smote_method": "auto",
                },
                "2026-01-01T00:00:02": {
                    "phase": "p4_feature_importance",
                    "models": ["mutualinformation"],
                    "models_params": {},
                },
            },
            handle,
        )

    ds = exp_root / "legacy_data"
    make_dataset(ds)
    (ds / "impute_scale").mkdir(parents=True)
    (ds / "impute_scale" / "scaler_cv0.pickle").write_bytes(b"")
    (ds / "feature_importance" / "mutualinformation").mkdir(parents=True)
    (ds / "feature_importance" / "mutualinformation" / "mutualinformation_scores_cv_0.csv").write_text("feature,score\nx,1\n")
    (ds / "feature_selection").mkdir(parents=True)
    (ds / "feature_selection" / "InformativeFeatureSummary.csv").write_text("CV_Partition,Informative,Uninformative\n0,2,0\n")
    (ds / "models" / "pickledModels").mkdir(parents=True)
    (ds / "models" / "pickledModels" / "LR_0.pickle").write_bytes(b"")
    (ds / "models" / "optuna_trials").mkdir(parents=True)
    (ds / "models" / "optuna_trials" / "LR_optuna_trials0.csv").write_text("number,value\n0,0.5\n1,0.6\n")
    (ds / "ensemble_evaluation" / "metrics_by_cv").mkdir(parents=True)
    (ds / "ensemble_evaluation" / "metrics_by_cv" / "HEV_CV_0.json").write_text("{}")
    make_dataset(ds / "replication" / "legacy_data_rep")

    ReportPhaseJob(
        experiment_path=str(exp_root),
        report_mode="replication",
        make_pdf=False,
        enable_plots=False,
    ).run()

    payload = json.loads((exp_root / "reporting_replication" / "report_data.json").read_text())
    lines = summary_lines(payload)

    assert "Saved Command Pickle" not in summary_titles(payload)
    assert any("Data Path: data/Legacy" in line for line in lines)
    assert any("SMOTE: False" in line for line in lines)
    assert any("FI Models: mutualinformation" in line for line in lines)
    assert any("Feature Learner: Not run" in line for line in lines)
    assert any("Models: LR" in line for line in lines)
    assert any("Ensembles: hard_voting" in line for line in lines)
    assert any("Optuna Trials Completed: 2 completed across 1 model/CV runs" in line for line in lines)
    assert any("Replication Data Path: legacy_data_rep from legacy_data" in line for line in lines)
    assert not any("Not specified" in line for line in lines)
