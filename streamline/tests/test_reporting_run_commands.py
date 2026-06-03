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
    assert any("Command Pickle: Found" in line for line in sections["Saved Command Pickle"])
    assert any("CV Splits: 3" in line for line in sections["Data and CV Settings"])
    assert any("SMOTE: False" in line for line in sections["EDA, Imputation, and SMOTE"])
    assert any("Models: NB,LR,DT" in line for line in sections["Modeling and Ensembles"])
    assert any("Replication Data Path: data/UCIRepBinaryClassification" in line for line in sections["Replication Settings"])


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
    assert any("Rep Report Focus: Held-out/external replication folders only" in line for line in sections["Replication Settings"])
    assert any("Report Mode: replication" in line for line in sections["Reporting Settings"])
    assert any("hcc_survival_rep from hcc_survival" in line for line in sections["Target Dataset(s)"])
