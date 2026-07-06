from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from streamline.p1_data_process.p1_runner import P1Runner
from streamline.p2_impute_scale.p2_runner import P2Runner
from streamline.p3_feature_learning.p3_runner import P3Runner
from streamline.p4_feature_importance.p4_runner import P4Runner
from streamline.p5_feature_selection.p5_runner import P5Runner
from streamline.p6_modeling.p6_runner import P6Runner
from streamline.p7_ensembles.p7_runner import P7Runner
from streamline.p8_summary_statistics.p8_runner import P8Runner
from streamline.p9_compare_datasets.p9_runner import P9Runner
from streamline.p10_replication.p10_runner import P10Runner
from streamline.p11_reporting.p11_runner import P11Runner


def pick_first_dataset_dir(exp_root: Path) -> Path:
    datasets = [
        d for d in exp_root.iterdir()
        if d.is_dir() and (d / "CVDatasets").is_dir()
    ]
    assert datasets, f"Expected at least one dataset with CVDatasets under {exp_root}"
    return sorted(datasets)[0]


@pytest.mark.integration
def test_full_streamline_pipeline_uci_binary_hcc(tmp_path: Path):
    repo_root = Path(__file__).resolve().parent.parent.parent
    tmp_path = repo_root / "test"
    data_root = repo_root / "data" / "UCIBinaryClassification"
    rep_data_root = repo_root / "data" / "UCIRepBinaryClassification"
    feature_root = repo_root / "data" / "UCIFeatureTypes"

    assert data_root.is_dir(), f"Expected UCI binary data under {data_root}"
    assert rep_data_root.is_dir(), f"Expected UCI binary replication data under {rep_data_root}"

    output_root = tmp_path / "out_full_uci_binary_pipeline"
    experiment_name = "UCIHCCBinary"
    outcome_label = "Class"
    instance_label = "InstanceID"
    cv_splits = 3
    output_root.mkdir(parents=True, exist_ok=True)
    exp_root = output_root / experiment_name

    p1 = P1Runner(
        data_path=str(data_root),
        output_path=str(output_root),
        experiment_name=experiment_name,
        outcome_label=outcome_label,
        outcome_type="Binary",
        instance_label=instance_label,
        n_splits=cv_splits,
        categorical_features=str(feature_root / "hcc_survival_categorical_features.csv"),
        quantitative_features=str(feature_root / "hcc_survival_quantitative_features.csv"),
        force=True,
    )
    p1.run()

    assert exp_root.is_dir(), "Phase 1 should create experiment directory"
    ds_dir = pick_first_dataset_dir(exp_root)

    p2 = P2Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        outcome_label=outcome_label,
        instance_label=instance_label,
        run_cluster="Serial",
    )
    p2.run()

    p3 = P3Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        outcome_label=outcome_label,
        instance_label=instance_label,
        run_cluster="Serial",
    )
    p3.run()

    assert (ds_dir / "feature_learning").exists(), "Phase 3 should produce feature learning outputs"

    p4 = P4Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        models="mutualinformation,multisurf",
        models_params={"mutualinformation": {"outcome_type": "Binary"}, "multisurf": {"n_jobs": 1}},
        outcome_label=outcome_label,
        outcome_type="Binary",
        instance_label=instance_label,
        instance_subset=2000,
        run_cluster="Serial",
    )
    p4.run()

    fi_dir = ds_dir / "feature_importance"
    assert fi_dir.exists(), "Phase 4 should write feature importance artifacts"
    selector_path = fi_dir / "multisurf" / "selector_cv0.pickle"
    assert selector_path.is_file(), "Phase 4 should save the MultiSURF selector payload"
    with open(selector_path, "rb") as f:
        selector_payload = pickle.load(f)
    assert selector_payload["instance_subset"] == 2000

    p5 = P5Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        outcome_label=outcome_label,
        instance_label=instance_label,
        n_splits=cv_splits,
        run_cluster="Serial",
    )
    p5.run()

    assert (ds_dir / "feature_selection").exists(), "Phase 5 should write feature selection artifacts"

    p6 = P6Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        outcome_label=outcome_label,
        outcome_type="Binary",
        instance_label=instance_label,
        n_splits=cv_splits,
        models="NB,LR,DT",
        calibrate=False,
        scoring_metric="balanced_accuracy",
        metric_direction="maximize",
        n_trials=2,
        timeout=15,
        training_subsample=0,
        uniform_fi=False,
        save_plot=False,
        random_state=42,
        run_cluster="Serial",
    )
    p6.run()

    models_dir = ds_dir / "models" / "pickledModels"
    assert models_dir.is_dir(), "Phase 6 should create pickled base models"
    assert list(models_dir.glob("*.pickle")), "Expected base model pickles"

    p7 = P7Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        n_splits=cv_splits,
        outcome_label=outcome_label,
        instance_label=instance_label,
        ensembles="hard_voting,soft_voting,stack_lr",
        base_models="NB,LR,DT",
        meta_train_source="train",
        calibrate=0,
        calibrate_method="sigmoid",
        calibrate_cv=3,
        run_cluster="Serial",
        queue="defq",
        reserved_memory=4,
        random_state=42,
    )
    p7.run()

    ens_root = ds_dir / "ensemble_evaluation"
    assert ens_root.is_dir(), "Phase 7 should create ensemble_evaluation directory"
    assert list((ens_root / "pickled_ensembles").glob("*.pickle")), "Expected at least one ensemble pickle"

    p8 = P8Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        outcome_label=outcome_label,
        outcome_type="Binary",
        instance_label=instance_label,
        n_splits=cv_splits,
        scoring_metric="balanced_accuracy",
        top_features=10,
        sig_cutoff=0.1,
        metric_weight="balanced_accuracy",
        scale_data=True,
        exclude_plots=None,
        show_plots=False,
        run_cluster="Serial",
    )
    p8.run()

    model_eval_dir = ds_dir / "model_evaluation"
    assert (model_eval_dir / "Summary_performance_mean.csv").is_file(), \
        "Expected Summary_performance_mean.csv from Phase 8"

    p9 = P9Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        outcome_label=outcome_label,
        outcome_type="Binary",
        instance_label=instance_label,
        sig_cutoff=0.1,
        show_plots=False,
        run_cluster="Serial",
    )
    p9.run()

    dc_root = exp_root / "DatasetComparisons"
    assert dc_root.is_dir(), "Phase 9 should create DatasetComparisons directory"
    assert any(dc_root.glob("*.csv")), "Expected at least one dataset comparison CSV"

    dataset_for_rep = data_root / "hcc_survival.csv"
    p10 = P10Runner(
        rep_data_path=str(rep_data_root),
        dataset_for_rep=str(dataset_for_rep),
        output_path=str(output_root),
        experiment_name=experiment_name,
        run_cluster="Serial",
        show_plots=False,
    )
    p10.run()

    rep_ds_dir = exp_root / dataset_for_rep.stem / "replication" / "hcc_survival_rep"
    assert rep_ds_dir.is_dir(), "Phase 10 should create replication dataset directory"
    assert (rep_ds_dir / "model_evaluation" / "Summary_performance_mean.csv").is_file(), \
        "Phase 10 should produce replication model evaluation summary"

    p11 = P11Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        outcome_label=outcome_label,
        outcome_type="Binary",
        instance_label=instance_label,
        run_cluster="Serial",
    )
    p11.run()

    assert list(exp_root.glob("**/*.pdf")), "Phase 11 should produce at least one PDF report"

    p11_rep = P11Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        report_mode="replication",
        outcome_label=outcome_label,
        outcome_type="Binary",
        instance_label=instance_label,
        run_cluster="Serial",
    )
    p11_rep.run()

    rep_report_json = exp_root / "reporting_replication" / "report_data.json"
    rep_report_pdf = exp_root / "reporting_replication" / f"{experiment_name}_STREAMLINE_Replication_Report.pdf"
    assert rep_report_json.is_file(), "Phase 11 replication mode should produce report_data.json"
    assert rep_report_pdf.is_file(), "Phase 11 replication mode should produce an experiment-named replication PDF"
