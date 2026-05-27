# tests/test_full_streamline_demodata_pipeline.py

import os
from pathlib import Path

import pytest

# pytest.skip("Tested Already", allow_module_level=True)

# Phase runners - adjust import paths if your repo differs
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


@pytest.mark.integration
def test_full_streamline_pipeline_uci_multiclass(tmp_path: Path):
    """
    End-to-end smoke test on the UCI Student Dropout multiclass demo data:

    P1: data process
    P2: impute & scale
    P3: feature learning
    P4: feature importance
    P5: feature selection
    P6: modeling
    P7: ensembles
    P8: statistics
    P9: dataset comparison
    P10: replication
    P11: reporting (standard + replication mode)

    This test intentionally keeps hyperparameters tiny / defaults where possible
    to keep runtime reasonable and only asserts for the presence of key artifacts.
    """

    # --- Layout ---------------------------------------------------------
    repo_root = Path(__file__).resolve().parent.parent.parent
    tmp_path = repo_root / "test"
    data_root = repo_root / "data" / "UCIMulticlassClassification"
    feature_root = repo_root / "data" / "UCIFeatureTypes"
    assert data_root.is_dir(), f"Expected UCI multiclass data under {data_root}"

    output_root = tmp_path / "out_full_pipeline"
    experiment_name = "UCIStudentDropoutExp"
    cv_splits = 3

    # P1 may create experiment folder itself; ensure parent exists
    output_root.mkdir(parents=True, exist_ok=True)

    # Convenience handle
    exp_root = output_root / experiment_name

    # ------------------------------------------------------------------
    # Phase 1: Data processing
    # ------------------------------------------------------------------
    p1 = P1Runner(
        data_path=str(data_root),
        output_path=str(output_root),
        experiment_name=experiment_name,
        outcome_label="Class",
        outcome_type="Multiclass",
        instance_label="InstanceID",
        n_splits=cv_splits,
        categorical_features=str(feature_root / "student_dropout_categorical_features.csv"),
        quantitative_features=str(feature_root / "student_dropout_quantitative_features.csv"),
        force=True
        # any other args you normally pass can be added here
    )
    p1.run()

    assert exp_root.is_dir(), "Phase 1 should create experiment directory"
    # At least one dataset directory with CVDatasets must exist
    datasets = [
        d for d in exp_root.iterdir()
        if d.is_dir() and (d / "CVDatasets").is_dir()
    ]
    assert datasets, "Expected at least one dataset with CVDatasets after Phase 1"
    ds_dir = datasets[0]  # use first dataset as canonical for downstream sanity checks

    # ------------------------------------------------------------------
    # Phase 2: Impute & scale
    # ------------------------------------------------------------------
    p2 = P2Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        instance_label="InstanceID",
        # use defaults for impute/scale options
        run_cluster="Serial",
    )
    p2.run()
    
    datasets = [
        d for d in exp_root.iterdir()
        if d.is_dir() and (d / "CVDatasets").is_dir()
    ]
    assert datasets, "Expected at least one dataset with CVDatasets after Phase 1"
    ds_dir = datasets[0]  # use first dataset as canonical for downstream sanity checks

    # Minimal sanity check: preprocessed data should exist (implementation-dependent)
    # assert (ds_dir / "ScaledData").exists() or (ds_dir / "ScaledData.csv").exists(), \
    #     "Phase 2 should produce scaled/imputed data artifacts"

    # ------------------------------------------------------------------
    # Phase 3: Feature learning
    # ------------------------------------------------------------------
    p3 = P3Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        instance_label="InstanceID",
        run_cluster="Serial",
    )
    p3.run()

    # Sanity: something like learned features / representation dir
    # Adjust name if your implementation differs
    assert any(
        (ds_dir / sub).exists()
        for sub in ["feature_learning"]
    ), "Phase 3 should produce some feature learning outputs"

    # ------------------------------------------------------------------
    # Phase 4: Feature importance
    # ------------------------------------------------------------------
    p4 = P4Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        instance_label="InstanceID",
        run_cluster="Serial",
    )
    p4.run()

    # There should be FI outputs of some kind
    assert (ds_dir / "feature_importance").exists() or (
        ds_dir / "model_evaluation" / "feature_importance"
    ).exists(), "Phase 4 should write feature importance artifacts"

    # ------------------------------------------------------------------
    # Phase 5: Feature selection
    # ------------------------------------------------------------------
    p5 = P5Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        instance_label="InstanceID",
        n_splits=cv_splits,
        run_cluster="Serial",
    )
    p5.run()

    # Check for feature selection outputs (e.g., SelectedFeatures.csv or similar)
    fs_dir_candidates = [
        ds_dir / "feature_selection",
    ]
    assert any(d.exists() for d in fs_dir_candidates), \
        "Phase 5 should write feature selection artifacts"

    # ------------------------------------------------------------------
    # Phase 6: Modeling
    # ------------------------------------------------------------------
    # Use a small model set + tiny search to keep tests fast, similar to earlier tests
    p6 = P6Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        outcome_label="Class",
        model_type="Multiclass",
        instance_label="InstanceID",
        n_splits=cv_splits,
        models="NB,LR,SVM",
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

    # ------------------------------------------------------------------
    # Phase 7: Ensembles
    # ------------------------------------------------------------------
    p7 = P7Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        n_splits=cv_splits,
        outcome_label="Class",
        instance_label="InstanceID",
        ensembles="hard_voting,soft_voting,stack_lr",
        base_models="NB,LR,SVM",
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
    assert (ens_root / "pickled_ensembles").is_dir(), "Expected pickled ensembles"
    assert list((ens_root / "pickled_ensembles").glob("*.pickle")), \
        "Expected at least one ensemble pickle from Phase 7"

    # ------------------------------------------------------------------
    # Phase 8: Statistics (per-dataset, base + ensembles)
    # ------------------------------------------------------------------
    p8 = P8Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        outcome_label="Class",
        outcome_type="Multiclass",
        instance_label="InstanceID",
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
    assert model_eval_dir.is_dir()
    assert (model_eval_dir / "Summary_performance_mean.csv").is_file(), \
        "Expected Summary_performance_mean.csv from Phase 8"

    # Ensemble summaries
    ens_summary_csvs = list(ens_root.glob("Ensembles*_performance_*.csv"))
    assert ens_summary_csvs, "Expected ensemble performance summary CSVs in Phase 8"

    # ------------------------------------------------------------------
    # Phase 9: Dataset-level comparison
    # ------------------------------------------------------------------
    p9 = P9Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        outcome_label="Class",
        outcome_type="Multiclass",
        instance_label="InstanceID",
        sig_cutoff=0.1,
        show_plots=False,
        run_cluster="Serial",
    )
    p9.run()

    dc_root = exp_root / "DatasetComparisons"
    assert dc_root.is_dir(), "Phase 9 should create DatasetComparisons directory"
    # e.g. Kruskal / Mann-Whitney outputs
    assert any(dc_root.glob("*.csv")), "Expected at least one dataset comparison CSV"

    # ------------------------------------------------------------------
    # Phase 10: Replication (whatever semantics you defined)
    # ------------------------------------------------------------------
    rep_data_root = repo_root / "data" / "UCIRepMulticlassClassification"
    assert rep_data_root.is_dir(), f"Expected UCI multiclass replication data under {rep_data_root}"

    dataset_for_rep = data_root / "student_dropout_academic_success.csv"
    assert dataset_for_rep.is_file(), f"Expected training dataset file at {dataset_for_rep}"

    p10 = P10Runner(
        rep_data_path=str(rep_data_root),
        dataset_for_rep=str(dataset_for_rep),
        output_path=str(output_root),
        experiment_name=experiment_name,
        run_cluster="Serial",
        show_plots=False,
    )
    p10.run()

    rep_root = exp_root / dataset_for_rep.stem / "replication"
    rep_ds_dir = rep_root / "student_dropout_academic_success_rep"
    assert rep_root.is_dir(), "Phase 10 should create replication directory under training dataset"
    assert rep_ds_dir.is_dir(), "Phase 10 should create replication dataset directory"
    assert (rep_ds_dir / "model_evaluation" / "Summary_performance_mean.csv").is_file(), \
        "Phase 10 should produce replication model evaluation summary"

    # ------------------------------------------------------------------
    # Phase 11: Reporting (Streamlit-based report rendered via WeasyPrint)
    # ------------------------------------------------------------------
    p11 = P11Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        outcome_label="Class",
        outcome_type="Multiclass",
        instance_label="InstanceID",
        run_cluster="Serial",
    )
    p11.run()

    # Expect at least a PDF under the experiment root
    pdf_candidates = list(exp_root.glob("**/*.pdf"))
    assert pdf_candidates, "Phase 11 should produce at least one PDF report"

    # Replication reporting mode (Phase 11)
    p11_rep = P11Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        report_mode="replication",
        outcome_label="Class",
        outcome_type="Multiclass",
        instance_label="InstanceID",
        run_cluster="Serial",
    )
    p11_rep.run()

    rep_report_json = exp_root / "reporting_replication" / "report_data.json"
    rep_report_pdf = exp_root / "reporting_replication" / "report.pdf"
    assert rep_report_json.is_file(), "Phase 11 replication mode should produce report_data.json"
    assert rep_report_pdf.is_file(), "Phase 11 replication mode should produce report.pdf"
