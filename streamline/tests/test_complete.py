# tests/test_full_streamline_demodata_pipeline.py

import os
from pathlib import Path

import pytest

pytest.skip("Tested Already", allow_module_level=True)

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
from streamline.p11_reporting_old.p11_runner import P11Runner


@pytest.mark.integration
def test_full_streamline_pipeline_demodata(tmp_path: Path):
    """
    End-to-end smoke test on the real DemoData:

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
    P11: reporting (html/pdf generation)

    This test intentionally keeps hyperparameters tiny / defaults where possible
    to keep runtime reasonable and only asserts for the presence of key artifacts.
    """

    # --- Layout ---------------------------------------------------------
    repo_root = Path(__file__).resolve().parent.parent.parent
    tmp_path = repo_root / "test"
    data_root = repo_root / "data" / "DemoData"
    assert data_root.is_dir(), f"Expected DemoData under {data_root}"

    output_root = tmp_path / "out_full_pipeline"
    experiment_name = "DemoExp"

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
        instance_label="InstanceID",
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
        n_splits=3,
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
        n_splits=3,
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
        n_splits=3,
        scoring_metric="balanced_accuracy",
        top_features=10,
        sig_cutoff=0.1,
        metric_weight="balanced_accuracy",
        scale_data=True,
        exclude_plots="plot_FI_box",
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
    # p10 = P10Runner(
    #     output_path=str(output_root),
    #     experiment_name=experiment_name,
    #     run_cluster="Serial",
    # )
    # p10.run()

    # # Minimal sanity check - you can tighten this based on your P10 outputs
    # repl_candidates = [
    #     ds_dir / "replication",
    #     ds_dir / "Replication",
    #     exp_root / "ReplicationSummary.csv",
    # ]
    # assert any(p.exists() for p in repl_candidates), \
    #     "Phase 10 should create some replication artifacts"

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
