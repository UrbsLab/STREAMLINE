# tests/test_full_streamline_demodata_regression_pipeline.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

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

SKIP_TILL_MODELING_PHASES = os.getenv("STREAMLINE_SKIP_TO_REGRESSION_PHASE8", "0").strip().lower() in {"1", "true", "yes"}

def _pick_first_dataset_dir(exp_root: Path) -> Path:
    """
    STREAMLINE phase outputs typically look like:
      <output>/<experiment>/<DatasetName>/CVDatasets/...
    """
    datasets = [
        d for d in exp_root.iterdir()
        if d.is_dir() and (d / "CVDatasets").is_dir()
    ]
    assert datasets, f"Expected at least one dataset with CVDatasets under {exp_root}"
    return sorted(datasets)[0]


def _exists_any(path_candidates: Iterable[Path]) -> bool:
    return any(p.exists() for p in path_candidates)


@pytest.mark.integration
def test_full_streamline_pipeline_demodata_regression(tmp_path: Path):
    """
    End-to-end smoke test on the UCI Auto MPG regression demo data.

    P1: data process
    P2: impute & scale
    P3: feature learning
    P4: feature importance
    P5: feature selection
    P6: modeling (Regression)
    P7: ensembles (Regression-capable)
    P8: statistics
    P9: dataset comparison
    P10: replication
    P11: reporting (standard + replication mode)

    Notes:
      - Set STREAMLINE_SKIP_TO_REGRESSION_PHASE8=1 only when precomputed outputs already exist.
    """

    # --- Layout ---------------------------------------------------------
    outcome_label = "MPG"
    instance_label = "InstanceID"
    
    repo_root = Path(__file__).resolve().parent.parent.parent
    tmp_path = repo_root / "test"
    data_root = repo_root / "data" / "UCIRegression"
    feature_root = repo_root / "data" / "UCIFeatureTypes"
    assert data_root.is_dir(), f"Expected UCI regression data under {data_root}"

    output_root = tmp_path / "out_full_uci_regression_pipeline"
    experiment_name = "UCIAutoMPGRegression"
    cv_splits = 3
    output_root.mkdir(parents=True, exist_ok=True)
    exp_root = output_root / experiment_name
    
    if not SKIP_TILL_MODELING_PHASES:

        # ------------------------------------------------------------------
        # Phase 1: Data processing
        # ------------------------------------------------------------------
        p1 = P1Runner(
            data_path=str(data_root),
            output_path=str(output_root),
            experiment_name=experiment_name,
            outcome_label=outcome_label,
            outcome_type="Continuous",
            instance_label=instance_label,
            n_splits=cv_splits,
            categorical_features=str(feature_root / "auto_mpg_categorical_features.csv"),
            quantitative_features=str(feature_root / "auto_mpg_quantitative_features.csv"),
            force=True,
        )
        p1.run()

        assert exp_root.is_dir(), "Phase 1 should create experiment directory"
        ds_dir = _pick_first_dataset_dir(exp_root)
        
        # assert False, "Intentional stop after Phase 1 for testing purposes; comment out to run full pipeline"

        # ------------------------------------------------------------------
        # Phase 2: Impute & scale
        # ------------------------------------------------------------------
        p2 = P2Runner(
            output_path=str(output_root),
            experiment_name=experiment_name,
            outcome_label=outcome_label,
            instance_label=instance_label,
            run_cluster="Serial",
        )
        p2.run()

        ds_dir = _pick_first_dataset_dir(exp_root)

        # (Optional) sanity: some scaled/imputed artifacts exist (names vary by implementation)
        scaled_candidates = [
            ds_dir / "impute_scale",
        ]
        # Don't hard-fail if your code writes elsewhere; comment in if you want stricter checks:
        # assert _exists_any(scaled_candidates), "Phase 2 should produce scaled/imputed artifacts"

        # ------------------------------------------------------------------
        # Phase 3: Feature learning
        # ------------------------------------------------------------------
        p3 = P3Runner(
            output_path=str(output_root),
            experiment_name=experiment_name,
            outcome_label=outcome_label,
            instance_label=instance_label,
            run_cluster="Serial",
        )
        p3.run()

        assert _exists_any([ds_dir / "feature_learning"]), "Phase 3 should produce feature learning outputs"

        # ------------------------------------------------------------------
        # Phase 4: Feature importance
        # ------------------------------------------------------------------
        p4 = P4Runner(
            output_path=str(output_root),
            experiment_name=experiment_name,
            outcome_label=outcome_label,
            outcome_type="Continuous",
            instance_label=instance_label,
            run_cluster="Serial",
        )
        p4.run()

        assert _exists_any([
            ds_dir / "feature_importance",
            ds_dir / "model_evaluation" / "feature_importance",
        ]), "Phase 4 should write feature importance artifacts"

        # ------------------------------------------------------------------
        # Phase 5: Feature selection
        # ------------------------------------------------------------------
        p5 = P5Runner(
            output_path=str(output_root),
            experiment_name=experiment_name,
            outcome_label=outcome_label,
            instance_label=instance_label,
            n_splits=cv_splits,
            run_cluster="Serial",
        )
        p5.run()

        assert _exists_any([ds_dir / "feature_selection"]), "Phase 5 should write feature selection artifacts"

        # ------------------------------------------------------------------
        # Phase 6: Modeling (Regression)
        # ------------------------------------------------------------------
        # Keep runtime tiny: small model set + tiny Optuna search
        #
        # Adjust 'models=' to whatever your regression registry supports.
        # Common STREAMLINE abbreviations often include: LR, RF, SVR, EN (ElasticNet), LASSO, RIDGE, XGB, etc.
        p6 = P6Runner(
            output_path=str(output_root),
            experiment_name=experiment_name,
            outcome_label=outcome_label,
            model_type="Regression",
            instance_label=instance_label,
            n_splits=cv_splits,
            models="LR,RF,SVR",
            calibrate=False,  # usually not relevant for regression; harmless if ignored
            scoring_metric="neg_mean_squared_error", # prefix with 'neg_' if using sklearn convention, don't change direction
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
        # Phase 7: Ensembles (Regression)
        # ------------------------------------------------------------------
        # If your ensemble phase is classification-only, you can skip by setting:
        #   STREAMLINE_SKIP_REGRESSION_ENSEMBLES=1
        # if os.getenv("STREAMLINE_SKIP_REGRESSION_ENSEMBLES", "0").strip() not in {"1", "true", "True"}:
        #     p7 = P7Runner(
        #         output_path=str(output_root),
        #         experiment_name=experiment_name,
        #         n_splits=cv_splits,
        #         outcome_label=outcome_label,
        #         instance_label=instance_label,
        #         # Choose ensembles likely to generalize to regression; adjust to your implementation.
        #         ensembles="hard_voting,soft_voting,stack_lr",
        #         base_models="LR,RF,SVR",
        #         meta_train_source="train",
        #         calibrate=0,
        #         calibrate_method="sigmoid",
        #         calibrate_cv=3,
        #         run_cluster="Serial",
        #         queue="defq",
        #         reserved_memory=4,
        #         random_state=42,
        #     )
        #     p7.run()

        #     ens_root = ds_dir / "ensemble_evaluation"
        #     assert ens_root.is_dir(), "Phase 7 should create ensemble_evaluation directory"
        #     assert (ens_root / "pickled_ensembles").is_dir(), "Expected pickled ensembles"
        #     assert list((ens_root / "pickled_ensembles").glob("*.pickle")), \
        #         "Expected at least one ensemble pickle from Phase 7"
    else:
        print("Skipping directly to Phase 8+ for faster testing of later phases")
        assert exp_root.is_dir(), "Phase 1 should create experiment directory"
        ds_dir = _pick_first_dataset_dir(exp_root)
        
        # (Optional) sanity: some scaled/imputed artifacts exist (names vary by implementation)
        scaled_candidates = [
            ds_dir / "impute_scale",
        ]
        # Don't hard-fail if your code writes elsewhere; comment in if you want stricter checks:
        # assert _exists_any(scaled_candidates), "Phase 2 should produce scaled/imputed artifacts"

        assert _exists_any([ds_dir / "feature_learning"]), "Phase 3 should produce feature learning outputs"
        
        assert _exists_any([
            ds_dir / "feature_importance",
            ds_dir / "model_evaluation" / "feature_importance",
        ]), "Phase 4 should write feature importance artifacts"
        
        assert _exists_any([ds_dir / "feature_selection"]), "Phase 5 should write feature selection artifacts"
        
        models_dir = ds_dir / "models" / "pickledModels"
        assert models_dir.is_dir(), "Phase 6 should create pickled base models"
        assert list(models_dir.glob("*.pickle")), "Expected base model pickles"
        
        # ens_root = ds_dir / "ensemble_evaluation"
        # assert ens_root.is_dir(), "Phase 7 should create ensemble_evaluation directory"
        # assert (ens_root / "pickled_ensembles").is_dir(), "Expected pickled ensembles"
        # assert list((ens_root / "pickled_ensembles").glob("*.pickle")), \
        #     "Expected at least one ensemble pickle from Phase 7"

    # ------------------------------------------------------------------
    # Phase 8: Statistics
    # ------------------------------------------------------------------
        
    p8 = P8Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        outcome_label=outcome_label,
        outcome_type="Continuous",
        instance_label=instance_label,
        n_splits=cv_splits,
        scoring_metric="mean_squared_error",
        top_features=10,
        sig_cutoff=0.1,
        metric_weight="mean_squared_error",
        scale_data=True,
        exclude_plots=None,
        show_plots=False,
        run_cluster="Serial",
    )
    p8.run()

    model_eval_dir = ds_dir / "model_evaluation"
    assert model_eval_dir.is_dir(), "Phase 8 should create model_evaluation directory"
    assert (model_eval_dir / "Summary_performance_mean.csv").is_file(), \
        "Expected Summary_performance_mean.csv from Phase 8"

    # ------------------------------------------------------------------
    # Phase 9: Dataset-level comparison
    # ------------------------------------------------------------------
    p9 = P9Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        outcome_label=outcome_label,
        outcome_type="Continuous",
        instance_label=instance_label,
        sig_cutoff=0.1,
        show_plots=False,
        run_cluster="Serial",
    )
    p9.run()

    dc_root = exp_root / "DatasetComparisons"
    assert dc_root.is_dir(), "Phase 9 should create DatasetComparisons directory"
    assert any(dc_root.glob("*.csv")), "Expected at least one dataset comparison CSV"

    # ------------------------------------------------------------------
    # Phase 10: Replication
    # ------------------------------------------------------------------
    rep_data_root = repo_root / "data" / "UCIRepRegression"
    assert rep_data_root.is_dir(), f"Expected UCI regression replication data under {rep_data_root}"

    dataset_for_rep = data_root / "auto_mpg.csv"
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
    rep_ds_dir = rep_root / "auto_mpg_rep"
    assert rep_root.is_dir(), "Phase 10 should create replication directory under training dataset"
    assert rep_ds_dir.is_dir(), "Phase 10 should create replication dataset directory"
    assert (rep_ds_dir / "model_evaluation" / "Summary_performance_mean.csv").is_file(), \
        "Phase 10 should produce replication model evaluation summary"

    # ------------------------------------------------------------------
    # Phase 11: Reporting
    # ------------------------------------------------------------------
    p11 = P11Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        outcome_label=outcome_label,
        outcome_type="Continuous",
        instance_label=instance_label,
        run_cluster="Serial",
    )
    p11.run()

    pdf_candidates = list(exp_root.glob("**/*.pdf"))
    assert pdf_candidates, "Phase 11 should produce at least one PDF report"

    # Replication reporting mode (Phase 11)
    p11_rep = P11Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        report_mode="replication",
        outcome_label=outcome_label,
        outcome_type="Continuous",
        instance_label=instance_label,
        run_cluster="Serial",
    )
    p11_rep.run()

    rep_report_json = exp_root / "reporting_replication" / "report_data.json"
    rep_report_pdf = exp_root / "reporting_replication" / "report.pdf"
    assert rep_report_json.is_file(), "Phase 11 replication mode should produce report_data.json"
    assert rep_report_pdf.is_file(), "Phase 11 replication mode should produce report.pdf"
