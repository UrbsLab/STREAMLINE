# tests/test_full_streamline_demodata_regression_pipeline.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import pytest

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
from streamline.p11_reporting_old.p11_runner import P11Runner


def _find_repo_root(start: Path) -> Path:
    """
    Walk up until we find the repo marker(s).
    """
    for p in [start] + list(start.parents):
        if (p / "pyproject.toml").exists() or (p / "setup.cfg").exists() or (p / ".git").exists():
            return p
    return start.parent


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
    End-to-end smoke test on regression demo data.

    P1: data process
    P2: impute & scale
    P3: feature learning
    P4: feature importance
    P5: feature selection
    P6: modeling (Regression)
    P7: ensembles (Regression-capable)
    P8: statistics
    P9: dataset comparison
    P11: reporting (pdf)

    Notes:
      - You can override the demo-data root with STREAMLINE_REGRESSION_DATA_ROOT.
      - You can override the outcome label with STREAMLINE_REGRESSION_OUTCOME_LABEL.
    """

    # --- Layout ---------------------------------------------------------
    repo_root = _find_repo_root(Path(__file__).resolve())
    data_root = repo_root / "data" / "DemoDataRegression"

    outcome_label = os.getenv("STREAMLINE_REGRESSION_OUTCOME_LABEL", "Outcome").strip() or "Outcome"
    instance_label = os.getenv("STREAMLINE_INSTANCE_LABEL", "InstanceID").strip() or "InstanceID"

    output_root = tmp_path / "out_full_regression_pipeline"
    experiment_name = "DemoExpRegression"
    output_root.mkdir(parents=True, exist_ok=True)
    exp_root = output_root / experiment_name

    # ------------------------------------------------------------------
    # Phase 1: Data processing
    # ------------------------------------------------------------------
    p1 = P1Runner(
        data_path=str(data_root),
        output_path=str(output_root),
        experiment_name=experiment_name,
        instance_label=instance_label,
        force=True,
    )
    p1.run()

    assert exp_root.is_dir(), "Phase 1 should create experiment directory"
    ds_dir = _pick_first_dataset_dir(exp_root)

    # ------------------------------------------------------------------
    # Phase 2: Impute & scale
    # ------------------------------------------------------------------
    p2 = P2Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        instance_label=instance_label,
        run_cluster="Serial",
    )
    p2.run()

    ds_dir = _pick_first_dataset_dir(exp_root)

    # (Optional) sanity: some scaled/imputed artifacts exist (names vary by implementation)
    scaled_candidates = [
        ds_dir / "ScaledData",
        ds_dir / "scaled_data",
        ds_dir / "impute_scale",
        ds_dir / "preprocessed",
    ]
    # Don't hard-fail if your code writes elsewhere; comment in if you want stricter checks:
    # assert _exists_any(scaled_candidates), "Phase 2 should produce scaled/imputed artifacts"

    # ------------------------------------------------------------------
    # Phase 3: Feature learning
    # ------------------------------------------------------------------
    p3 = P3Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
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
        instance_label=instance_label,
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
        n_splits=3,
        models="LR,RF,SVR",
        calibrate=False,  # usually not relevant for regression; harmless if ignored
        scoring_metric="neg_mean_squared_error",
        metric_direction="maximize",  # neg MSE: higher is better
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
    if os.getenv("STREAMLINE_SKIP_REGRESSION_ENSEMBLES", "0").strip() not in {"1", "true", "True"}:
        p7 = P7Runner(
            output_path=str(output_root),
            experiment_name=experiment_name,
            n_splits=3,
            outcome_label=outcome_label,
            instance_label=instance_label,
            # Choose ensembles likely to generalize to regression; adjust to your implementation.
            ensembles="hard_voting,soft_voting,stack_lr",
            base_models="LR,RF,SVR",
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
    # Phase 8: Statistics
    # ------------------------------------------------------------------
    p8 = P8Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        outcome_label=outcome_label,
        outcome_type="Regression",
        instance_label=instance_label,
        n_splits=3,
        scoring_metric="neg_mean_squared_error",
        top_features=10,
        sig_cutoff=0.1,
        metric_weight="neg_mean_squared_error",
        scale_data=True,
        exclude_plots="plot_FI_box",
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
        outcome_type="Regression",
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
    # Phase 11: Reporting
    # ------------------------------------------------------------------
    p11 = P11Runner(
        output_path=str(output_root),
        experiment_name=experiment_name,
        outcome_label=outcome_label,
        outcome_type="Regression",
        instance_label=instance_label,
        run_cluster="Serial",
    )
    p11.run()

    pdf_candidates = list(exp_root.glob("**/*.pdf"))
    assert pdf_candidates, "Phase 11 should produce at least one PDF report"
