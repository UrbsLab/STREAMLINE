# tests/test_p6_p7_p8_integration.py

import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

# pytest.skip("Tested Already", allow_module_level=True)

# Phase 6 / 7 / 8 imports – adjust if your module paths differ
from streamline.p6_modeling.p6_runner import P6Runner
from streamline.p7_ensembles.p7_runner import P7Runner
from streamline.p8_summary_statistics.p8_runner import P8Runner


def _make_synthetic_cv_dataset(root: Path, n_splits: int = 3, n_samples: int = 120, random_state: int = 0):
    """
    Create a tiny synthetic binary classification dataset and write
    CV Train/Test CSVs into the expected STREAMLINE layout:

      <output>/<experiment>/<dataset>/CVDatasets/<dataset>_CV_<k>_Train.csv
      <output>/<experiment>/<dataset>/CVDatasets/<dataset>_CV_<k>_Test.csv

    Columns: InstanceID, Class, f0, f1, f2, ...
    """
    rng = np.random.RandomState(random_state)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        class_sep=1.2,
        flip_y=0.03,
        random_state=random_state,
    )

    idx = np.arange(n_samples)
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]

    dataset_name = "toy_dataset"
    ds_dir = root / dataset_name
    cv_dir = ds_dir / "CVDatasets"
    cv_dir.mkdir(parents=True, exist_ok=True)

    # simple CV: contiguous chunks
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    current = 0

    for cv_idx, fold_size in enumerate(fold_sizes):
        start, stop = current, current + fold_size
        current = stop

        test_mask = np.zeros(n_samples, dtype=bool)
        test_mask[start:stop] = True
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Build DataFrames
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        cols = ["InstanceID", "Class"] + [f"f{i}" for i in range(X.shape[1])]

        train_df = pd.DataFrame(
            np.column_stack(
                [np.arange(n_train), y_train, X_train]
            ),
            columns=cols,
        )
        test_df = pd.DataFrame(
            np.column_stack(
                [np.arange(n_test), y_test, X_test]
            ),
            columns=cols,
        )

        train_path = cv_dir / f"{dataset_name}_CV_{cv_idx}_Train.csv"
        test_path = cv_dir / f"{dataset_name}_CV_{cv_idx}_Test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

    # minimal metadata so phases that read it don't explode
    meta = {
        "Outcome Type": "Binary",
        "Outcome Label": "Class",
        "Instance Label": "InstanceID",
    }
    with open(root / "metadata.pickle", "wb") as f:
        pickle.dump(meta, f)

    return dataset_name


@pytest.mark.integration
def test_p6_p7_p8_pipeline():
    """
    End-to-end smoke test:

    1. Create synthetic CV datasets.
    2. Run Phase 6 modeling for a subset of models.
    3. Run Phase 7 ensembles using the base models.
    4. Run Phase 8 statistics summarization (incl. ensemble summary).
    5. Check that key artifacts were created.
    """
    tmp_path = Path("./test/")
    output_path = tmp_path / "out"
    experiment_name = "exp_integration"
    exp_root = output_path / experiment_name
    exp_root.mkdir(parents=True, exist_ok=True)

    # 1) Synthetic dataset (writes CVDatasets + metadata.pickle)
    dataset_name = _make_synthetic_cv_dataset(exp_root, n_splits=3, n_samples=90, random_state=42)
    ds_dir = exp_root / dataset_name

    # 2) Phase 6 – run a few simple models.
    # Adjust "NB,LR,SVM" to match registry IDs (small_name) in your p6 loader.
    p6 = P6Runner(
        output_path=str(output_path),
        experiment_name=experiment_name,
        outcome_label="Class",
        model_type="Binary",
        instance_label="InstanceID",
        n_splits=3,
        models="NB,LR,SVM",
        calibrate=False,
        scoring_metric="balanced_accuracy",
        metric_direction="maximize",
        n_trials=2,          # keep tiny for tests
        timeout=15,
        training_subsample=0,
        uniform_fi=False,
        save_plot=False,
        random_state=42,
        run_cluster="Serial",
    )
    p6.run()

    # Check that at least some base models were trained & pickled
    models_dir = ds_dir / "models" / "pickledModels"
    assert models_dir.is_dir(), "Phase 6 should create models/pickledModels"
    base_pickles = [p for p in models_dir.glob("*.pickle")]
    assert base_pickles, "Expected at least one base model pickle from Phase 6"

    # 3) Phase 7 – ensembles on top of the base models
    # ensembles: hard/soft voting and one stacking variant. Adjust IDs to match your get_ensemble_by_id registry.
    p7 = P7Runner(
        output_path=str(output_path),
        experiment_name=experiment_name,
        n_splits=3,
        outcome_label="Class",
        instance_label="InstanceID",
        ensembles="hard_voting,soft_voting,stack_lr",
        base_models="NB,LR,SVM",
        calibrate=0,             # keep off for speed
        calibrate_method="sigmoid",
        calibrate_cv=3,
        random_state=42,
        run_cluster="Serial",
    )
    p7.run()

    ens_root = ds_dir / "ensemble_evaluation"
    assert ens_root.is_dir(), "Phase 7 should create ensemble_evaluation directory"

    ens_models_dir = ens_root / "pickled_ensembles"
    assert ens_models_dir.is_dir(), "Ensembles should be pickled under ensemble_evaluation/pickled_ensembles"
    ens_pickles = list(ens_models_dir.glob("*.pickle"))
    assert ens_pickles, "Expected at least one ensemble pickle from Phase 7"

    # check metrics jsons
    metrics_dir = ens_root / "metrics_by_cv"
    assert metrics_dir.is_dir()
    metrics_files = list(metrics_dir.glob("*.json"))
    assert metrics_files, "Expected per-CV ensemble metrics JSONs"

    # sanity check a metric file structure
    with open(metrics_files[0]) as f:
        m = json.load(f)
    # metrics JSON now uses sklearn-style lowercase metric names
    assert "balanced_accuracy" in m and "accuracy" in m

    # 4) Phase 8 – statistics summarization (base models + ensemble summaries)
    p8 = P8Runner(
        output_path=str(output_path),
        experiment_name=experiment_name,
        outcome_label="Class",
        outcome_type="Binary",
        instance_label="InstanceID",
        n_splits=3,
        scoring_metric="balanced_accuracy",
        top_features=10,
        sig_cutoff=0.1,         # relaxed for tiny data
        metric_weight="balanced_accuracy",
        scale_data=True,
        exclude_plots=["plot_FI_box"],  # make tests faster / headless-safe
        show_plots=False,
        run_cluster="Serial",
    )
    p8.run()

    # 5) Verify statistics outputs

    # Base model summaries
    model_eval_dir = ds_dir / "model_evaluation"
    assert model_eval_dir.is_dir()

    summary_mean = model_eval_dir / "Summary_performance_mean.csv"
    assert summary_mean.is_file(), "Expected Summary_performance_mean.csv from P8Runner"

    df_mean = pd.read_csv(summary_mean)
    assert not df_mean.empty
    assert "Balanced Accuracy" in df_mean.columns

    # Ensemble summaries – file naming can be tweaked to match your implementation,
    # here we only assert that at least one Summary*.csv exists in ensemble_evaluation.
    ensemble_summary_csvs = list(ens_root.glob("Ensembles*_performance_*.csv"))
    assert (
        ensemble_summary_csvs
    ), "Expected at least one ensemble summary CSV (Ensembles*...) in ensemble_evaluation"

    # If a specific name is used (e.g. Summary_ensemble_performance_mean.csv), you can tighten this:
    # ens_summary_mean = ens_root / "Summary_ensemble_performance_mean.csv"
    # assert ens_summary_mean.is_file()
