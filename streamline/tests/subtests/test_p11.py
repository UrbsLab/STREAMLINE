import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

pytest.skip("Tested Already", allow_module_level=True)

from streamline.p6_modeling.p6_runner import P6Runner
from streamline.p7_ensembles.p7_runner import P7Runner
from streamline.p8_summary_statistics.p8_runner import P8Runner
from streamline.p9_compare_datasets.p9_runner import P9Runner
from streamline.p11_reporting.p11_runner import P11Runner


def _make_synthetic_cv_dataset(root: Path, n_splits: int = 3, n_samples: int = 120, random_state: int = 0):
    """
    Same helper shape as in test_p6_p7_p8_integration.py:
    creates <root>/<dataset>/CVDatasets with CV Train/Test CSVs and metadata.pickle.
    """
    from sklearn.datasets import make_classification

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

    meta = {
        "Outcome Type": "Binary",
        "Outcome Label": "Class",
        "Instance Label": "InstanceID",
    }
    with open(root / "metadata.pickle", "wb") as f:
        pickle.dump(meta, f)

    return dataset_name


@pytest.mark.integration
def test_p11_reporting_end_to_end():
    """
    End-to-end smoke test for reporting phase:

    1. Create synthetic CV datasets.
    2. Run Phase 6 modeling (a few models).
    3. Run Phase 7 ensembles.
    4. Run Phase 8 statistics.
    5. Run Phase 9 dataset comparisons.
    6. Run Phase 10 reporting.
    7. Check that HTML + PDF reports exist and are non-empty.
    """
    tmp_path = Path('./test')
    # --- basic layout ---
    output_path = tmp_path / "out"
    experiment_name = "exp_reporting"
    exp_root = output_path / experiment_name
    exp_root.mkdir(parents=True, exist_ok=True)

    # synthetic dataset
    dataset_name = _make_synthetic_cv_dataset(
        exp_root, n_splits=3, n_samples=90, random_state=42
    )
    ds_dir = exp_root / dataset_name

    # --- Phase 6: modeling ---
    p6 = P6Runner(
        output_path=str(output_path),
        experiment_name=experiment_name,
        outcome_label="Class",
        model_type="Binary",
        instance_label="InstanceID",
        n_splits=3,
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

    # sanity: base models exist
    models_dir = ds_dir / "models" / "pickledModels"
    assert models_dir.is_dir()
    assert list(models_dir.glob("*.pickle"))

    # --- Phase 7: ensembles ---
    p7 = P7Runner(
        output_path=str(output_path),
        experiment_name=experiment_name,
        n_splits=3,
        outcome_label="Class",
        instance_label="InstanceID",
        ensembles="hard_voting,soft_voting,stack_lr",
        base_models="NB,LR,DT",
        meta_train_source="train",
        calibrate=0,
        calibrate_method="sigmoid",
        calibrate_cv=3,
        random_state=42,
        run_cluster="Serial",
    )
    p7.run()

    ens_root = ds_dir / "ensemble_evaluation"
    assert ens_root.is_dir()
    assert list((ens_root / "metrics_by_cv").glob("*.json"))

    # --- Phase 8: statistics ---
    from streamline.p8_summary_statistics.p8_runner import P8Runner

    p8 = P8Runner(
        output_path=str(output_path),
        experiment_name=experiment_name,
        outcome_label="Class",
        outcome_type="Binary",
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
    assert (model_eval_dir / "Summary_performance_mean.csv").is_file()

    # --- Phase 9: dataset comparisons ---
    from streamline.p9_compare_datasets.p9_runner import P9Runner

    p9 = P9Runner(
        output_path=str(output_path),
        experiment_name=experiment_name,
        outcome_label="Class",
        outcome_type="Binary",
        instance_label="InstanceID",
        sig_cutoff=0.1,
        show_plots=False,
        run_cluster="Serial",
    )
    p9.run()

    dc_root = exp_root / "DatasetComparisons"
    assert dc_root.is_dir()
    assert list(dc_root.glob("KruskalWallis_*.csv"))

    # --- Phase 10: reporting ---
    from streamline.p11_reporting.p11_runner import P11Runner

    p10 = P11Runner(
        output_path=str(output_path),
        experiment_name=experiment_name,
        run_cluster="Serial",
    )
    p10.run()

    reports_dir = exp_root / "reporting"
    html_report = reports_dir / "report.html"
    pdf_report = reports_dir / "report.pdf"

    assert reports_dir.is_dir(), "Reporting phase should create a reports/ directory"
    assert html_report.is_file(), "Expected HTML report from reporting phase"
    assert pdf_report.is_file(), "Expected PDF report from reporting phase"

    # basic non-empty checks
    assert html_report.stat().st_size > 0
    assert pdf_report.stat().st_size > 0
