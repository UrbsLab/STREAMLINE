import os
import io
import json
import glob
import csv
import time
import shutil
import pathlib
import builtins

import pytest

from streamline.p5_feature_selection.p5_runner import P5Runner


def _make_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _seed_experiment(tmp_path, n_splits=2, algs=("mutualinformation", "multisurf")):
    """
    Create a minimal experiment tree:

    out/Exp/D1/
      CVDatasets/D1_CV_{i}_{Train,Test}.csv
      feature_importance/<alg>/<alg>_scores_cv_{i}.csv
    """
    out = tmp_path / "out"
    exp = out / "Exp"
    d1 = exp / "D1"
    (d1 / "CVDatasets").mkdir(parents=True)
    # minimal train/test with features
    for i in range(n_splits):
        _make_csv(
            d1 / "CVDatasets" / f"D1_CV_{i}_Train.csv",
            [
                ["Class", "f1", "f2", "noise0"],
                [1, 10, 0.1, 7],
                [0, 5,  0.2, 9],
            ],
        )
        _make_csv(
            d1 / "CVDatasets" / f"D1_CV_{i}_Test.csv",
            [
                ["Class", "f1", "f2", "noise0"],
                [0, 8, 0.05, 3],
            ],
        )

    # phase-4 score CSVs
    for alg in algs:
        for i in range(n_splits):
            _make_csv(
                d1 / "feature_importance" / alg / f"{alg}_scores_cv_{i}.csv",
                [
                    ["feature", "score"],
                    ["f1", 0.9],
                    ["f2", 0.4],
                    ["noise0", 0.0],  # should be filtered by > 0 rule
                ],
            )

    # folders needed by runner in some modes
    (exp / "jobs").mkdir(parents=True, exist_ok=True)
    (exp / "logs").mkdir(parents=True, exist_ok=True)
    return out, exp, d1


def test_p5_runner_auto_serial_creates_outputs(tmp_path):
    out, exp, d1 = _seed_experiment(tmp_path, n_splits=2)

    runner = P5Runner(
        output_path=str(out),
        experiment_name="Exp",
        algorithms="auto",          # <- discover from feature_importance/*
        n_splits=2,
        class_label="Class",
        instance_label=None,
        max_features_to_keep=5,
        filter_poor_features=True,
        overwrite_cv=False,         # exercise rename path
        selector_id="default",
        selector_params=None,
        export_scores=True,
        top_features=5,
        show_plots=False,
        run_cluster="Serial",
    )
    runner.run()

    # 1) InformativeFeatureSummary.csv
    fs_summary = d1 / "feature_selection" / "InformativeFeatureSummary.csv"
    assert fs_summary.exists(), "Phase 5 summary CSV not created"

    # 2) Plots per algorithm
    for alg in ("mutualinformation", "multisurf"):
        plot = d1 / "feature_importance" / alg / "TopAverageScores.png"
        assert plot.exists(), f"Missing TopAverageScores.png for {alg}"

    # 3) Filtered CV CSVs should exist with only class + informative features (f1, f2)
    for i in range(2):
        tr = d1 / "CVDatasets" / f"D1_CV_{i}_Train.csv"
        te = d1 / "CVDatasets" / f"D1_CV_{i}_Test.csv"
        assert tr.exists() and te.exists()
        # When overwrite_cv=False, originals are renamed to *_CVPre_*
        pre_tr = d1 / "CVDatasets" / f"D1_CVPre_{i}_Train.csv"
        pre_te = d1 / "CVDatasets" / f"D1_CVPre_{i}_Test.csv"
        assert pre_tr.exists() and pre_te.exists()

        import pandas as pd
        df_tr = pd.read_csv(tr)
        df_te = pd.read_csv(te)
        for df in (df_tr, df_te):
            cols = list(df.columns)
            assert cols[0] == "Class"
            # noise0 should be removed because score==0
            assert "f1" in cols and "f2" in cols and "noise0" not in cols


@pytest.mark.skipif(
    pytest.importorskip("dask") is None or pytest.importorskip("dask.distributed") is None,
    reason="dask.distributed not available",
)
def test_p5_runner_auto_local_parallel(tmp_path):
    out, exp, d1 = _seed_experiment(tmp_path, n_splits=2)

    runner = P5Runner(
        output_path=str(out),
        experiment_name="Exp",
        algorithms="auto",
        n_splits=2,
        run_cluster="Local",  # <- parallel via LocalCluster
        class_label="Class",
        max_features_to_keep=5,
        filter_poor_features=True,
        overwrite_cv=True,   # exercise overwrite branch this time
        selector_id="default",
        export_scores=False, # no plots this time
    )
    runner.run()

    # overwrite_cv=True means no *_CVPre_* files; final CVs exist
    for i in range(2):
        tr = d1 / "CVDatasets" / f"D1_CV_{i}_Train.csv"
        te = d1 / "CVDatasets" / f"D1_CV_{i}_Test.csv"
        assert tr.exists() and te.exists()
        assert not (d1 / "CVDatasets" / f"D1_CVPre_{i}_Train.csv").exists()
        assert not (d1 / "CVDatasets" / f"D1_CVPre_{i}_Test.csv").exists()


def test_p5_runner_bash_script_includes_discovered_algorithms(tmp_path, monkeypatch):
    out, exp, d1 = _seed_experiment(tmp_path, n_splits=2)

    calls = []
    def _fake_system(cmd):
        # Capture the command so we can ensure we didn't actually run sbatch/bsub
        calls.append(cmd)
        return 0

    monkeypatch.setattr(os, "system", _fake_system)

    runner = P5Runner(
        output_path=str(out),
        experiment_name="Exp",
        algorithms="auto",
        n_splits=2,
        run_cluster="BashSLURM",   # <- script generation path
        queue="defq",
        reserved_memory=2,
        class_label="Class",
        selector_id="default",
        export_scores=False,
    )
    runner.run()

    # a script should be created under exp/jobs with algorithms expanded, not "auto"
    jobs_dir = exp / "jobs"
    scripts = sorted(glob.glob(str(jobs_dir / "P5_*_run.sh")))
    assert scripts, "No SLURM job script was generated"

    # The script content should include the concrete algorithms list
    script_path = scripts[0]
    with open(script_path, "r") as sh:
        content = sh.read()
    assert "--algorithms mutualinformation,multisurf" in content.replace("  ", " "), \
        "Discovered algorithms were not embedded into the bash script"

    # And os.system should have been called with 'sbatch <script>'
    assert calls, "No os.system call captured for submission"
    assert calls[0].startswith("sbatch "), "Expected sbatch submission command"
