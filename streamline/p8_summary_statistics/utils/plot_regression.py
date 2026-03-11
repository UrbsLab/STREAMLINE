from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme()


def residuals_regression(
    full_path: str,
    algorithms: List[str],
    abbrev: Dict[str, str],
    cv_partitions: int,
    colors: Dict[str, Tuple[float, float, float]],
    show_plots: bool = False,
) -> None:
    """
    Generate residual-related regression plots across all algorithms.

    This is factored out from StatisticsPhaseJob.residuals_regression and
    kept IO-compatible: it still reads residual pickles and writes
    PNGs under <full_path>/model_evaluation/evalPlots.
    """

    s_res_tests: List[np.ndarray] = []       # testing residual
    s_y_test_preds: List[np.ndarray] = []    # testing prediction
    s_y_tests: List[np.ndarray] = []         # testing label
    m_tests: List[float] = []                # slope of testing fit line
    b_tests: List[float] = []                # intercept of testing fit line
    valid_algorithms: List[str] = []

    eval_plots_dir = os.path.join(full_path, "model_evaluation", "evalPlots")
    os.makedirs(eval_plots_dir, exist_ok=True)

    for algorithm in algorithms:
        res_test_parts: List[np.ndarray] = []
        y_test_pred_parts: List[np.ndarray] = []
        y_test_parts: List[np.ndarray] = []

        for cv_count in range(cv_partitions):
            cv_result_file = f"{full_path}/model_evaluation/pickled_metrics/" \
                             f"{abbrev[algorithm]}_CV_{cv_count}_residuals.pickle"
            with open(cv_result_file, "rb") as f:
                results = pickle.load(f)

            # Regression residual pickle payload order:
            # 0 train residual, 1 test residual, 2 train pred, 3 test pred, 4 train y, 5 test y
            res_test = np.asarray(results[1], dtype=float).ravel()
            y_test_pred = np.asarray(results[3], dtype=float).ravel()
            y_test = np.asarray(results[5], dtype=float).ravel()

            if res_test.size == 0 or y_test_pred.size == 0 or y_test.size == 0:
                continue
            res_test_parts.append(res_test)
            y_test_pred_parts.append(y_test_pred)
            y_test_parts.append(y_test)

        if not res_test_parts:
            continue

        s_res_test = np.concatenate(res_test_parts)
        s_y_test_pred = np.concatenate(y_test_pred_parts)
        s_y_test = np.concatenate(y_test_parts)

        finite_mask = (
            np.isfinite(s_res_test)
            & np.isfinite(s_y_test_pred)
            & np.isfinite(s_y_test)
        )
        s_res_test = s_res_test[finite_mask]
        s_y_test_pred = s_y_test_pred[finite_mask]
        s_y_test = s_y_test[finite_mask]

        if s_res_test.size < 2:
            continue

        if np.unique(s_y_test_pred).size > 1:
            m_test, b_test = np.polyfit(s_y_test_pred, s_y_test, 1)
        else:
            m_test, b_test = 0.0, float(np.nanmean(s_y_test))

        valid_algorithms.append(algorithm)
        s_res_tests.append(s_res_test)
        s_y_test_preds.append(s_y_test_pred)
        s_y_tests.append(s_y_test)
        m_tests.append(float(m_test))
        b_tests.append(float(b_test))

    if not valid_algorithms:
        return

    # Build testing residual dataframe
    test_df_parts = []
    for i, alg in enumerate(valid_algorithms):
        test_df_parts.append(
            pd.DataFrame(
                {
                    "Residual": s_res_tests[i],
                    "Algorithm": alg,
                    "Type": "Testing",
                }
            )
        )
    test_df = pd.concat(test_df_parts, ignore_index=True)
    test_df["Residual"] = pd.to_numeric(test_df["Residual"], errors="raise")
    # test_df = test_df[np.isfinite(test_df["Residual"])].reset_index(drop=True)

    # Keep only testing outputs for regression evaluation artifacts.
    test_df.to_csv(
        os.path.join(full_path, "model_evaluation", "residual_test.csv"),
        index=False,
    )
    stale_train_csv = os.path.join(full_path, "model_evaluation", "residual_train.csv")
    if os.path.exists(stale_train_csv):
        os.remove(stale_train_csv)
    stale_train_prob_plot = os.path.join(
        eval_plots_dir, "probability_train_residual_all_algorithms.png"
    )
    if os.path.exists(stale_train_prob_plot):
        os.remove(stale_train_prob_plot)

    # --- Fig 1: test residual vs predicted + test violin distribution ---
    fig_1, axes_1 = plt.subplots(1, 2, figsize=[20, 8])
    for i, alg in enumerate(valid_algorithms):
        axes_1[0].scatter(
            s_y_test_preds[i],
            s_res_tests[i],
            alpha=0.35,
            c=colors[alg],
            label=alg,
            s=16,
        )
    axes_1[0].axhline(y=0, color="black", linestyle="-")
    axes_1[0].set_title("Residual vs Predicted Outcome (Testing)")
    axes_1[0].set_ylabel("Residual")
    axes_1[0].set_xlabel("Predicted Outcome")

    sns.violinplot(
        x="Algorithm",
        y="Residual",
        data=test_df,
        order=valid_algorithms,
        color="r",
        ax=axes_1[1],
        inner="quartile",
        cut=0,
        bw_adjust=0.8,
        linewidth=1.0,
    )
    axes_1[1].axhline(y=0, color="black", linestyle="-")
    axes_1[1].set_title("Residual Distribution (Testing)")
    axes_1[1].set_xlabel("Algorithm")
    axes_1[1].set_ylabel("Residual")

    fig_1.legend(loc="upper right")
    fig_1.tight_layout(rect=[0, 0, 0.95, 1])
    fig_1.savefig(
        os.path.join(eval_plots_dir, "residual_distrib_all_algorithms.png")
    )
    if show_plots:
        plt.show()
    else:
        plt.close(fig_1)

    # --- Fig 2: test actual vs predicted with fitted lines ---
    fig_2, ax2 = plt.subplots(1, 1, figsize=[12, 10])
    for i, alg in enumerate(valid_algorithms):
        ax2.scatter(
            s_y_test_preds[i],
            s_y_tests[i],
            alpha=0.3,
            c=colors[alg],
            s=16,
        )
        x_sorted = np.sort(s_y_test_preds[i])
        ax2.plot(
            x_sorted,
            m_tests[i] * x_sorted + b_tests[i],
            color=colors[alg],
            label=alg,
        )
    ax2.set_title("Actual Outcome vs. Predicted Outcome (Test)")
    ax2.set_ylabel("Actual Outcome")
    ax2.set_xlabel("Predicted Outcome")
    fig_2.legend(loc="upper right")
    fig_2.tight_layout()
    fig_2.savefig(
        os.path.join(eval_plots_dir, "actual_vs_predict_all_algorithms.png")
    )
    if show_plots:
        plt.show()
    else:
        plt.close(fig_2)

    # --- Fig 3: probability plot of test residuals ---
    fig_3, ax3 = plt.subplots(1, 1, figsize=(10, 10))
    for i, alg in enumerate(valid_algorithms):
        qq_x, qq_y = stats.probplot(
            s_res_tests[i],
            dist=stats.norm,
            sparams=(2, 3),
            fit=False,
        )
        ax3.scatter(
            qq_x,
            qq_y,
            alpha=0.5,
            c=[colors[alg]],
            s=16,
            label=alg,
        )
    ax3.set_title("Probability Plot of Testing Residual")
    ax3.set_xlabel("Theoretical Quantiles")
    ax3.set_ylabel("Ordered Residual")
    ax3.legend(loc="upper right")
    fig_3.tight_layout()
    fig_3.savefig(
        os.path.join(eval_plots_dir, "probability_test_residual_all_algorithms.png")
    )
    if show_plots:
        plt.show()
    else:
        plt.close(fig_3)
