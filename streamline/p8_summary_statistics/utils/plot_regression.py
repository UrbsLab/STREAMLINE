from __future__ import annotations

import os
from typing import Dict, List, Tuple

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

    s_res_trains = []      # training residual
    s_res_tests = []       # testing residual
    s_y_train_preds = []   # training prediction
    s_y_test_preds = []    # testing prediction
    s_y_trains = []        # training label
    s_y_tests = []         # testing label

    m_trains = []          # slope of training plot
    b_trains = []          # intercept of training plot
    m_tests = []           # slope of testing plot
    b_tests = []           # intercept of testing plot

    eval_plots_dir = os.path.join(full_path, "model_evaluation", "evalPlots")
    os.makedirs(eval_plots_dir, exist_ok=True)

    for algorithm in algorithms:
        s_res_train = []
        s_res_test = []
        s_y_train_pred = []
        s_y_test_pred = []
        s_y_train = []
        s_y_test = []

        for cv_count in range(cv_partitions):
            result_file = (
                f"{full_path}/model_evaluation/pickled_metrics/"
                f"{abbrev[algorithm]}_CV_{cv_count}_residuals.pickle"
            )
            with open(result_file, "rb") as f:
                results = pickle.load(f)  # type: ignore[name-defined]

            res_train = results[0]
            res_test = results[1]
            y_train_pred = results[2]
            y_test_pred = results[3]
            y_train = results[4]
            y_test = results[5]

            s_res_train = np.stack([res_train], axis=0)
            s_res_test = np.stack([res_test], axis=0)
            s_y_train_pred = np.stack([y_train_pred], axis=0)
            s_y_test_pred = np.stack([y_test_pred], axis=0)
            s_y_train = np.stack([y_train], axis=0)
            s_y_test = np.stack([y_test], axis=0)

        s_res_train = s_res_train[0]
        s_res_test = s_res_test[0]
        s_y_train_pred = s_y_train_pred[0]
        s_y_test_pred = s_y_test_pred[0]
        s_y_train = s_y_train[0]
        s_y_test = s_y_test[0]

        s_res_trains.append(s_res_train)
        s_res_tests.append(s_res_test)
        s_y_train_preds.append(y_train_pred)
        s_y_test_preds.append(y_test_pred)
        s_y_trains.append(y_train)
        s_y_tests.append(s_y_test)

        m_1, b_1 = np.polyfit(s_y_train_pred, s_y_train, 1)
        m_2, b_2 = np.polyfit(s_y_test_pred, s_y_test, 1)
        m_trains.append(m_1)
        m_tests.append(m_2)
        b_trains.append(b_1)
        b_tests.append(b_2)

    # Build train / test residual dataframes
    train_df_parts = []
    for i, alg in enumerate(algorithms):
        df = pd.DataFrame(
            [
                s_res_trains[i],
                [alg] * len(s_res_trains[i]),
                ["Training"] * len(s_res_trains[i]),
            ]
        ).transpose()
        df.columns = ["Residual", "Algorithm", "Type"]
        train_df_parts.append(df)
    train_df = pd.concat(train_df_parts).reset_index(drop=True)

    test_df_parts = []
    for i, alg in enumerate(algorithms):
        df = pd.DataFrame(
            [
                s_res_tests[i],
                [alg] * len(s_res_tests[i]),
                ["Testing"] * len(s_res_tests[i]),
            ]
        ).transpose()
        df.columns = ["Residual", "Algorithm", "Type"]
        test_df_parts.append(df)
    test_df = pd.concat(test_df_parts).reset_index(drop=True)

    train_df.to_csv(os.path.join(full_path, "model_evaluation", "residual_train.csv"))
    test_df.to_csv(os.path.join(full_path, "model_evaluation", "residual_test.csv"))

    # --- Fig 2: residual vs predicted + violin distributions ---
    fig_2, axes_2 = plt.subplots(2, 2, sharey="all", figsize=[20, 15])

    for i, alg in enumerate(algorithms):
        axes_2[0, 0].scatter(
            s_y_train_preds[i],
            s_res_trains[i],
            alpha=0.4,
            c=colors[alg],
            label=alg,
        )
        axes_2[1, 0].scatter(
            s_y_test_preds[i],
            s_res_tests[i],
            alpha=0.4,
            c=colors[alg],
        )

    for ax in [axes_2[0, 0], axes_2[0, 1], axes_2[1, 0], axes_2[1, 1]]:
        ax.axhline(y=0, color="black", linestyle="-")

    sns.violinplot(
        x="Algorithm",
        y="Residual",
        data=train_df,
        color="b",
        ax=axes_2[0, 1],
    )
    sns.violinplot(
        x="Algorithm",
        y="Residual",
        data=test_df,
        color="r",
        ax=axes_2[1, 1],
    )

    axes_2[0, 0].title.set_text("Residual vs Predicted Outcome (Training)")
    axes_2[1, 0].title.set_text("Residual vs Predicted Outcome (Testing)")
    axes_2[0, 1].title.set_text("Residual Distribution (Training)")
    axes_2[1, 1].title.set_text("Residual Distribution (Testing)")
    axes_2[0, 0].set_ylabel("Residual")
    axes_2[1, 0].set_ylabel("Residual")
    axes_2[1, 0].set_xlabel("Predicted Outcome")
    fig_2.legend(loc="upper right")
    fig_2.savefig(
        os.path.join(
            eval_plots_dir, "residual_distrib_all_algorithms.png"
        )
    )
    if show_plots:
        plt.show()
    else:
        plt.close("all")

    # --- Fig 3: actual vs predicted with fitted lines ---
    fig_3, axes_3 = plt.subplots(1, 2, sharey="all", figsize=[20, 10])
    for i, alg in enumerate(algorithms):
        axes_3[0].scatter(
            s_y_train_preds[i], s_y_trains[i], alpha=0.3, c=colors[alg]
        )
        axes_3[1].scatter(
            s_y_test_preds[i], s_y_tests[i], alpha=0.3, c=colors[alg]
        )
        axes_3[0].plot(
            s_y_train_preds[i],
            m_trains[i] * s_y_train_preds[i] + b_trains[i],
            color=colors[alg],
            label=alg,
        )
        axes_3[1].plot(
            s_y_test_preds[i],
            m_tests[i] * s_y_test_preds[i] + b_tests[i],
            color=colors[alg],
        )

    axes_3[0].title.set_text("Actual Outcome vs. Predicted Outcome (Train)")
    axes_3[1].title.set_text("Actual Outcome vs. Predicted Outcome (Test)")
    axes_3[0].set_ylabel("Actual Outcome")
    axes_3[0].set_xlabel("Predicted Outcome")
    axes_3[1].set_xlabel("Predicted Outcome")
    fig_3.legend(loc="upper right")
    fig_3.savefig(
        os.path.join(eval_plots_dir, "actual_vs_predict_all_algorithms.png")
    )
    if show_plots:
        plt.show()
    else:
        plt.close("all")

    # --- Fig 4: probability plot of train residuals ---
    fig_4, ax4 = plt.subplots(1, 1, figsize=(10, 10))
    for i, alg in enumerate(algorithms):
        stats.probplot(
            s_res_trains[i],
            dist=stats.norm,
            sparams=(2, 3),
            plot=plt,
            fit=False,
        )
        line = ax4.get_lines()[i]
        line.set_markerfacecolor(colors[alg])
        line.set_alpha(0.5)
        line.set_color(colors[alg])
        line.set_label(alg)

    ax4.title.set_text("Probability Plot of Training Residual")
    ax4.set_xlabel("Theoretical Quantiles")
    ax4.set_ylabel("Ordered Residual")
    ax4.legend(loc="upper right")
    fig_4.savefig(
        os.path.join(
            eval_plots_dir, "probability_train_residual_all_algorithms.png"
        )
    )
    if show_plots:
        plt.show()
    else:
        plt.close("all")

    # --- Fig 5: probability plot of test residuals ---
    fig_5, ax5 = plt.subplots(1, 1, figsize=(10, 10))
    for i, alg in enumerate(algorithms):
        stats.probplot(
            s_res_tests[i],
            dist=stats.norm,
            sparams=(2, 3),
            plot=plt,
            fit=False,
        )
        line = ax5.get_lines()[i]
        line.set_markerfacecolor(colors[alg])
        line.set_alpha(0.5)
        line.set_color(colors[alg])
        line.set_label(alg)

    ax5.title.set_text("Probability Plot of Testing Residual")
    ax5.set_xlabel("Theoretical Quantiles")
    ax5.set_ylabel("Ordered Residual")
    ax5.legend(loc="upper right")
    fig_5.savefig(
        os.path.join(
            eval_plots_dir, "probability_test_residual_all_algorithms.png"
        )
    )
    if show_plots:
        plt.show()
    else:
        plt.close("all")
