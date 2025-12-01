from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

import seaborn as sns

sns.set_theme()


def plot_fi_boxplots(
    full_path: str,
    algorithms: List[str],
    feature_headers: List[str],
    fi_df_list: List[pd.DataFrame],
    fi_med_list: List[List[float]],
    metric_ranking: str = "median",
    show_plots: bool = False,
):
    """
    Boxplots of feature importance per algorithm.
    """
    out_dir = Path(full_path) / "model_evaluation" / "feature_importance"
    out_dir.mkdir(parents=True, exist_ok=True)

    for alg_idx, algorithm in enumerate(algorithms):
        score_dict = {}
        for idx, med_score in enumerate(fi_med_list[alg_idx]):
            score_dict[feature_headers[idx]] = med_score

        score_dict_features = sorted(score_dict, key=lambda x: score_dict[x], reverse=True)
        if len(feature_headers) > 0:
            top_n = min(len(score_dict_features), len(feature_headers))
        else:
            top_n = len(score_dict_features)
        features_to_viz = score_dict_features[:top_n]

        df = fi_df_list[alg_idx]
        viz_df = df[features_to_viz]

        plt.figure(figsize=(15, 4))
        viz_df.boxplot(rot=90)
        plt.title(algorithm)
        plt.ylabel("Feature Importance")
        if metric_ranking == "mean":
            plt.xlabel("Features (Mean Ranking)")
        elif metric_ranking == "median":
            plt.xlabel("Features (Median Ranking)")
        else:
            plt.xlabel("Features")

        plt.xticks(np.arange(1, len(features_to_viz) + 1), features_to_viz, rotation="vertical")
        out = out_dir / f"{algorithm}_boxplot.png"
        plt.savefig(out.as_posix(), bbox_inches="tight")
        if show_plots:
            plt.show()
        else:
            plt.close("all")


def plot_fi_histogram(
    full_path: str,
    algorithms: List[str],
    fi_med_list: List[List[float]],
    metric_ranking: str = "median",
    show_plots: bool = False,
):
    """
    Histogram of median/mean FI scores per algorithm.
    """
    out_dir = Path(full_path) / "model_evaluation" / "feature_importance"
    out_dir.mkdir(parents=True, exist_ok=True)

    for alg_idx, algorithm in enumerate(algorithms):
        med_scores = fi_med_list[alg_idx]
        plt.hist(med_scores, bins=100)
        if metric_ranking == "mean":
            plt.xlabel("Mean Feature Importance")
        elif metric_ranking == "median":
            plt.xlabel("Median Feature Importance")
        else:
            plt.xlabel("Feature Importance")
        plt.ylabel("Frequency")
        plt.title(str(algorithm))
        plt.xticks(rotation="vertical")
        out = out_dir / f"{algorithm}_histogram.png"
        plt.savefig(out.as_posix(), bbox_inches="tight")
        if show_plots:
            plt.show()
        else:
            plt.close("all")


def plot_composite_fi(
    full_path: str,
    algorithms: List[str],
    colors: Dict[str, Any],
    fi_list: List[List[float]],
    all_feature_list_to_viz: List[str],
    fig_name: str,
    y_label_text: str,
    metric_ranking: str,
    metric_weighting: str,
    metric_weight_label: str,
    show_plots: bool = False,
):
    """
    Composite stacked-bar FI plot across algorithms.
    """
    # sort algorithms + lists together to keep consistent ordering
    alg_colors = [colors[k] for k in algorithms]
    algorithms, alg_colors, fi_list = (list(t) for t in zip(*sorted(
        zip(algorithms, alg_colors, fi_list),
        reverse=True,
    )))

    rc("font", weight="bold", size=16)

    r = all_feature_list_to_viz
    bar_width = 0.75
    plt.figure(figsize=(24, 12))

    # base bar
    p1 = plt.bar(r, fi_list[0], color=alg_colors[0], edgecolor="white", width=bar_width)

    bottoms = []
    bottom = None
    for i in range(len(algorithms) - 1):
        for j in range(i + 1):
            if j == 0:
                bottom = np.array(fi_list[0]).astype("float64")
            else:
                bottom += np.array(fi_list[j]).astype("float64")
        bottoms.append(bottom)
    if not isinstance(bottoms, list):
        bottoms = bottoms.tolist()

    if len(algorithms) > 1:
        ps = [p1[0]]
        for i in range(len(algorithms) - 1):
            p = plt.bar(
                r,
                fi_list[i + 1],
                bottom=bottoms[i],
                color=alg_colors[i + 1],
                edgecolor="white",
                width=bar_width,
            )
            ps.append(p[0])
        lines = tuple(ps)
    else:
        lines = (p1[0],)

    plt.xticks(np.arange(len(all_feature_list_to_viz)), all_feature_list_to_viz, rotation="vertical")
    plt.xlabel(
        "Features (ranked by sum of "
        + metric_ranking
        + " feature importance: weighted by "
        + metric_weighting
        + " model "
        + metric_weight_label.lower()
        + ")",
        fontsize=20,
    )
    plt.ylabel(y_label_text, fontsize=20)
    plt.legend(lines[::-1], algorithms[::-1], loc="upper left", bbox_to_anchor=(1.01, 1))

    out = Path(full_path) / "model_evaluation" / "feature_importance" / f"Compare_FI_{fig_name}.png"
    plt.savefig(out.as_posix(), bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close("all")
