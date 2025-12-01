from __future__ import annotations

from statistics import mean, median
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def prep_fi(
    full_path: str,
    algorithms: List[str],
    abbrev: Dict[str, str],
    metric_dict: Dict[str, Dict[str, List[float]]],
    metric_ranking: str,
    metric_weighting: str,
    metric_weight_name: str,
) -> Tuple[
    List[pd.DataFrame],
    List[List[float]],
    List[List[float]],
    List[float],
    List[str],
    List[str],
    List[int],
]:
    """
    Load and organize feature-importance data from per-algorithm FI CSVs.

    Parameters
    ----------
    full_path
        Dataset root (<output>/<experiment>/<dataset>).
    algorithms
        List of algorithm display names (StatisticsPhaseJob.algorithms).
    abbrev
        Map from algorithm display name -> small_name / abbrev used in filenames.
    metric_dict
        Per-algorithm metric lists, as produced by primary_stats_*.
    metric_ranking
        'mean' or 'median' – how to summarize FI across CVs.
    metric_weighting
        'mean' or 'median' – how to summarize model performance for weighting FI.
    metric_weight_name
        Human-readable metric key in metric_dict to weight FI with
        (e.g. 'Balanced Accuracy', 'Explained Variance').

    Returns
    -------
    fi_df_list
        List of DataFrames (one per algorithm) with FI per CV.
    fi_med_list
        List of per-algorithm lists of median/mean FI per feature.
    fi_med_norm_list
        Same, but normalized to [0, 1] within each algorithm.
    med_metric_list
        Per-algorithm scalar metric used for weighting FI.
    all_feature_list
        List of feature names as they appear in FI CSV columns.
    non_zero_union_features
        Features that have non-zero FI in at least one algorithm.
    non_zero_union_indexes
        Indexes in all_feature_list for non_zero_union_features.
    """
    # algorithm feature importance dataframe list (used to generate FI boxplot for each algorithm)
    fi_df_list: List[pd.DataFrame] = []
    # algorithm feature importance medians list (used to generate composite FI barplots)
    fi_med_list: List[List[float]] = []
    # algorithm focus metric medians list (used in weighted FI viz)
    med_metric_list: List[float] = []
    # list of feature names as they appear in FI reports
    all_feature_list: List[str] = []

    # --- Load FI CSVs and compute per-feature medians/means ---
    for idx, algorithm in enumerate(algorithms):
        fi_path = (
            f"{full_path}/model_evaluation/feature_importance/"
            f"{abbrev[algorithm]}_FI.csv"
        )
        temp_df = pd.read_csv(fi_path)  # CV FI scores for all original features
        if idx == 0:
            all_feature_list = temp_df.columns.tolist()

        fi_df_list.append(temp_df)

        if metric_ranking == "mean":
            fi_med_list.append(temp_df.mean().tolist())
        elif metric_ranking == "median":
            fi_med_list.append(temp_df.median().tolist())
        else:
            raise ValueError("metric_ranking must be 'mean' or 'median'")

        # Get relevant performance metric info (for weighting)
        metric_vals = metric_dict[algorithm][metric_weight_name]
        if metric_weighting == "mean":
            med_ba = mean(metric_vals)
        elif metric_weighting == "median":
            med_ba = median(metric_vals)
        else:
            raise ValueError("metric_weighting must be 'mean' or 'median'")
        med_metric_list.append(med_ba)

    # --- Normalize FI (within each algorithm) ---
    fi_med_norm_list: List[List[float]] = []
    for each in fi_med_list:  # each algorithm
        norm_list: List[float] = []
        max_val = max(each) if each else 0.0
        for val in each:
            if val <= 0 or max_val <= 0:
                norm_list.append(0.0)
            else:
                norm_list.append(val / max_val)
        fi_med_norm_list.append(norm_list)

    # --- Identify union of non-zero features across algorithms ---
    alg_non_zero_fi_list: List[List[str]] = []
    for each in fi_med_list:  # each algorithm
        temp_non_zero_list: List[str] = []
        for i, val in enumerate(each):
            if val > 0.0:
                temp_non_zero_list.append(all_feature_list[i])
        alg_non_zero_fi_list.append(temp_non_zero_list)

    if alg_non_zero_fi_list:
        non_zero_union_features = list(alg_non_zero_fi_list[0])
        for j in range(1, len(algorithms)):
            non_zero_union_features = list(
                set(non_zero_union_features) | set(alg_non_zero_fi_list[j])
            )
    else:
        non_zero_union_features = []

    non_zero_union_indexes: List[int] = [
        all_feature_list.index(f) for f in non_zero_union_features
    ]

    return (
        fi_df_list,
        fi_med_list,
        fi_med_norm_list,
        med_metric_list,
        all_feature_list,
        non_zero_union_features,
        non_zero_union_indexes,
    )


def select_for_composite_viz(
    non_zero_union_features: List[str],
    non_zero_union_indexes: List[int],
    ave_metric_list: List[float],
    fi_ave_norm_list: List[List[float]],
    algorithms: List[str],
    top_features: int,
) -> List[str]:
    """
    Determine which features to visualize in composite FI plots.

    Score for feature f is:
        sum_over_algorithms( normalized_FI[alg, f] * performance_weight[alg] )

    If there are fewer than `top_features` features, all non-zero features are used.
    """
    score_sum_dict: Dict[str, float] = {}
    for i, feat_name in enumerate(non_zero_union_features):
        idx = non_zero_union_indexes[i]
        for j in range(len(algorithms)):
            score = fi_ave_norm_list[j][idx]
            weight = ave_metric_list[j]
            score *= weight
            if feat_name not in score_sum_dict:
                score_sum_dict[feat_name] = score
            else:
                score_sum_dict[feat_name] += score

    # Sort by decreasing score
    score_sum_dict_features = sorted(
        score_sum_dict, key=lambda x: score_sum_dict[x], reverse=True
    )
    if len(non_zero_union_features) > top_features:
        features_to_viz = score_sum_dict_features[:top_features]
    else:
        features_to_viz = score_sum_dict_features

    return features_to_viz


def get_fi_to_viz_sorted(
    features_to_viz: List[str],
    all_feature_list: List[str],
    fi_med_norm_list: List[List[float]],
    algorithms: List[str],
) -> Tuple[List[List[float]], List[str]]:
    """
    Given selected feature names, pull their normalized FI values in the same
    order for all algorithms, ready for stacked-bar plotting.
    """
    # Indexes of selected features in the full list
    feature_index_to_viz: List[int] = [
        all_feature_list.index(f) for f in features_to_viz
    ]

    # Build list-of-lists: per-algorithm FI values for the selected features
    top_fi_med_norm_list: List[List[float]] = []
    for i in range(len(algorithms)):
        temp_list: List[float] = []
        for j in feature_index_to_viz:
            temp_list.append(fi_med_norm_list[i][j])
        top_fi_med_norm_list.append(temp_list)

    all_feature_list_to_viz = features_to_viz
    return top_fi_med_norm_list, all_feature_list_to_viz


def frac_fi(top_fi_med_norm_list: List[List[float]]) -> List[List[float]]:
    """
    Fractionate FI scores so that they sum to 1 over all features for a given algorithm.
    Useful if you want to equalize 'total bar area' per algorithm.
    """
    frac_lists: List[List[float]] = []
    for each in top_fi_med_norm_list:  # each algorithm
        total = sum(each)
        if total == 0:
            frac_lists.append([0.0 for _ in each])
        else:
            frac_lists.append([val / total for val in each])
    return frac_lists


def weight_fi(
    med_metric_list: List[float],
    top_fi_med_norm_list: List[List[float]],
) -> Tuple[List[List[float]], List[float]]:
    """
    Weight normalized FI scores by algorithm performance.

    - Any med_metric <= 0.5 is treated as 0 (no better than chance).
    - Remaining metrics are linearly scaled from [0.5, 1] -> [0, 1].
    """
    # Prepare weights
    metrics = list(med_metric_list)  # copy so we don't mutate caller's list
    weights: List[float] = []

    for i, v in enumerate(metrics):
        if v <= 0.5:
            metrics[i] = 0.0

    for v in metrics:
        if v == 0:
            weights.append(0.0)
        else:
            weights.append((v - 0.5) / 0.5)

    # Weight normalized FI
    weighted_lists: List[List[float]] = []
    for i, fi_vals in enumerate(top_fi_med_norm_list):
        w = weights[i] if i < len(weights) else 0.0
        weighted_lists.append(np.multiply(w, fi_vals).tolist())

    return weighted_lists, weights


def weight_frac_fi(
    frac_lists: List[List[float]],
    weights: List[float],
) -> List[List[float]]:
    """
    Weight normalized and fractionated feature importances by performance weights.
    """
    weighted_frac_lists: List[List[float]] = []
    for i, frac_vals in enumerate(frac_lists):
        w = weights[i] if i < len(weights) else 0.0
        weighted_frac_lists.append(np.multiply(w, frac_vals).tolist())
    return weighted_frac_lists
