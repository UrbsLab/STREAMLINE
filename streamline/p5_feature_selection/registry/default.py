from __future__ import annotations
import os
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
from statistics import median
from streamline.p5_feature_selection.utils.fi_resolver import resolve_algorithms


class DefaultFeatureSelector:
    """
    Implements Phase-5 selection:
      • load P4 scores (per-CV, per-alg) from CSV
      • median-aggregate, plot top-N (optional)
      • union of informative (score>0) per CV across algs
      • cap via round-robin of ranked lists to max_features_to_keep
      • write filtered CV CSVs
    """
    id = "default"
    model_name = "Default Feature Selection"
    small_name = "Default"
    path_name = "default"

    def __init__(self, *, export_scores: bool = True, top_features: int = 20, show_plots: bool = False):
        self.export_scores = bool(export_scores)
        self.top_features = int(top_features)
        self.show_plots = bool(show_plots)

    # ---- public API expected by the job ----
    def select(
        self,
        *,
        dataset_dir: str,
        dataset_name: str,
        n_splits: int,
        algorithms: List[str],
        outcome_label: str,
        instance_label: str | None,
        max_features_to_keep: int,
        filter_poor_features: bool,
        overwrite_cv: bool,
    ):
        algs = resolve_algorithms(dataset_dir, algorithms)
        fs_root = os.path.join(dataset_dir, "feature_importance")
        out_root = os.path.join(dataset_dir, "feature_selection")
        os.makedirs(out_root, exist_ok=True)

        selected_feature_lists: Dict[str, List[List[str]]] = {}
        meta_feature_ranks: Dict[str, List[List[str]]] = {}
        for alg in algs:
            keep_per_cv, ranks_per_cv, med_table = self._collect_scores(fs_root, alg, n_splits)
            selected_feature_lists[alg] = keep_per_cv
            meta_feature_ranks[alg] = ranks_per_cv
            if self.export_scores and med_table is not None:
                logging.info("Plotting Feature Importance Scores for %s...", alg)
                logging.info("%s", med_table.head(10).to_string(index=False))
                self._plot_top_medians(fs_root, alg, med_table)
                logging.info("Saved Feature Importance Plots at")
                logging.info("%s", os.path.join(fs_root, alg, "TopAverageScores.png"))

        if not filter_poor_features or not algs:
            return  # nothing else to do

        logging.info("Applying collective feature selection...")
        cv_selected_list, informative_counts, uninformative_counts = self._select_union_cap(
            selected_feature_lists, max_features_to_keep, meta_feature_ranks, algs, n_splits
        )
        self._write_info_counts(out_root, informative_counts, uninformative_counts)
        self._write_filtered_cv(dataset_dir, dataset_name, n_splits, outcome_label, instance_label, cv_selected_list, overwrite_cv)

    # ---- internals ----
    def _score_csv(self, root: str, alg: str, cv: int) -> str:
        return os.path.join(root, alg, f"{alg}_scores_cv_{cv}.csv")

    def _collect_scores(self, root: str, alg: str, n_splits: int):
        keep_per_cv: List[List[str]] = []
        ranks_per_cv: List[List[str]] = []
        per_feature: Dict[str, List[float]] = {}

        for i in range(n_splits):
            path = self._score_csv(root, alg, i)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing Phase 4 scores: {path}")
            df = pd.read_csv(path).dropna(subset=["feature"]).copy()
            if "score" not in df.columns:
                df["score"] = 0.0
            df["score"] = df["score"].astype(float)
            df_sorted = df.sort_values("score", ascending=False)
            ranks_per_cv.append(df_sorted["feature"].tolist())
            keep_per_cv.append(df_sorted.loc[df_sorted["score"] > 0.0, "feature"].tolist())
            for _, r in df.iterrows():
                per_feature.setdefault(str(r["feature"]), []).append(float(r["score"]))

        med_table = None
        if self.export_scores:
            med_table = (
                pd.DataFrame([(f, median(v)) for f, v in per_feature.items()], columns=["Feature", "Importance"])
                .sort_values("Importance", ascending=False)
            )
        return keep_per_cv, ranks_per_cv, med_table

    def _plot_top_medians(self, root: str, alg: str, table: pd.DataFrame):
        out_dir = os.path.join(root, alg); os.makedirs(out_dir, exist_ok=True)
        ns = table.head(self.top_features)
        plt.figure(figsize=(7, max(4, 0.35 * len(ns))))
        plt.barh(ns["Feature"][::-1], ns["Importance"][::-1])
        title = {"mutualinformation": "Mutual Information",
                 "multisurf": "MultiSURF",
                 "multisurfstar": "MultiSURF*",
                 "multiswrfdb": "MultiSWRFDB",
                 "multiswrfdbstar": "MultiSWRFDB*"}.get(alg, alg)
        plt.xlabel("Median Score"); plt.title(f"Sorted Median {title} Scores")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "TopAverageScores.png"), bbox_inches="tight")
        if self.show_plots: plt.show()
        plt.close()

    def _select_union_cap(
        self,
        selected_feature_lists: Dict[str, List[List[str]]],
        max_features_to_keep: int,
        meta_feature_ranks: Dict[str, List[List[str]]],
        alg_order: List[str],
        n_splits: int,
    ):
        cv_selected_list: List[List[str]] = []
        informative_counts: List[int] = []
        uninformative_counts: List[int] = []
        try:
            total_features = len(meta_feature_ranks[alg_order[0]][0])
        except Exception:
            total_features = 0

        if len(alg_order) > 1:
            for i in range(n_splits):
                union = set()
                for alg in alg_order: union.update(selected_feature_lists[alg][i])
                union_list = list(union)
                informative_counts.append(len(union_list))
                uninformative_counts.append(max(0, total_features - len(union_list)))
                if len(union_list) > max_features_to_keep:
                    new_list, seen, k = [], set(), 0
                    while len(new_list) < max_features_to_keep:
                        progressed = False
                        for alg in alg_order:
                            rl = meta_feature_ranks[alg][i]
                            if k < len(rl):
                                cand = rl[k]
                                if cand not in seen and cand in union:
                                    new_list.append(cand); seen.add(cand); progressed = True
                                    if len(new_list) >= max_features_to_keep: break
                        if not progressed: break
                        k += 1
                    union_list = new_list
                union_list.sort()
                cv_selected_list.append(union_list)
        else:
            single = alg_order[0]
            for i in range(n_splits):
                base = list(selected_feature_lists[single][i])
                informative_counts.append(len(base))
                uninformative_counts.append(max(0, total_features - len(base)))
                if len(base) > max_features_to_keep:
                    rl = meta_feature_ranks[single][i]
                    new_list, k = [], 0
                    while len(new_list) < max_features_to_keep and k < len(rl):
                        if rl[k] in base: new_list.append(rl[k])
                        k += 1
                    base = new_list
                base.sort(); cv_selected_list.append(base)

        return cv_selected_list, informative_counts, uninformative_counts

    def _write_info_counts(self, out_root: str, inf: List[int], uninf: List[int]):
        pd.DataFrame({"Informative": inf, "Uninformative": uninf}).to_csv(
            os.path.join(out_root, "InformativeFeatureSummary.csv"), index_label="CV_Partition"
        )

    def _write_filtered_cv(
        self,
        dataset_dir: str,
        dataset_name: str,
        n_splits: int,
        outcome_label: str,
        instance_label: "str | None",
        cv_selected_list: List[List[str]],
        overwrite_cv: bool,
    ):
        cv_dir = os.path.join(dataset_dir, "CVDatasets")
        for i in range(n_splits):
            tr = os.path.join(cv_dir, f"{dataset_name}_CV_{i}_Train.csv")
            te = os.path.join(cv_dir, f"{dataset_name}_CV_{i}_Test.csv")
            if not (os.path.exists(tr) and os.path.exists(te)):
                raise FileNotFoundError(f"Missing CV files for i={i}: {tr} / {te}")
            df_tr = pd.read_csv(tr, na_values="NA"); df_te = pd.read_csv(te, na_values="NA")
            labels = [outcome_label] + ([instance_label] if (instance_label and instance_label in df_tr.columns) else [])
            feat_keep = [f for f in cv_selected_list[i] if f in df_tr.columns]
            td_train = df_tr.loc[:, labels + feat_keep]; td_test = df_te.loc[:, labels + feat_keep]
            if overwrite_cv:
                os.remove(tr); os.remove(te)
            else:
                os.rename(tr, os.path.join(cv_dir, f"{dataset_name}_CVPre_{i}_Train.csv"))
                os.rename(te, os.path.join(cv_dir, f"{dataset_name}_CVPre_{i}_Test.csv"))
            td_train.to_csv(os.path.join(cv_dir, f"{dataset_name}_CV_{i}_Train.csv"), index=False)
            td_test.to_csv(os.path.join(cv_dir, f"{dataset_name}_CV_{i}_Test.csv"), index=False)
