# streamline/p11_reporting/dashboard_app.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.is_file():
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Failed to read CSV: {path} ({e})")
    return None


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.is_file():
            with path.open("r") as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Failed to read JSON: {path} ({e})")
    return None


def _list_datasets(exp_root: Path) -> List[Path]:
    return [
        p for p in sorted(exp_root.iterdir())
        if p.is_dir()
        and (p / "CVDatasets").is_dir()
        and p.name not in {"DatasetComparisons", "jobs", "jobsCompleted", "logs", "runtime", "reporting"}
    ]


# -------------------------------------------------------------------
# Phase 1 - Exploratory / Data processing views
# -------------------------------------------------------------------

def view_phase1_exploratory(dataset_dir: Path) -> None:
    st.header("Phase 1 - Exploratory / Data Processing")

    exp_dir = dataset_dir / "exploratory"
    if not exp_dir.is_dir():
        st.info("No exploratory directory found for this dataset.")
        return

    # 1) Class distribution
    class_counts_csv = exp_dir / "ClassCounts.csv"
    df_class = _safe_read_csv(class_counts_csv)
    if df_class is not None and not df_class.empty:
        st.subheader("Class distribution")
        # Try some sensible defaults, fall back to generic
        if {"Class", "Count"}.issubset(df_class.columns):
            fig = px.bar(
                df_class,
                x="Class",
                y="Count",
                title="Class counts",
            )
        else:
            melted = df_class.melt(var_name="Category", value_name="Value")
            fig = px.bar(melted, x="Category", y="Value", title="ClassCounts (raw)")
        st.plotly_chart(fig, use_container_width=True)

    # 2) Feature missingness
    miss_csv = exp_dir / "DataMissingness.csv"
    df_miss = _safe_read_csv(miss_csv)
    if df_miss is not None and not df_miss.empty:
        st.subheader("Feature missingness")
        # try to infer feature / percent columns
        col_feature = None
        col_pct = None
        for c in df_miss.columns:
            lc = c.lower()
            if col_feature is None and ("feature" in lc or "variable" in lc or "name" in lc):
                col_feature = c
            if col_pct is None and ("percent" in lc or "pct" in lc):
                col_pct = c
        if col_feature and col_pct:
            df_miss_sorted = df_miss.sort_values(col_pct, ascending=False)
            fig = px.bar(
                df_miss_sorted,
                x=col_feature,
                y=col_pct,
                title="Missingness by feature",
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Could not infer feature / missingness columns from DataMissingness.csv.")

    # 3) Correlation heatmap
    corr_csv = exp_dir / "FeatureCorrelations.csv"
    df_corr = _safe_read_csv(corr_csv)
    if df_corr is not None and not df_corr.empty:
        st.subheader("Feature correlation heatmap")
        try:
            # Assume wide correlation matrix or edge-list
            # If "Feature1", "Feature2", "Correlation", pivot
            if {"Feature1", "Feature2", "Correlation"}.issubset(df_corr.columns):
                mat = df_corr.pivot(index="Feature1", columns="Feature2", values="Correlation")
            else:
                # try to interpret as square matrix
                mat = df_corr.set_index(df_corr.columns[0])
            fig = px.imshow(
                mat,
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                title="Feature correlation (Phase 1)",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not build correlation heatmap from {corr_csv.name}: {e}")


# -------------------------------------------------------------------
# Phase 2 - Impute & scale (preprocessing runtime / components)
# -------------------------------------------------------------------

def view_phase2_impute_scale(dataset_dir: Path) -> None:
    st.header("Phase 2 - Impute & Scale")

    # Use runtimes.csv and runtime_preprocessingX.txt to show per-fold preprocessing runtime
    runtimes_csv = dataset_dir / "runtimes.csv"
    df_rt = _safe_read_csv(runtimes_csv)
    if df_rt is not None and not df_rt.empty:
        st.subheader("Pipeline runtimes (all phases)")
        fig = px.bar(
            df_rt,
            x=df_rt.columns[0],
            y=df_rt.columns[1],
            title="Per-phase runtimes",
        )
        st.plotly_chart(fig, use_container_width=True)

    # More detailed preprocessing runtimes (one text file per CV)
    runtime_dir = dataset_dir / "runtime"
    if runtime_dir.is_dir():
        files = sorted(runtime_dir.glob("runtime_preprocessing*.txt"))
        records = []
        for f in files:
            try:
                fold = int("".join(filter(str.isdigit, f.stem)))
            except Exception:
                fold = None
            try:
                val = float(f.read_text().strip())
            except Exception:
                val = None
            records.append({"Fold": fold, "RuntimeSeconds": val})
        df_pre = pd.DataFrame(records).dropna()
        if not df_pre.empty:
            st.subheader("Preprocessing runtime per fold")
            fig2 = px.bar(df_pre, x="Fold", y="RuntimeSeconds",
                          title="Preprocessing runtime by CV fold")
            st.plotly_chart(fig2, use_container_width=True)


# -------------------------------------------------------------------
# Phase 3 - Feature learning
# -------------------------------------------------------------------

def view_phase3_feature_learning(dataset_dir: Path) -> None:
    st.header("Phase 3 - Feature Learning")

    fl_dir = dataset_dir / "feature_learning"
    if not fl_dir.is_dir():
        st.info("No feature_learning directory found for this dataset.")
        return

    # 1) Number of learned features per fold
    records = []
    for j in range(0, 100):  # arbitrary upper bound; break when missing
        manifest = fl_dir / f"feature_manifest_cv{j}.json"
        if not manifest.is_file():
            continue
        data = _safe_read_json(manifest)
        if not data:
            continue
        # Try to interpret: assume 'features' key or list
        if isinstance(data, dict) and "features" in data:
            n_feat = len(data["features"])
        elif isinstance(data, list):
            n_feat = len(data)
        else:
            # fallback: count keys
            n_feat = len(data)
        records.append({"Fold": j, "NumLearnedFeatures": n_feat})
    df_feat = pd.DataFrame(records)
    if not df_feat.empty:
        st.subheader("Number of learned features per CV fold")
        fig = px.bar(df_feat, x="Fold", y="NumLearnedFeatures",
                     title="Feature learning - features per fold")
        st.plotly_chart(fig, use_container_width=True)

    # 2) Simple feature frequency (across text files features_cvX.txt)
    freq: Dict[str, int] = {}
    for j in range(0, 100):
        feat_txt = fl_dir / f"features_cv{j}.txt"
        if not feat_txt.is_file():
            continue
        try:
            for line in feat_txt.read_text().splitlines():
                name = line.strip()
                if not name:
                    continue
                freq[name] = freq.get(name, 0) + 1
        except Exception:
            continue
    if freq:
        df_freq = (
            pd.DataFrame(
                [{"Feature": k, "Count": v} for k, v in freq.items()]
            )
            .sort_values("Count", ascending=False)
            .head(30)
        )
        st.subheader("Most frequently learned features (top 30)")
        fig2 = px.bar(df_freq, x="Feature", y="Count",
                      title="Feature learning - frequency",
                      labels={"Count": "Number of folds"})
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)


# -------------------------------------------------------------------
# Phase 4 & 5 - Feature importance / selection
# -------------------------------------------------------------------

def view_phase4_5_feature_importance_selection(dataset_dir: Path) -> None:
    st.header("Phases 4 & 5 - Feature Importance and Selection")

    fi_root = dataset_dir / "feature_importance"
    if fi_root.is_dir():
        st.subheader("Feature importance across CVs")

        # Two methods: multisurf and mutualinformation
        for method in ("multisurf", "mutualinformation"):
            method_dir = fi_root / method
            if not method_dir.is_dir():
                continue

            # Collect scores across folds
            scores = []
            for csv_path in sorted(method_dir.glob(f"{method}_scores_cv_*.csv")):
                df = _safe_read_csv(csv_path)
                if df is None or df.empty:
                    continue
                # Guess columns: first column feature name, second score
                if df.shape[1] >= 2:
                    feat_col = df.columns[0]
                    score_col = df.columns[1]
                    fold = int("".join(filter(str.isdigit, csv_path.stem)))
                    tmp = df[[feat_col, score_col]].copy()
                    tmp.columns = ["Feature", "Score"]
                    tmp["Fold"] = fold
                    scores.append(tmp)
            if not scores:
                continue

            df_scores = pd.concat(scores, ignore_index=True)
            # Aggregate mean / std
            agg = (
                df_scores.groupby("Feature")["Score"]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("mean", ascending=False)
                .head(30)
            )
            fig = px.bar(
                agg,
                x="Feature",
                y="mean",
                error_y="std",
                title=f"Top 30 features by {method} (mean ± std over CV)",
                labels={"mean": "Importance (mean score)"},
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    # Feature selection summary
    fs_csv = dataset_dir / "feature_selection" / "InformativeFeatureSummary.csv"
    df_fs = _safe_read_csv(fs_csv)
    if df_fs is not None and not df_fs.empty:
        st.subheader("Selected / informative features")

        # Assume columns like Feature, Selected, Method, etc. Fall back gracefully.
        cols = df_fs.columns
        if "Feature" in cols and "Selected" in cols:
            fig = px.histogram(
                df_fs,
                x="Feature",
                color="Selected",
                title="Selected vs non-selected features",
            )
            fig.update_layout(xaxis_tickangle=-60)
            st.plotly_chart(fig, use_container_width=True)

        # Distribution of number of methods supporting a feature, if that column exists
        method_count_col = None
        for c in cols:
            if "num" in c.lower() and "method" in c.lower():
                method_count_col = c
                break
        if method_count_col:
            fig2 = px.histogram(
                df_fs,
                x=method_count_col,
                nbins=10,
                title=f"Support across methods ({method_count_col})",
            )
            st.plotly_chart(fig2, use_container_width=True)


# -------------------------------------------------------------------
# Phase 6 - Base modeling (per-dataset)
# -------------------------------------------------------------------

def _plot_summary_metrics(summary_csv: Path, title_prefix: str) -> None:
    df = _safe_read_csv(summary_csv)
    if df is None or df.empty:
        st.info(f"No summary file found at {summary_csv}")
        return
    # Assume first column is algorithm, others metrics
    alg_col = df.columns[0]
    metric_cols = [c for c in df.columns[1:] if isinstance(c, str)]

    st.subheader(f"{title_prefix}: metric overview")

    metric = st.selectbox(
        "Metric",
        metric_cols,
        key=f"{title_prefix}_metric_select",
    )
    df_metric = df[[alg_col, metric]].copy()
    fig = px.bar(
        df_metric.sort_values(metric, ascending=False),
        x=alg_col,
        y=metric,
        title=f"{title_prefix}: {metric}",
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_curves_from_json(curves_dir: Path, title_prefix: str) -> None:
    if not curves_dir.is_dir():
        st.info(f"No curves_by_cv directory found at {curves_dir}")
        return

    # Expect files like MODEL_CV_k_prc.json / MODEL_CV_k_roc.json
    files = list(curves_dir.glob("*_CV_*_roc.json")) + list(curves_dir.glob("*_CV_*_prc.json"))
    if not files:
        st.info("No curve JSON files found.")
        return

    # Identify available models and curve types
    models = sorted({f.name.split("_CV_")[0] for f in files})
    curve_types = ["roc", "prc"]

    model = st.selectbox("Model", models, key=f"{title_prefix}_curve_model")
    curve_type = st.radio("Curve type", curve_types, horizontal=True,
                          key=f"{title_prefix}_curve_type")

    # Aggregate all folds for that model + curve_type
    traces = []
    for f in sorted(curves_dir.glob(f"{model}_CV_*_{curve_type}.json")):
        data = _safe_read_json(f)
        if not data:
            continue
        x = data.get("fpr") or data.get("recall") or data.get("x") or []
        y = data.get("tpr") or data.get("precision") or data.get("y") or []
        if not x or not y:
            continue
        fold_name = f.stem.split("_CV_")[1].split("_")[0]
        traces.append((fold_name, x, y))

    if not traces:
        st.info("No usable curve data found for this selection.")
        return

    fig = go.Figure()
    for fold, x, y in traces:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"CV {fold}",
            )
        )
    fig.update_layout(
        title=f"{title_prefix}: {model} - {curve_type.upper()} curves",
        xaxis_title="False Positive Rate" if curve_type == "roc" else "Recall",
        yaxis_title="True Positive Rate" if curve_type == "roc" else "Precision",
    )
    st.plotly_chart(fig, use_container_width=True)


def view_phase6_modeling(dataset_dir: Path) -> None:
    st.header("Phase 6 - Base Modeling")

    me_root = dataset_dir / "model_evaluation"

    # Summary metrics across models
    summary_mean = me_root / "Summary_performance_mean.csv"
    if summary_mean.is_file():
        _plot_summary_metrics(summary_mean, title_prefix="Base models (mean)")
    else:
        st.info("Summary_performance_mean.csv not found for base models.")

    # CV curve plots (ROC / PRC)
    st.subheader("Cross-validation ROC / PRC curves")
    curves_dir = me_root / "curves_by_cv"
    _plot_curves_from_json(curves_dir, title_prefix="Base models")


# -------------------------------------------------------------------
# Phase 7 - Ensembles (per-dataset)
# -------------------------------------------------------------------

def view_phase7_ensembles(dataset_dir: Path) -> None:
    st.header("Phase 7 - Ensembles")

    ens_root = dataset_dir / "ensemble_evaluation"
    if not ens_root.is_dir():
        st.info("No ensemble_evaluation directory found for this dataset.")
        return

    # Summary ensemble metrics
    ens_summary = ens_root / "Ensembles_performance_mean.csv"
    if ens_summary.is_file():
        _plot_summary_metrics(ens_summary, title_prefix="Ensembles (mean)")
    else:
        st.info("Ensembles_performance_mean.csv not found for this dataset.")

    # Ensemble ROC / PRC curves
    st.subheader("Ensemble ROC / PRC curves")
    curves_dir = ens_root / "curves_by_cv"
    _plot_curves_from_json(curves_dir, title_prefix="Ensembles")


# -------------------------------------------------------------------
# Phase 8 - Summary statistics / model comparisons
# -------------------------------------------------------------------

def view_phase8_statistics(dataset_dir: Path) -> None:
    st.header("Phase 8 - Summary Statistics")

    me_root = dataset_dir / "model_evaluation"
    stats_dir = me_root / "statistical_comparisons"
    if not stats_dir.is_dir():
        st.info("No statistical_comparisons directory found.")
        return

    # 1) Kruskal-Wallis per metric (algorithm comparisons)
    kw_csv = stats_dir / "KruskalWallis.csv"
    df_kw = _safe_read_csv(kw_csv)
    if df_kw is not None and not df_kw.empty:
        st.subheader("Kruskal-Wallis test results")
        # assume columns Metric, Statistic, P-Value, Sig(*)
        metric_col = None
        stat_col = None
        p_col = None
        for c in df_kw.columns:
            lc = c.lower()
            if metric_col is None and "metric" in lc:
                metric_col = c
            if stat_col is None and "statistic" in lc:
                stat_col = c
            if p_col is None and ("p-value" in lc or "p value" in lc or lc == "p"):
                p_col = c
        if metric_col and p_col:
            df_kw["-log10(p)"] = -np.log10(df_kw[p_col].replace(0, np.nan))
            fig = px.bar(
                df_kw.sort_values(p_col),
                x=metric_col,
                y="-log10(p)",
                title="Kruskal-Wallis significance by metric (-log10 p)",
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_kw)

    # 2) Mann-Whitney and Wilcoxon per metric
    mw_files = sorted(stats_dir.glob("MannWhitneyU_*.csv")) + sorted(
        stats_dir.glob("MannWhitneyU_*.csv")
    )
    if mw_files:
        st.subheader("Mann-Whitney U results")
        f = st.selectbox("Select Mann-WhitneyU result file", [p.name for p in mw_files])
        df_mw = _safe_read_csv(stats_dir / f)
        if df_mw is not None and not df_mw.empty:
            st.dataframe(df_mw.head(200))

    w_files = sorted(stats_dir.glob("WilcoxonRank_*.csv"))
    if w_files:
        st.subheader("Wilcoxon rank-sum results")
        f2 = st.selectbox("Select Wilcoxon result file", [p.name for p in w_files])
        df_w = _safe_read_csv(stats_dir / f2)
        if df_w is not None and not df_w.empty:
            st.dataframe(df_w.head(200))


# -------------------------------------------------------------------
# Phase 9 - Dataset comparisons (experiment-level)
# -------------------------------------------------------------------

def view_phase9_dataset_comparisons(exp_root: Path) -> None:
    st.header("Phase 9 - Dataset Comparisons")

    dc_dir = exp_root / "DatasetComparisons"
    if not dc_dir.is_dir():
        st.info("No DatasetComparisons directory found at experiment root.")
        return

    # 1) BestCompare_KruskalWallis (best algorithm per dataset, per metric)
    best_kw = dc_dir / "BestCompare_KruskalWallis.csv"
    df_best_kw = _safe_read_csv(best_kw)
    if df_best_kw is not None and not df_best_kw.empty:
        st.subheader("Best algorithm per dataset - Kruskal-Wallis")

        metric_options = list(df_best_kw.index) if df_best_kw.index.name else list(
            df_best_kw[df_best_kw.columns[0]].unique()
        )
        metric = st.selectbox("Metric", metric_options, key="p9_metric")
        if df_best_kw.index.name:
            # DataFrame indexed by metric
            row = df_best_kw.loc[metric]
        else:
            row = df_best_kw[df_best_kw[df_best_kw.columns[0]] == metric].iloc[0]

        # Extract dataset-level stats
        cols = list(df_best_kw.columns)
        ds_stats = []
        ds_idx = 1
        while True:
            alg_col = f"Best_Alg_D{ds_idx}"
            mean_col = f"Mean_D{ds_idx}"
            std_col = f"Std_D{ds_idx}"
            if alg_col not in cols:
                break
            alg = row.get(alg_col, None)
            mean_val = row.get(mean_col, None)
            std_val = row.get(std_col, None)
            if pd.isna(mean_val):
                break
            ds_stats.append(
                {
                    "Dataset": f"D{ds_idx}",
                    "BestAlgorithm": alg,
                    "MeanScore": float(mean_val),
                    "StdScore": float(std_val) if not pd.isna(std_val) else None,
                }
            )
            ds_idx += 1

        if ds_stats:
            df_stats = pd.DataFrame(ds_stats)
            fig = px.bar(
                df_stats,
                x="Dataset",
                y="MeanScore",
                color="BestAlgorithm",
                error_y="StdScore",
                barmode="group",
                title=f"Best algorithm per dataset - {metric}",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_stats)

    # 2) MannWhitney_all / WilcoxonRank_all - dataset pairwise comparisons
    mw_all = dc_dir / "MannWhitney_all.csv"
    df_mw_all = _safe_read_csv(mw_all)
    if df_mw_all is not None and not df_mw_all.empty:
        st.subheader("Pairwise dataset comparisons - Mann-Whitney U (all algorithms)")
        metric_col = "Metric" if "Metric" in df_mw_all.columns else df_mw_all.columns[0]
        metric = st.selectbox("Metric (Mann-Whitney)", sorted(df_mw_all[metric_col].unique()))
        df_sub = df_mw_all[df_mw_all[metric_col] == metric]
        # Build heatmap of -log10(p) per dataset pair
        d1 = df_sub["Data1"].astype(str)
        d2 = df_sub["Data2"].astype(str)
        p_vals = df_sub["P-Value"] if "P-Value" in df_sub.columns else df_sub[df_sub.columns[4]]
        df_heat = pd.DataFrame({"Data1": d1, "Data2": d2, "p": p_vals})
        ds = sorted(set(df_heat["Data1"]).union(set(df_heat["Data2"])))
        mat = pd.DataFrame(index=ds, columns=ds, dtype=float)
        for _, row in df_heat.iterrows():
            mat.loc[row["Data1"], row["Data2"]] = -np.log10(max(row["p"], 1e-12))
            mat.loc[row["Data2"], row["Data1"]] = mat.loc[row["Data1"], row["Data2"]]
        np.fill_diagonal(mat.values, 0.0)
        fig_hm = px.imshow(
            mat,
            title=f"Mann-Whitney U - dataset pairwise significance (-log10 p) for {metric}",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig_hm, use_container_width=True)


# -------------------------------------------------------------------
# Phase 11 - High-level reporting overview
# -------------------------------------------------------------------

def view_phase_11_reporting(exp_root: Path) -> None:
    st.header("Phases 11 - Reporting")

    rep_dir = exp_root / "reporting"
    if not rep_dir.is_dir():
        st.info("No reporting directory found.")
        return

    report_json = rep_dir / "report_data.json"
    data = _safe_read_json(report_json)
    if data:
        st.subheader("Report summary (JSON)")
        st.json(data)
    else:
        st.info("report_data.json not found or empty.")

    # Runtime of reporting / compare phase
    rt_dir = exp_root / "runtime"
    if rt_dir.is_dir():
        runtimes = []
        for f in ["runtime_report.txt", "runtime_compare_datasets.txt"]:
            path = rt_dir / f
            if path.is_file():
                try:
                    val = float(path.read_text().strip())
                    runtimes.append({"Phase": f.replace(".txt", ""), "RuntimeSeconds": val})
                except Exception:
                    continue
        if runtimes:
            df = pd.DataFrame(runtimes)
            fig = px.bar(df, x="Phase", y="RuntimeSeconds", title="Reporting-related runtimes")
            st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# Main App
# -------------------------------------------------------------------

def main(exp_root: Path) -> None:
    st.set_page_config(page_title="STREAMLINE Reporting Dashboard", layout="wide")
    st.title("STREAMLINE Reporting Dashboard (Plotly + Streamlit)")

    if not exp_root.is_dir():
        st.error(f"Experiment root not found: {exp_root}")
        return

    datasets = _list_datasets(exp_root)
    dataset_names = [p.name for p in datasets]

    st.sidebar.header("Navigation")
    dataset_name = st.sidebar.selectbox("Dataset", dataset_names, index=0 if dataset_names else None)
    phase = st.sidebar.radio(
        "Phase",
        [
            "Overview",
            "P1 - Exploratory Analysis",
            "P2 - Impute & Scale",
            "P3 - Feature Learning",
            "P4 - Feature Importance",
            "P6 - Base Modeling",
            "P7 - Ensembles",
            "P8 - Summary Statistics",
            "P9 - Dataset Comparisons",
            "P11 - Reporting",
        ],
    )

    dataset_dir = exp_root / dataset_name

    if phase == "Overview":
        st.header("Experiment Overview")
        st.write(f"Selected dataset: `{dataset_name}`")

        # Simple overview: show runtimes.csv for both datasets in one plot
        overview_records = []
        for ds in datasets:
            rt = _safe_read_csv(ds / "runtimes.csv")
            if rt is None or rt.empty:
                continue
            # Assume two columns: Phase, RuntimeSeconds (or similar)
            if rt.shape[1] >= 2:
                phase_col = rt.columns[0]
                val_col = rt.columns[1]
                tmp = rt[[phase_col, val_col]].copy()
                tmp.columns = ["Phase", "RuntimeSeconds"]
                tmp["Dataset"] = ds.name
                overview_records.append(tmp)
        if overview_records:
            df_over = pd.concat(overview_records, ignore_index=True)
            fig = px.bar(
                df_over,
                x="Phase",
                y="RuntimeSeconds",
                color="Dataset",
                barmode="group",
                title="Runtimes by phase and dataset",
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No runtimes.csv found for overview plot.")

    elif phase == "P1 - Data Process / Exploratory":
        view_phase1_exploratory(dataset_dir)

    elif phase == "P2 - Impute & Scale":
        view_phase2_impute_scale(dataset_dir)

    elif phase == "P3 - Feature Learning":
        view_phase3_feature_learning(dataset_dir)

    elif phase == "P4&5 - Feature Importance / Selection":
        view_phase4_5_feature_importance_selection(dataset_dir)

    elif phase == "P6 - Base Modeling":
        view_phase6_modeling(dataset_dir)

    elif phase == "P7 - Ensembles":
        view_phase7_ensembles(dataset_dir)

    elif phase == "P8 - Summary Statistics":
        view_phase8_statistics(dataset_dir)

    elif phase == "P9 - Dataset Comparisons":
        view_phase9_dataset_comparisons(exp_root)

    elif phase == "P11 - Reporting":
        view_phase_11_reporting(exp_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_root",
        type=str,
        default="./out/",
        help="Path to experiments root",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="DemoRun",
        help="Experiment name",
    )

    args, _ = parser.parse_known_args()
    main(Path(args.exp_root + "/" + args.experiment_name))
