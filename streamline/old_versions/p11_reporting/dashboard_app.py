# streamline/reporting/dashboard_app.py

from __future__ import annotations

import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def safe_read_csv(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    try:
        if path.is_file():
            return pd.read_csv(path, **kwargs)
    except Exception as e:
        st.warning(f"Failed to read CSV: {path} ({e})")
    return None


def safe_read_json(path: Path) -> Optional[dict]:
    try:
        if path.is_file():
            with path.open("r") as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Failed to read JSON: {path} ({e})")
    return None


def discover_datasets(exp_root: Path) -> List[Path]:
    """Return dataset folders (have CVDatasets)."""
    if not exp_root.is_dir():
        return []
    ds = []
    for p in sorted(exp_root.iterdir()):
        if p.is_dir() and (p / "CVDatasets").is_dir():
            ds.append(p)
    return ds


# -------------------------------------------------------------------
# Phase 1–2: exploratory / preprocessing
# -------------------------------------------------------------------
def render_phase12_exploratory(ds_dir: Path):
    st.subheader("Phase 1: Exploratory Analysis")

    exp_dir = ds_dir / "exploratory"
    if not exp_dir.is_dir():
        st.info("No exploratory outputs found for this dataset.")
        return

    # 1) Class counts
    class_counts = safe_read_csv(exp_dir / "ClassCounts.csv")
    if class_counts is not None and {"Class", "Count"}.issubset(class_counts.columns):
        fig = px.bar(
            class_counts,
            x="Class",
            y="Count",
            title="Class Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    # 2) Missingness
    missing = safe_read_csv(exp_dir / "DataMissingness.csv")
    if missing is not None and {"Variable", "Count"}.issubset(missing.columns):
        fig = px.bar(
            missing.sort_values("Count", ascending=False),
            x="Variable",
            y="Count",
            title="Missingness Count in Dataset",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Could not infer feature / missingness columns from DataMissingness.csv.")

    # 3) Correlation heatmap
    corr_csv = exp_dir / "FeatureCorrelations.csv"
    df_corr = safe_read_csv(corr_csv)
    if df_corr is not None and not df_corr.empty:
        st.subheader("Feature correlation heatmap")
        try:
            # Assume wide correlation matrix with feature names in first column and header
            mat = df_corr.set_index(df_corr.columns[0])
            fig = px.imshow(
                mat,
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                title="Feature correlation (Phase 1)",
                
            )
            fig.layout.update(
                width=800, height=800,
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                yaxis_autorange='reversed',
                dragmode='pan',
            )
            fig.update_xaxes(type='category')
            fig.update_yaxes(type='category')

            st.plotly_chart(fig, use_container_width=False, config = {'scrollZoom': True})
        except Exception as e:
            st.warning(f"Could not build correlation heatmap from {corr_csv.name}: {e}")

    # 4) Univariate significance
    uni_dir = exp_dir / "univariate_analyses"
    uni = safe_read_csv(uni_dir / "Univariate_Significance.csv")
    if uni is not None and {"Feature", "p_value"}.issubset(uni.columns):
        fig = px.bar(
            uni.sort_values("p_value"),
            x="Feature",
            y="p_value",
            title="Univariate Significance (sorted by p-value)",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# Phase 3: feature learning
# -------------------------------------------------------------------
def render_phase3_feature_learning(ds_dir: Path):
    st.subheader("Phase 3: Feature Learning")

    fl_dir = ds_dir / "feature_learning"
    if not fl_dir.is_dir():
        st.info("No feature learning outputs found.")
        return

    # Count engineered features per CV from feature_manifest_cv*.json
    rows = []
    for p in sorted(fl_dir.glob("feature_manifest_cv*.json")):
        manifest = safe_read_json(p)
        if not manifest:
            continue
        cv_id = p.stem.replace("feature_manifest_", "")
        # Try some generic keys, fallback to len of feature list if present
        n_feat = manifest.get("n_features") or manifest.get("num_features")
        if n_feat is None:
            feats = manifest.get("features") or manifest.get("feature_list")
            if isinstance(feats, list):
                n_feat = len(feats)
        if n_feat is not None:
            rows.append({"cv": cv_id, "n_features": n_feat})

    if rows:
        df = pd.DataFrame(rows)
        fig = px.bar(
            df,
            x="cv",
            y="n_features",
            title="Number of Learned Features per CV Fold",
        )
        st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# Phase 4–5: feature importance & selection
# -------------------------------------------------------------------
def render_phase45_feature_importance(ds_dir: Path):
    st.subheader("Phase 4–5: Feature Importance & Selection")

    fi_phase_dir = ds_dir / "feature_importance"
    sel_dir = ds_dir / "feature_selection"

    # P4: MultiSURF & Mutual information (top mean scores)
    ms_dir = fi_phase_dir / "multisurf"
    mi_dir = fi_phase_dir / "mutualinformation"

    def _aggregate_fi_scores(root: Path, label: str) -> Optional[pd.DataFrame]:
        if not root.is_dir():
            return None
        frames = []
        for p in sorted(root.glob("*_scores_cv_*.csv")):
            df = safe_read_csv(p)
            if df is None:
                continue
            cv = p.stem.split("_cv_")[-1]
            if "Feature" in df.columns and "Score" in df.columns:
                df = df[["Feature", "Score"]].copy()
                df["cv"] = cv
                frames.append(df)
        if not frames:
            return None
        all_scores = pd.concat(frames, ignore_index=True)
        agg = all_scores.groupby("Feature")["Score"].mean().reset_index()
        agg["method"] = label
        return agg

    ms_agg = _aggregate_fi_scores(ms_dir, "MultiSURF")
    mi_agg = _aggregate_fi_scores(mi_dir, "Mutual Information")

    agg_all = None
    if ms_agg is not None and mi_agg is not None:
        agg_all = pd.concat([ms_agg, mi_agg], ignore_index=True)
    elif ms_agg is not None:
        agg_all = ms_agg
    elif mi_agg is not None:
        agg_all = mi_agg

    if agg_all is not None:
        top = (
            agg_all.sort_values("Score", ascending=False)
            .groupby("method")
            .head(20)
        )
        fig = px.bar(
            top,
            x="Feature",
            y="Score",
            color="method",
            barmode="group",
            title="Global Feature Importance (Phase 4)",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # P5: Informative feature summary
    if sel_dir.is_dir():
        info = safe_read_csv(sel_dir / "InformativeFeatureSummary.csv")
        if info is not None and "Feature" in info.columns:
            # Use any importance/score-like columns if present
            score_cols = [
                c
                for c in info.columns
                if c.lower().startswith("score")
                or "importance" in c.lower()
                or "rank" in c.lower()
            ]
            if score_cols:
                col = score_cols[0]
                top_sel = info.sort_values(col, ascending=False).head(20)
                fig = px.bar(
                    top_sel,
                    x="Feature",
                    y=col,
                    title=f"Selected Features (Phase 5) – {col}",
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# Phase 6: modeling
# -------------------------------------------------------------------
def render_phase6_modeling(ds_dir: Path):
    st.subheader("Phase 6: Base Models")

    me_dir = ds_dir / "model_evaluation"
    if not me_dir.is_dir():
        st.info("No model_evaluation folder found.")
        return

    summary_mean = safe_read_csv(me_dir / "Summary_performance_mean.csv", index_col=0)
    if summary_mean is not None:
        summary_mean = summary_mean.reset_index().rename(columns={"index": "Model"})
        # Melt for metric comparison
        df_long = summary_mean.melt(
            id_vars="Model",
            var_name="Metric",
            value_name="Score",
        )
        fig = px.bar(
            df_long,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Mean CV Performance per Model",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Per-CV performance (metrics_by_cv JSON)
    metrics_dir = me_dir / "metrics_by_cv"
    if metrics_dir.is_dir():
        rows = []
        for p in sorted(metrics_dir.glob("*.json")):
            m = safe_read_json(p)
            if not m:
                continue
            base = p.stem  # e.g., LR_CV_0
            try:
                model_id, cv_id = base.split("_CV_")
            except ValueError:
                continue
            # assume flat dict of metrics
            for metric, val in m.items():
                if isinstance(val, (int, float)):
                    rows.append(
                        {
                            "Model": model_id,
                            "CV": cv_id,
                            "Metric": metric,
                            "Score": float(val),
                        }
                    )
        if rows:
            df = pd.DataFrame(rows)
            metric_sel = st.selectbox(
                "Metric (Phase 6 CV performance)",
                sorted(df["Metric"].unique()),
                key=f"p6_metric_{ds_dir.name}",
            )
            sub = df[df["Metric"] == metric_sel]
            fig = px.line(
                sub,
                x="CV",
                y="Score",
                color="Model",
                markers=True,
                title=f"Per-CV {metric_sel} by Model",
            )
            st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# Phase 7: ensembles
# -------------------------------------------------------------------
def render_phase7_ensembles(ds_dir: Path):
    st.subheader("Phase 7: Ensembles")

    ens_dir = ds_dir / "ensemble_evaluation"
    if not ens_dir.is_dir():
        st.info("No ensemble_evaluation folder found.")
        return

    summary_mean = safe_read_csv(ens_dir / "Ensembles_performance_mean.csv", index_col=0)
    if summary_mean is not None:
        summary_mean = summary_mean.reset_index().rename(columns={"index": "Ensemble"})
        df_long = summary_mean.melt(
            id_vars="Ensemble",
            var_name="Metric",
            value_name="Score",
        )
        fig = px.bar(
            df_long,
            x="Ensemble",
            y="Score",
            color="Metric",
            barmode="group",
            title="Mean CV Performance per Ensemble",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Per-CV ensemble metrics
    cv_dir = ens_dir / "metrics_by_cv"
    if cv_dir.is_dir():
        rows = []
        for p in sorted(cv_dir.glob("*.json")):
            m = safe_read_json(p)
            if not m:
                continue
            base = p.stem  # e.g., HEV_CV_0
            try:
                ens_id, cv_id = base.split("_CV_")
            except ValueError:
                continue
            for metric, val in m.items():
                if isinstance(val, (int, float)):
                    rows.append(
                        {
                            "Ensemble": ens_id,
                            "CV": cv_id,
                            "Metric": metric,
                            "Score": float(val),
                        }
                    )
        if rows:
            df = pd.DataFrame(rows)
            metric_sel = st.selectbox(
                "Metric (Phase 7 CV performance)",
                sorted(df["Metric"].unique()),
                key=f"p7_metric_{ds_dir.name}",
            )
            sub = df[df["Metric"] == metric_sel]
            fig = px.line(
                sub,
                x="CV",
                y="Score",
                color="Ensemble",
                markers=True,
                title=f"Per-CV {metric_sel} by Ensemble",
            )
            st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# Phase 8: summary statistics
# -------------------------------------------------------------------
def render_phase8_stats(ds_dir: Path):
    st.subheader("Phase 8: Statistics & Model Comparisons")

    me_dir = ds_dir / "model_evaluation"
    if not me_dir.is_dir():
        st.info("No model_evaluation folder for stats.")
        return

    # Metric comparison boxplots built from Summary_performance_mean
    summary_mean = safe_read_csv(me_dir / "Summary_performance_mean.csv", index_col=0)
    if summary_mean is not None:
        summary_mean = summary_mean.reset_index().rename(columns={"index": "Model"})
        metric_sel = st.selectbox(
            "Metric for model comparison (Phase 8)",
            [c for c in summary_mean.columns if c != "Model"],
            key=f"p8_metric_{ds_dir.name}",
        )
        sub = summary_mean[["Model", metric_sel]]
        fig = px.box(
            sub,
            x="Model",
            y=metric_sel,
            points="all",
            title=f"Model Comparison for {metric_sel}",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Statistical tests (Mann-Whitney / Wilcoxon) present under statistical_comparisons
    stats_dir = me_dir / "statistical_comparisons"
    if stats_dir.is_dir():
        mw_files = sorted(stats_dir.glob("MannWhitneyU_*.csv"))
        if mw_files:
            mw_file = st.selectbox(
                "Mann-WhitneyU result file",
                mw_files,
                format_func=lambda p: p.name,
                key=f"p8_mw_{ds_dir.name}",
            )
            df = safe_read_csv(mw_file)
            if df is not None and {"Model1", "Model2", "P-Value"}.issubset(df.columns):
                sig = df[df["P-Value"] < 0.05]
                if not sig.empty:
                    fig = px.scatter(
                        sig,
                        x="Model1",
                        y="Model2",
                        size="-log10(P-Value)" if "-log10(P-Value)" in sig.columns else "P-Value",
                        color="P-Value",
                        title="Significant Pairwise Differences (Mann-WhitneyU)",
                    )
                    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# Phase 9: dataset comparisons (experiment-level)
# -------------------------------------------------------------------
def render_phase9_dataset_comparisons(exp_root: Path):
    st.subheader("Phase 9: Dataset Comparisons")

    dc_dir = exp_root / "DatasetComparisons"
    if not dc_dir.is_dir():
        st.info("No DatasetComparisons folder at experiment root.")
        return

    # 1) BestCompare_KruskalWallis summary
    best_kw = safe_read_csv(dc_dir / "BestCompare_KruskalWallis.csv")
    if best_kw is not None:
        metric = st.selectbox(
            "Metric (BestCompare Kruskal-Wallis)",
            [m for m in best_kw.index] if isinstance(best_kw.index, pd.Index) else best_kw["Metric"].unique(),
            key="p9_metric_kw",
        )
        row = best_kw.loc[metric]
        cols = [c for c in best_kw.columns if c.startswith("Mean_D")]
        data = []
        for i, c in enumerate(cols, start=1):
            data.append(
                {
                    "DatasetIdx": f"D{i}",
                    "MeanScore": float(row[c]) if pd.notnull(row[c]) else None,
                    "BestAlg": row.get(f"Best_Alg_D{i}", None),
                }
            )
        df = pd.DataFrame(data).dropna(subset=["MeanScore"])
        if not df.empty:
            fig = px.bar(
                df,
                x="DatasetIdx",
                y="MeanScore",
                color="BestAlg",
                title=f"Best Algorithm per Dataset – {metric}",
            )
            st.plotly_chart(fig, use_container_width=True)

    # 2) Global Mann-Whitney & Wilcoxon across datasets
    for label, fname in [
        ("Mann-Whitney Across Datasets", "MannWhitney_all.csv"),
        ("Wilcoxon Rank Across Datasets", "WilcoxonRank_all.csv"),
    ]:
        df = safe_read_csv(dc_dir / fname)
        if df is not None and {"Metric", "Data1", "Data2", "P-Value"}.issubset(df.columns):
            st.markdown(f"#### {label}")
            sig = df[df["P-Value"] < 0.05].copy()
            if sig.empty:
                st.write("No significant differences at p < 0.05.")
                continue
            sig["pair"] = sig["Data1"].astype(str) + " vs " + sig["Data2"].astype(str)
            fig = px.scatter(
                sig,
                x="Metric",
                y="pair",
                size=-np.log10(sig["P-Value"]) if "np" in globals() else sig["P-Value"],
                color="P-Value",
                title=f"Significant Dataset Pairs ({label})",
            )
            st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# Reporting phase (Phase 11) artifacts
# -------------------------------------------------------------------
def render_phase11_reporting(exp_root: Path):
    st.subheader("Phase 11: Reporting")

    rep_dir = exp_root / "reporting"
    if not rep_dir.is_dir():
        st.info("No reporting folder found.")
        return
    
    html_path = rep_dir / "report.html"
    pdf_path = rep_dir / "report.pdf"
    if html_path.is_file():
        st.markdown(f"[Download HTML report]({html_path.as_posix()})")
    if pdf_path.is_file():
        st.markdown(f"[Download PDF report]({pdf_path.as_posix()})")

    
    report_data = safe_read_json(rep_dir / "report_data.json")
    if report_data:
        st.json(report_data)

# -------------------------------------------------------------------
# Phase runtimes per dataset
# -------------------------------------------------------------------
def render_runtimes(ds_dir: Path):
    st.subheader("Phase Runtimes (per dataset)")

    rt_csv = safe_read_csv(ds_dir / "runtimes.csv")
    if rt_csv is not None and {"Phase", "RuntimeSeconds"}.issubset(rt_csv.columns):
        fig = px.bar(
            rt_csv,
            x="Phase",
            y="RuntimeSeconds",
            title="Runtime per Phase",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="STREAMLINE Experiment Dashboard", layout="wide")

    st.title("STREAMLINE Experiment Dashboard")

    default_root = "out/DemoRun"
    exp_root_str = st.sidebar.text_input(
        "Experiment root path",
        value=default_root,
        help="Top-level folder containing dataset subfolders, DatasetComparisons, reporting, etc.",
    )
    exp_root = Path(exp_root_str).expanduser().resolve()

    st.sidebar.write(f"Using experiment root: `{exp_root}`")

    if not exp_root.is_dir():
        st.error("Experiment root does not exist.")
        return

    datasets = discover_datasets(exp_root)
    if not datasets:
        st.error("No dataset subfolders with CVDatasets found under this experiment root.")
        return

    ds_names = [d.name for d in datasets]
    selected_ds_names = st.sidebar.multiselect(
        "Datasets to display",
        ds_names,
        default=ds_names,
    )
    selected_ds = [d for d in datasets if d.name in selected_ds_names]

    if not selected_ds:
        st.warning("No datasets selected.")
        return

    top_tabs = st.tabs(
        [
            "Per-dataset (P1–P8)",
            "Dataset Comparisons (P9)",
            "Reporting (P11)",
        ]
    )

    # ------------------------------
    # Tab 1: per-dataset dashboard
    # ------------------------------
    with top_tabs[0]:
        st.header("Per-dataset Dashboard")

        # One accordion per dataset, each with phase tabs
        for ds in selected_ds:
            with st.expander(f"Dataset: {ds.name}", expanded=(len(selected_ds) == 1)):
                phase_tabs = st.tabs(
                    [
                        "Exploratory (P1–P2)",
                        "Feature Learning (P3)",
                        "Feature Importance/Selection (P4–P5)",
                        "Modeling (P6)",
                        "Ensembles (P7)",
                        "Summary Stats (P8)",
                        "Runtimes",
                    ]
                )

                with phase_tabs[0]:
                    render_phase12_exploratory(ds)
                with phase_tabs[1]:
                    render_phase3_feature_learning(ds)
                with phase_tabs[2]:
                    render_phase45_feature_importance(ds)
                with phase_tabs[3]:
                    render_phase6_modeling(ds)
                with phase_tabs[4]:
                    render_phase7_ensembles(ds)
                with phase_tabs[5]:
                    render_phase8_stats(ds)
                with phase_tabs[6]:
                    render_runtimes(ds)

    # ------------------------------
    # Tab 2: experiment-level dataset comparisons
    # ------------------------------
    with top_tabs[1]:
        render_phase9_dataset_comparisons(exp_root)

    # ------------------------------
    # Tab 3: reporting artifacts
    # ------------------------------
    with top_tabs[2]:
        render_phase11_reporting(exp_root)


if __name__ == "__main__":
    main()
