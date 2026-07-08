# app.py
import numpy as np
import pandas as pd
import streamlit as st

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import factor_cmap

# -------------------------
# Sample Data Generation
# -------------------------

def make_sample_metrics():
    models = ["LogisticRegression", "RandomForest", "XGBoost", "SVM", "NB"]
    metrics = ["Balanced Accuracy", "Accuracy", "F1", "Precision", "Recall", "ROC AUC", "PRC APS"]

    rows = []
    rng = np.random.default_rng(42)
    for m in models:
        for metric in metrics:
            mean = rng.uniform(0.7, 0.95)
            std = rng.uniform(0.01, 0.04)
            rows.append(
                dict(
                    model=m,
                    metric=metric,
                    mean=mean,
                    std=std,
                )
            )

    df = pd.DataFrame(rows)
    # Derive rank per metric (higher is better)
    df["rank"] = df.groupby("metric")["mean"].rank(ascending=False, method="min")
    return df


def make_sample_ensembles():
    rows = [
        # ensemble, best_base, Δ BalancedAcc, Δ ROC AUC, Δ PRC APS
        ("Vote_Hard", "RandomForest", 0.01, 0.005, 0.007),
        ("Vote_Soft", "XGBoost", 0.02, 0.012, 0.015),
        ("Stack_LR", "XGBoost", 0.025, 0.018, 0.020),
        ("Stack_RF", "RandomForest", 0.015, 0.010, 0.012),
    ]
    df = pd.DataFrame(rows, columns=["ensemble", "best_base", "delta_bal_acc", "delta_roc_auc", "delta_prc_aps"])
    return df


def make_sample_feature_importance():
    features = [f"Feature_{i}" for i in range(1, 21)]
    algos = ["MI", "MSWRFDB", "MSWRFDB*"]
    rng = np.random.default_rng(123)

    rows = []
    for algo in algos:
        for rank, feat in enumerate(features, start=1):
            importance = rng.uniform(0.0, 1.0) * (1.0 / rank)
            rows.append(
                dict(
                    feature=feat,
                    algorithm=algo,
                    avg_importance=importance,
                    avg_rank=rank,
                )
            )
    df = pd.DataFrame(rows)
    # Normalize importance within algorithm
    df["norm_importance"] = df.groupby("algorithm")["avg_importance"].transform(
        lambda x: x / x.max()
    )
    return df


def make_sample_crossphase_correlations():
    features = [f"Feature_{i}" for i in range(1, 16)]
    metrics = ["ROC AUC", "PRC APS", "F1"]
    rng = np.random.default_rng(999)

    rows = []
    for feat in features:
        for met in metrics:
            corr = rng.uniform(-1.0, 1.0)
            rows.append(
                dict(
                    feature=feat,
                    metric=met,
                    correlation=corr,
                )
            )
    df = pd.DataFrame(rows)
    return df


def make_sample_curves():
    """Generate toy ROC & PRC curves per model."""
    rng = np.random.default_rng(777)
    models = ["LogisticRegression", "RandomForest", "XGBoost", "SVM"]

    roc_curves = {}
    prc_curves = {}

    for m in models:
        # ROC
        fpr = np.linspace(0, 1, 50)
        base = rng.uniform(0.7, 0.9)
        tpr_noise = rng.normal(0, 0.03, size=fpr.shape)
        tpr = np.clip(base * fpr + 0.1 + tpr_noise, 0, 1)
        roc_curves[m] = pd.DataFrame({"fpr": fpr, "tpr": tpr})

        # PRC
        recall = np.linspace(0, 1, 50)
        prec_base = rng.uniform(0.7, 0.9)
        prec_noise = rng.normal(0, 0.03, size=recall.shape)
        precision = np.clip(prec_base * (1 - recall / 2) + prec_noise, 0, 1)
        prc_curves[m] = pd.DataFrame({"recall": recall, "precision": precision})

    return roc_curves, prc_curves


# -------------------------
# Bokeh Plot Helpers
# -------------------------

def bokeh_bar_model_metric(df_metrics, metric):
    df = df_metrics[df_metrics["metric"] == metric].copy()
    df = df.sort_values("mean", ascending=False)
    models = df["model"].tolist()
    source = ColumnDataSource(df)

    p = figure(
        x_range=models,
        height=400,
        title=f"{metric} by Model",
        toolbar_location="right",
        tools="pan,box_zoom,reset,save"
    )
    cmap = factor_cmap("model", palette="Category10_10", factors=models)

    p.vbar(
        x="model",
        top="mean",
        width=0.7,
        source=source,
        fill_color=cmap,
        line_color="black",
    )

    p.add_tools(
        HoverTool(
            tooltips=[
                ("Model", "@model"),
                ("Mean", "@mean{0.3f}"),
                ("Std", "@std{0.3f}"),
                ("Rank", "@rank{0}"),
            ]
        )
    )
    p.yaxis.axis_label = metric
    p.xaxis.major_label_orientation = 1.0
    return p


def bokeh_bar_ensembles(df_ens, metric_col, title=None):
    df = df_ens.copy()
    ensembles = df["ensemble"].tolist()
    source = ColumnDataSource(df)

    if title is None:
        title = f"Ensemble Δ {metric_col} vs Best Base"

    p = figure(
        x_range=ensembles,
        height=400,
        title=title,
        toolbar_location="right",
        tools="pan,box_zoom,reset,save"
    )
    cmap = factor_cmap("ensemble", palette="Category10_10", factors=ensembles)

    p.vbar(
        x="ensemble",
        top=metric_col,
        width=0.7,
        source=source,
        fill_color=cmap,
        line_color="black",
    )
    p.add_tools(
        HoverTool(
            tooltips=[
                ("Ensemble", "@ensemble"),
                ("Best Base", "@best_base"),
                (f"Δ {metric_col}", f"@{metric_col}{{0.3f}}"),
            ]
        )
    )
    p.yaxis.axis_label = f"Δ {metric_col}"
    p.xaxis.major_label_orientation = 1.0
    return p


def bokeh_bar_feature_importance(df_fi, algorithm, top_k=15):
    df = df_fi[df_fi["algorithm"] == algorithm].copy()
    df = df.sort_values("norm_importance", ascending=False).head(top_k)
    features = df["feature"].tolist()
    df["feature_str"] = df["feature"].astype(str)

    source = ColumnDataSource(df)

    p = figure(
        x_range=features,
        height=400,
        title=f"Top {top_k} Features ({algorithm})",
        toolbar_location="right",
        tools="pan,box_zoom,reset,save"
    )
    cmap = factor_cmap("feature_str", palette="Category10_10", factors=features)

    p.vbar(
        x="feature_str",
        top="norm_importance",
        width=0.7,
        source=source,
        fill_color=cmap,
        line_color="black",
    )

    p.add_tools(
        HoverTool(
            tooltips=[
                ("Feature", "@feature"),
                ("Avg Importance", "@avg_importance{0.3f}"),
                ("Norm Importance", "@norm_importance{0.3f}"),
                ("Avg Rank", "@avg_rank{0}"),
            ]
        )
    )

    p.yaxis.axis_label = "Normalized Importance"
    p.xaxis.major_label_orientation = 1.2
    return p


def bokeh_heatmap_crossphase(df_corr):
    features = sorted(df_corr["feature"].unique())
    metrics = sorted(df_corr["metric"].unique())

    df = df_corr.copy()
    df["feature_idx"] = df["feature"].astype(str)
    df["metric_idx"] = df["metric"].astype(str)

    source = ColumnDataSource(df)

    p = figure(
        x_range=features,
        y_range=list(reversed(metrics)),
        x_axis_location="above",
        height=400,
        title="Cross-phase Correlations (Feature vs Metric)",
        tools="pan,box_zoom,reset,save",
        toolbar_location="right"
    )

    mapper_min, mapper_max = -1.0, 1.0

    p.rect(
        x="feature_idx",
        y="metric_idx",
        width=1,
        height=1,
        source=source,
        line_color=None,
        fill_color="navy",
        fill_alpha=0.5,
    )

    p.add_tools(
        HoverTool(
            tooltips=[
                ("Feature", "@feature"),
                ("Metric", "@metric"),
                ("Correlation", "@correlation{0.3f}")
            ]
        )
    )

    p.xaxis.major_label_orientation = 1.2
    return p


def bokeh_roc_plot(roc_curves, selected_models):
    p = figure(
        height=400,
        title="ROC Curves",
        x_axis_label="False Positive Rate",
        y_axis_label="True Positive Rate",
        tools="pan,box_zoom,reset,save",
        toolbar_location="right",
    )
    p.line([0, 1], [0, 1], line_dash="dashed", color="gray", legend_label="No-skill")

    colors = ["blue", "green", "red", "purple", "orange", "brown"]
    for i, model in enumerate(selected_models):
        df = roc_curves[model]
        p.line(df["fpr"], df["tpr"], line_width=2, color=colors[i % len(colors)], legend_label=model)

    p.legend.location = "lower right"
    return p


def bokeh_prc_plot(prc_curves, selected_models, no_skill_precision=0.3):
    p = figure(
        height=400,
        title="PRC Curves",
        x_axis_label="Recall",
        y_axis_label="Precision",
        tools="pan,box_zoom,reset,save",
        toolbar_location="right",
    )
    p.line([0, 1], [no_skill_precision, no_skill_precision], line_dash="dashed", color="gray", legend_label="No-skill")

    colors = ["blue", "green", "red", "purple", "orange", "brown"]
    for i, model in enumerate(selected_models):
        df = prc_curves[model]
        p.line(df["recall"], df["precision"], line_width=2, color=colors[i % len(colors)], legend_label=model)

    p.legend.location = "lower left"
    return p


# -------------------------
# Streamlit App
# -------------------------

def main():
    st.set_page_config(page_title="STREAMLINE Statistics & Reporting Demo", layout="wide")

    st.title("STREAMLINE Phase 8 — Statistics & Reporting Demo")
    st.write(
        "Sample Streamlit app using **pandas** and **Bokeh** to visualize "
        "the kinds of outputs our statistics/reporting phase should produce."
    )

    # Generate sample data once (could cache with st.cache_data in real app)
    df_metrics = make_sample_metrics()
    df_ensembles = make_sample_ensembles()
    df_fi = make_sample_feature_importance()
    df_corr = make_sample_crossphase_correlations()
    roc_curves, prc_curves = make_sample_curves()

    st.sidebar.header("Controls")
    section = st.sidebar.selectbox(
        "Section",
        [
            "Summary Metrics",
            "Ensemble vs Base",
            "Feature Importance",
            "Cross-phase Correlations",
            "ROC & PRC Curves",
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Demo only - all numbers are synthetic.")

    if section == "Summary Metrics":
        st.subheader("Summary Metrics (per model)")
        metric = st.selectbox("Metric", sorted(df_metrics["metric"].unique()))
        st.dataframe(df_metrics[df_metrics["metric"] == metric].sort_values("mean", ascending=False))

        p = bokeh_bar_model_metric(df_metrics, metric)
        st.bokeh_chart(p, use_container_width=True)

    elif section == "Ensemble vs Base":
        st.subheader("Ensemble vs Best Base Model")
        st.dataframe(df_ensembles)

        metric_choice = st.selectbox(
            "Delta metric",
            ["delta_bal_acc", "delta_roc_auc", "delta_prc_aps"],
            format_func=lambda x: {
                "delta_bal_acc": "Δ Balanced Accuracy",
                "delta_roc_auc": "Δ ROC AUC",
                "delta_prc_aps": "Δ PRC APS",
            }[x]
        )
        p = bokeh_bar_ensembles(df_ensembles, metric_choice)
        st.bokeh_chart(p, use_container_width=True)

    elif section == "Feature Importance":
        st.subheader("Feature Importance Summary")
        algorithm = st.selectbox("Algorithm", sorted(df_fi["algorithm"].unique()))
        top_k = st.slider("Top K Features", 5, 30, 15)
        st.dataframe(
            df_fi[df_fi["algorithm"] == algorithm]
            .sort_values("norm_importance", ascending=False)
            .head(top_k)
        )
        p = bokeh_bar_feature_importance(df_fi, algorithm, top_k=top_k)
        st.bokeh_chart(p, use_container_width=True)

    elif section == "Cross-phase Correlations":
        st.subheader("Feature vs Metric Correlations")
        st.write("E.g., correlation between Feature Importance ranks and performance metrics.")

        st.dataframe(df_corr.head(25))
        p = bokeh_heatmap_crossphase(df_corr)
        st.bokeh_chart(p, use_container_width=True)

    elif section == "ROC & PRC Curves":
        st.subheader("ROC & PRC Curves (Aggregated by Model)")
        models = sorted(roc_curves.keys())
        selected_models = st.multiselect("Models to plot", models, default=models[:3])

        if selected_models:
            col1, col2 = st.columns(2)
            with col1:
                p_roc = bokeh_roc_plot(roc_curves, selected_models)
                st.bokeh_chart(p_roc, use_container_width=True)

            with col2:
                # toy no-skill level
                no_skill_precision = 0.3
                p_prc = bokeh_prc_plot(prc_curves, selected_models, no_skill_precision=no_skill_precision)
                st.bokeh_chart(p_prc, use_container_width=True)
        else:
            st.info("Select at least one model to view curves.")

    st.markdown("---")
    st.caption("STREAMLINE Phase 8 demo - replace synthetic data with real P6/P7/P4 outputs in production.")


if __name__ == "__main__":
    main()
