# render_plots_from_artifacts.py

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _exploratory_dir(experiment_path: str, dataset_name: str) -> str:
    return os.path.join(experiment_path, dataset_name, "exploratory")


def render_missingness_hist(experiment_path: str, dataset_name: str, show: bool = False):
    """Render Missingness histogram from DataMissingness.csv."""
    exp_dir = _exploratory_dir(experiment_path, dataset_name)
    inp = os.path.join(exp_dir, "DataMissingness.csv")
    out = os.path.join(exp_dir, "DataMissingnessHistogram.png")
    df = pd.read_csv(inp, index_col=0)
    counts = df["Count"].values
    plt.figure()
    plt.hist(counts, bins=100)
    plt.xlabel("Missing Value Counts")
    plt.ylabel("Frequency")
    plt.title("Histogram of Missing Value Counts in Dataset")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    if show: plt.show()
    plt.close()


def render_class_counts_bar(experiment_path: str, dataset_name: str, show: bool = False, outcome_type: "str | None" = None):
    """Render class count bar chart from ClassCounts.csv."""
    exp_dir = _exploratory_dir(experiment_path, dataset_name)
    inp = os.path.join(exp_dir, "ClassCounts.csv")
    out = os.path.join(exp_dir, "ClassCountsBarPlot.png")
    df = pd.read_csv(inp, index_col=0)
    plt.figure()
    if outcome_type == "Continuous":
        plt.hist(df.index.astype(float), bins=100, weights=df["Count"].values)
        plt.ylabel("Count"); plt.xlabel("Label"); plt.title("Label Counts")
    else:
        df["Count"].plot(kind="bar")
        plt.ylabel("Count"); plt.title("Class Counts")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    if show: plt.show()
    plt.close()


def render_correlation_heatmap(experiment_path: str, dataset_name: str, initial: str = "", show: bool = False):
    """Render heatmap from FeatureCorrelations.csv."""
    exp_dir = _exploratory_dir(experiment_path, dataset_name)
    inp = os.path.join(exp_dir, f"{initial}FeatureCorrelations.csv")
    out = os.path.join(exp_dir, f"{initial}FeatureCorrelations.png")
    corr = pd.read_csv(inp, index_col=0)
    sns.set_style("white")
    plt.figure(figsize=(max(6, corr.shape[0] // 2), max(6, corr.shape[1] // 2)))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, vmax=1, vmin=-1, square=True, cmap="RdBu", cbar_kws={"shrink": .75})
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    if show: plt.show()
    plt.close()
    sns.set_theme()


def render_univariate_topk(experiment_path: str, dataset_name: str, data_csv: str,
                           outcome_label: str, categorical_features: list[str],
                           top_k: int = 20, sig_cutoff: float = 0.05, show: bool = False):
    """
    Plot top-k significant features using Univariate_Significance.csv and the processed data CSV.
    """
    exp_dir = _exploratory_dir(experiment_path, dataset_name)
    uni_path = os.path.join(exp_dir, "univariate_analyses", "Univariate_Significance.csv")
    df = pd.read_csv(uni_path, index_col=0)
    df = df.sort_values(by="p-value", ascending=True).head(top_k)
    data = pd.read_csv(data_csv)

    out_dir = os.path.join(exp_dir, "univariate_analyses")
    os.makedirs(out_dir, exist_ok=True)

    for feat, row in df.iterrows():
        pval = row["p-value"]
        if pd.isna(pval) or pval > sig_cutoff: 
            continue
        safe = feat.replace(" ", "").replace("*", "").replace("/", "")
        if feat in categorical_features:
            table = pd.crosstab(data[feat], data[outcome_label])
            table.plot(kind="bar"); plt.ylabel("Contingency Table Count")
            out = os.path.join(out_dir, f"Barplot_{safe}.png")
        else:
            if data[outcome_label].nunique() == 2:
                data.boxplot(column=feat, by=outcome_label); plt.ylabel(feat); plt.title("")
                out = os.path.join(out_dir, f"Boxplot_{safe}.png")
            else:
                data.plot(x=feat, y=outcome_label, kind="scatter")
                out = os.path.join(out_dir, f"Scatter_{safe}.png")
        plt.tight_layout(); plt.savefig(out, bbox_inches="tight")
        if show: plt.show()
        plt.close('all')


def render_anomaly_histograms(experiment_path: str, dataset_name: str, show: bool = False):
    """Recreate IF/LOF/EE histograms from imputed_anomaly_scores.csv."""
    base = os.path.join(_exploratory_dir(experiment_path, dataset_name), "anomaly_detection")
    scores = pd.read_csv(os.path.join(base, "imputed_anomaly_scores.csv"))
    for col in scores.columns:
        plt.figure(figsize=(8, 6))
        plt.hist(scores[col].values, bins=30, edgecolor='black')
        plt.title(f'Histogram of {col} Anomaly Scores')
        plt.xlabel('Anomaly Score'); plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(base, f'{col.lower().replace(" ", "_")}_histogram.png'))
        if show: plt.show()
        plt.close()


def render_anomaly_rank_heatmap(experiment_path: str, dataset_name: str, top_n: int | None = None, show: bool = False):
    """Recreate anomaly rank heatmap (optionally top-n instances) from rankings.csv."""
    base = os.path.join(_exploratory_dir(experiment_path, dataset_name), "anomaly_detection")
    ranks = pd.read_csv(os.path.join(base, "rankings.csv"))
    # Normalize 0-1 (1 most anomalous)
    norm = 1 - (ranks.drop(columns=["Avg_Rank"], errors="ignore") / ranks.drop(columns=["Avg_Rank"], errors="ignore").values.max())
    if top_n is not None and top_n < len(norm):
        # pick rows with smallest Avg_Rank
        top_idx = ranks["Avg_Rank"].nsmallest(top_n).index
        norm = norm.loc[top_idx]

    plt.figure(figsize=(14, 10))
    sns.heatmap(norm, annot=False, cmap=sns.diverging_palette(220, 10, as_cmap=True),
                cbar_kws={'label': 'Anomaly Score (1.0 - Most, 0.0 - Least)'},
                yticklabels=True)
    plt.yticks(rotation=0)
    plt.title('Anomaly Detection Heatmap')
    plt.xlabel('Algorithm'); plt.ylabel('Instance ID')
    plt.tight_layout()
    suffix = f"_top_{top_n}" if top_n else ""
    plt.savefig(os.path.join(base, f'anomaly_detection_heatmap{suffix}.png'))
    if show: plt.show()
    plt.close()


def render_all(experiment_path: str, dataset_name: str,
               *,
               initial: bool = False,
               data_csv: "str | None" = None,
               outcome_label: "str | None" = None,
               categorical_features_csv: "str | None" = None,
               outcome_type: "str | None" = None,
               univariate_top_k: int = 20,
               sig_cutoff: float = 0.05,
               show: bool = False):
    """
    Convenience wrapper to render common plots from artifacts.
    - If you want univariate plots, provide data_csv, outcome_label, categorical_features_csv.
    - If you want initial-correlation heatmap, set initial=True.
    """
    init_prefix = "initial/" if initial else ""

    # Missingness + Class counts + Correlation
    try: render_missingness_hist(experiment_path, dataset_name, show=show)
    except Exception as e: logging.warning(f"Missingness plot skipped: {e}")
    try: render_class_counts_bar(experiment_path, dataset_name, show=show, outcome_type=outcome_type)
    except Exception as e: logging.warning(f"Class counts plot skipped: {e}")
    try: render_correlation_heatmap(experiment_path, dataset_name, initial=init_prefix, show=show)
    except Exception as e: logging.warning(f"Correlation heatmap skipped: {e}")

    # Univariate (optional)
    if data_csv and outcome_label and categorical_features_csv:
        try:
            cat = list(pd.read_csv(categorical_features_csv)['Feature'])
        except Exception:
            # also allow one-line headerless CSV with features
            cat = list(pd.read_csv(categorical_features_csv, header=None).iloc[0].dropna())
        try:
            render_univariate_topk(
                experiment_path, dataset_name, data_csv,
                outcome_label, cat, top_k=univariate_top_k, sig_cutoff=sig_cutoff, show=show
            )
        except Exception as e:
            logging.warning(f"Univariate plots skipped: {e}")

    # Anomaly (if available)
    try: render_anomaly_histograms(experiment_path, dataset_name, show=show)
    except Exception as e: logging.info(f"Anomaly histograms skipped: {e}")
    try: render_anomaly_rank_heatmap(experiment_path, dataset_name, top_n=None, show=show)
    except Exception as e: logging.info(f"Anomaly heatmap skipped: {e}")

# --- CLI stub ---
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Render STREAMLINE P1 plots from saved CSV artifacts."
    )
    parser.add_argument("--experiment_path", "-e", required=True, help="Root experiments folder")
    parser.add_argument("--dataset_name", "-d", required=True, help="Dataset folder name under experiments")

    # global toggles
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    parser.add_argument("--all", action="store_true", help="Render all available plots (CSV must exist)")

    # individual plot switches
    parser.add_argument("--missingness", action="store_true", help="Render missingness histogram")
    parser.add_argument("--class_counts", action="store_true", help="Render class-counts bar chart")
    parser.add_argument("--correlation", action="store_true", help="Render correlation heatmap")
    parser.add_argument("--initial", action="store_true", help="Use initial/ prefix for correlation heatmap")

    # univariate options
    parser.add_argument("--univariate", action="store_true", help="Render top-k univariate plots")
    parser.add_argument("--data_csv", help="Path to processed data CSV (for univariate plots)")
    parser.add_argument("--outcome_label", help="Outcome column name in data_csv")
    parser.add_argument("--categorical_features_csv", help="Path to processed_categorical_features.csv")
    parser.add_argument("--univariate_top_k", type=int, default=20, help="Top-k features to plot (default: 20)")
    parser.add_argument("--sig_cutoff", type=float, default=0.05, help="p-value cutoff (default: 0.05)")

    # class counts rendering hint
    parser.add_argument("--outcome_type", choices=["Binary", "Multiclass", "Continuous"],
                        help="Outcome type (improves class-counts rendering)")

    # anomaly options
    parser.add_argument("--anomaly_hists", action="store_true", help="Render anomaly histograms (IF/LOF/EE)")
    parser.add_argument("--anomaly_heatmap", action="store_true", help="Render anomaly rank heatmap")
    parser.add_argument("--anomaly_top_n", type=int, default=None,
                        help="Limit anomaly heatmap to top-N most anomalous instances")

    args = parser.parse_args()

    # convenience: --all flips the right switches
    if args.all:
        args.missingness = True
        args.class_counts = True
        args.correlation = True
        args.univariate = args.univariate or (args.data_csv and args.outcome_label and args.categorical_features_csv)
        args.anomaly_hists = True
        args.anomaly_heatmap = True

    # dispatch
    try:
        if args.missingness:
            render_missingness_hist(args.experiment_path, args.dataset_name, show=args.show)

        if args.class_counts:
            render_class_counts_bar(args.experiment_path, args.dataset_name,
                                    outcome_type=args.outcome_type, show=args.show)

        if args.correlation:
            render_correlation_heatmap(args.experiment_path, args.dataset_name,
                                       initial=("initial/" if args.initial else ""), show=args.show)

        if args.univariate:
            if not (args.data_csv and args.outcome_label and args.categorical_features_csv):
                print("--univariate requires --data_csv, --outcome_label, and --categorical_features_csv", file=sys.stderr)
                sys.exit(2)
            try:
                cat = list(pd.read_csv(args.categorical_features_csv)['Feature'])
            except Exception:
                cat = list(pd.read_csv(args.categorical_features_csv, header=None).iloc[0].dropna())
            render_univariate_topk(
                experiment_path=args.experiment_path,
                dataset_name=args.dataset_name,
                data_csv=args.data_csv,
                outcome_label=args.outcome_label,
                categorical_features=cat,
                top_k=args.univariate_top_k,
                sig_cutoff=args.sig_cutoff,
                show=args.show,
            )

        if args.anomaly_hists:
            render_anomaly_histograms(args.experiment_path, args.dataset_name, show=args.show)

        if args.anomaly_heatmap:
            render_anomaly_rank_heatmap(args.experiment_path, args.dataset_name,
                                        top_n=args.anomaly_top_n, show=args.show)

        # if nothing selected, hint usage
        if not any([args.missingness, args.class_counts, args.correlation,
                    args.univariate, args.anomaly_hists, args.anomaly_heatmap, args.all]):
            parser.print_help()

    except Exception as exc:
        logging.exception("Rendering failed: %s", exc)
        sys.exit(1)
