from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from fpdf import FPDF

logger = logging.getLogger(__name__)

Number = Union[int, float]


# ============================================================
# Plot export helper (only used as fallback)
# ============================================================

def _safe_plotly_to_png(fig, out_path: Path, scale: int = 2) -> bool:
    """
    Try to export plotly fig to PNG (kaleido). Return True if success.
    """
    try:
        import plotly.io as pio  # type: ignore

        out_path.parent.mkdir(parents=True, exist_ok=True)
        pio.write_image(fig, str(out_path), format="png", scale=scale)
        return True
    except Exception as e:
        logger.warning("Plotly export failed for %s: %r", out_path, e)
        return False


def _now_iso_local() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _try_streamline_version() -> str:
    try:
        import importlib.metadata as im
        return im.version("streamline")
    except Exception:
        return "unknown"


# ============================================================
# Precision formatting
# ============================================================

def _is_nan(x: Any) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False


def format_number(
    x: Any,
    *,
    decimals: int = 3,
    sci_small: float = 1e-3,
    sci_decimals: int = 3,
) -> str:
    """
    Central formatting for numeric table cells.

    - ints -> "42"
    - floats -> fixed decimals unless abs(x) < sci_small (and non-zero), then scientific
    - numeric strings -> parsed & formatted
    - other strings -> unchanged
    """
    if x is None or _is_nan(x):
        return ""

    if isinstance(x, bool):
        return str(x)

    if isinstance(x, int):
        return str(x)

    if isinstance(x, float):
        if x == 0.0:
            return f"{0:.{decimals}f}"
        ax = abs(x)
        if ax < sci_small:
            return f"{x:.{sci_decimals}e}"
        return f"{x:.{decimals}f}"

    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return ""
        try:
            f = float(s)
            return format_number(f, decimals=decimals, sci_small=sci_small, sci_decimals=sci_decimals)
        except Exception:
            return x

    return str(x)


# ============================================================
# Paths
# ============================================================

@dataclass
class ReportPaths:
    reporting_dir: Path
    data_json: Path
    pdf: Path
    figures_dir: Path


# ============================================================
# Main job
# ============================================================

class ReportPhaseJob:
    """
    Phase 11: Reporting (FPDF)

    Key behavior:
    - Prefer "original" precomputed PNGs in the experiment folder.
    - Only fallback to replot/export to reporting/figures if originals are missing.
    - Render a STREAMLINE-style boxed PDF (FPDF).
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        experiment_name: Optional[str] = None,
        experiment_path: Optional[str] = None,
        outcome_label: Optional[str] = None,
        outcome_type: Optional[str] = None,
        instance_label: Optional[str] = None,
        make_pdf: bool = True,
        float_decimals: int = 3,
    ):
        assert (output_path and experiment_name) or experiment_path, (
            "Provide (output_path, experiment_name) or experiment_path."
        )

        if experiment_path:
            self.exp_root = Path(experiment_path)
            self.output_path = str(self.exp_root.parent)
            self.experiment_name = self.exp_root.name
        else:
            self.output_path = str(output_path)
            self.experiment_name = str(experiment_name)
            self.exp_root = Path(self.output_path) / self.experiment_name

        if not self.exp_root.is_dir():
            raise FileNotFoundError(f"Experiment folder not found: {self.exp_root}")

        self.title = f"{self.experiment_name} - STREAMLINE Report"
        self.make_pdf = make_pdf
        self.job_start_time: Optional[float] = None

        self.outcome_label = outcome_label
        self.outcome_type = outcome_type
        self.instance_label = instance_label

        self.float_decimals = int(float_decimals)
        self.paths = self._init_paths()

    def _init_paths(self) -> ReportPaths:
        reporting_dir = self.exp_root / "reporting"
        figures_dir = reporting_dir / "figures"
        reporting_dir.mkdir(exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        return ReportPaths(
            reporting_dir=reporting_dir,
            data_json=reporting_dir / "report_data.json",
            pdf=reporting_dir / "report.pdf",
            figures_dir=figures_dir,
        )

    # ----------------------------
    # Discovery helpers
    # ----------------------------
    def _list_datasets(self) -> List[Path]:
        ignore = {
            "jobs",
            "logs",
            "jobsCompleted",
            "dask_logs",
            "runtime",
            "DatasetComparisons",
            "reporting",
        }
        ds = []
        for p in sorted(self.exp_root.iterdir()):
            if not p.is_dir():
                continue
            if p.name in ignore:
                continue
            if (p / "CVDatasets").is_dir():
                ds.append(p)
        return ds

    def _read_csv_if_exists(self, path: Path) -> Optional[pd.DataFrame]:
        try:
            if path.is_file():
                return pd.read_csv(path)
        except Exception as e:
            logger.warning("Failed reading CSV %s: %r", path, e)
        return None

    def _read_json_if_exists(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            if path.is_file():
                return json.loads(path.read_text())
        except Exception as e:
            logger.warning("Failed reading JSON %s: %r", path, e)
        return None

    # ============================================================
    # Prefer-original figure resolution (NEW)
    # ============================================================

    def _first_existing(self, candidates: Sequence[Path]) -> Optional[str]:
        for p in candidates:
            try:
                if p.is_file():
                    return str(p)
            except Exception:
                continue
        return None

    def _figure_path_exploratory_class_balance(self, ds_dir: Path) -> Optional[str]:
        # tree shows exploratory/ClassCountsBarPlot.png
        return self._first_existing([
            ds_dir / "exploratory" / "ClassCountsBarPlot.png",
            ds_dir / "exploratory" / "ClassCountsBarplot.png",
            ds_dir / "exploratory" / "ClassCounts.png",  # sometimes people export directly
        ])

    def _figure_path_exploratory_missingness(self, ds_dir: Path) -> Optional[str]:
        # no explicit PNG in your tree; prefer any likely existing plot if present
        # (if none exist, we'll fallback to plotting from DataMissingness.csv)
        return self._first_existing([
            ds_dir / "exploratory" / "DataMissingness.png",
            ds_dir / "exploratory" / "Missingness.png",
            ds_dir / "exploratory" / "MissingnessTop25.png",
        ])

    def _figure_path_model_summary_roc_prc(self, ds_dir: Path, kind: str) -> Optional[str]:
        # tree shows model_evaluation/Summary_ROC.png and Summary_PRC.png
        if kind.lower() == "roc":
            return self._first_existing([ds_dir / "model_evaluation" / "Summary_ROC.png"])
        return self._first_existing([ds_dir / "model_evaluation" / "Summary_PRC.png"])

    def _figure_path_model_metric_boxplot(self, ds_dir: Path, preferred_metric: str) -> Optional[str]:
        # tree shows model_evaluation/metricBoxplots/Compare_<metric>.png
        # filenames include spaces and parentheses exactly as in CSV headers
        return self._first_existing([
            ds_dir / "model_evaluation" / "metricBoxplots" / f"Compare_{preferred_metric}.png",
        ])

    def _figure_path_model_curves_single(self, ds_dir: Path, alg: str, kind: str) -> Optional[str]:
        # tree shows LR_ROC.png, LR_PRC.png etc.
        suffix = "ROC" if kind.lower() == "roc" else "PRC"
        return self._first_existing([
            ds_dir / "model_evaluation" / f"{alg}_{suffix}.png",
        ])

    def _figure_path_ensemble_summary(self, ds_dir: Path, kind: str) -> Optional[str]:
        # tree shows ensemble_evaluation/Summary_ROC_ensembles.png + Summary_PRC_ensembles.png
        if kind.lower() == "roc":
            return self._first_existing([ds_dir / "ensemble_evaluation" / "Summary_ROC_ensembles.png"])
        return self._first_existing([ds_dir / "ensemble_evaluation" / "Summary_PRC_ensembles.png"])

    def _figure_path_fs_top_scores(self, ds_dir: Path) -> Optional[str]:
        # tree shows dataset_root/feature_importance/<method>/TopAverageScores.png
        return self._first_existing([
            ds_dir / "feature_importance" / "multisurf" / "TopAverageScores.png",
            ds_dir / "feature_importance" / "mutualinformation" / "TopAverageScores.png",
        ])

    def _figure_path_model_fi_top(self, ds_dir: Path) -> Optional[str]:
        # no explicit FI plot under model_evaluation/feature_importance, only CSVs
        # but we can also use feature_importance/*/TopAverageScores.png which exists
        return self._figure_path_fs_top_scores(ds_dir)

    def _figure_path_dataset_comparisons_any(self) -> Optional[str]:
        # choose a strong default for dataset comparisons figure
        box_dir = self.exp_root / "DatasetComparisons" / "dataCompBoxplots"
        return self._first_existing([
            box_dir / "DataCompareAllModels_Balanced Accuracy.png",
            box_dir / "DataCompareAllModels_ROC AUC.png",
            box_dir / "DataCompareAllModels_PRC AUC.png",
            box_dir / "DataCompareAllModels_Accuracy.png",
        ])

    # ============================================================
    # Plot builders (fallback only)
    # ============================================================

    def _plot_model_summary_bars(self, summary_mean: pd.DataFrame, metric: str, title: str):
        import plotly.express as px  # type: ignore

        df = summary_mean.copy()
        if df.columns[0].lower() not in {"unnamed: 0", "model", "algorithm", "ml algorithm", "ml_algorithm"}:
            if df.index.name is not None:
                df = df.reset_index()
        name_col = df.columns[0]

        if metric not in df.columns:
            raise KeyError(metric)

        df = df[[name_col, metric]].dropna()
        df = df.sort_values(metric, ascending=False)

        fig = px.bar(df, x=name_col, y=metric, title=title)
        fig.update_layout(xaxis_title="Model", yaxis_title=metric)
        return fig

    def _plot_cv_metric_distribution(self, metrics_by_cv_dir: Path, metric: str, title: str):
        import plotly.express as px  # type: ignore

        rows = []
        for fn in sorted(metrics_by_cv_dir.glob("*.json")):
            blob = self._read_json_if_exists(fn)
            if not blob:
                continue
            alg = fn.name.split("_CV_")[0]
            val = blob.get(metric)
            if val is None:
                continue
            rows.append({"Model": alg, "Value": float(val)})

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError(f"No per-CV metric values found for metric={metric} under {metrics_by_cv_dir}")

        fig = px.box(df, x="Model", y="Value", points="all", title=title)
        fig.update_layout(xaxis_title="Model", yaxis_title=metric)
        return fig

    def _plot_class_counts(self, class_counts: pd.DataFrame, title: str):
        import plotly.express as px  # type: ignore

        df = class_counts.copy()
        low = {c.lower(): c for c in df.columns}
        label_col = None
        count_col = None

        for cand in ["class", "label", "outcome", "y", "group"]:
            if cand in low:
                label_col = low[cand]
                break
        for cand in ["count", "n", "num", "frequency", "freq"]:
            if cand in low:
                count_col = low[cand]
                break

        if label_col is None:
            label_col = df.columns[0]
        if count_col is None:
            count_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        df = df[[label_col, count_col]].dropna()
        df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
        df = df.dropna()
        fig = px.bar(df, x=label_col, y=count_col, title=title)
        fig.update_layout(xaxis_title="Class", yaxis_title="Count")
        return fig

    def _plot_missingness(self, missingness: pd.DataFrame, title: str):
        import plotly.express as px  # type: ignore

        df = missingness.copy()
        low = {c.lower(): c for c in df.columns}
        fcol = low.get("feature", df.columns[0])

        pcol = None
        for cand in ["missingpercent", "missing_percent", "percentmissing", "pct_missing", "missingpct"]:
            if cand in low:
                pcol = low[cand]
                break
        ccol = None
        for cand in ["missingcount", "missing_count", "countmissing", "n_missing"]:
            if cand in low:
                ccol = low[cand]
                break

        val_col = pcol or ccol or df.columns[-1]
        df = df[[fcol, val_col]].dropna()
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna().sort_values(val_col, ascending=False).head(25)

        fig = px.bar(df.iloc[::-1], x=val_col, y=fcol, orientation="h", title=title)
        fig.update_layout(xaxis_title=val_col, yaxis_title="Feature")
        return fig

    # ============================================================
    # Table building
    # ============================================================

    def _infer_alg_col(self, df: pd.DataFrame) -> str:
        low = {c.lower(): c for c in df.columns}
        for key in ["ml algorithm", "ml_algorithm", "algorithm", "model"]:
            if key in low:
                return low[key]
        return df.columns[0]

    def _build_mean_std_table(
        self,
        mean_df: Optional[pd.DataFrame],
        std_df: Optional[pd.DataFrame],
        highlight_metric_candidates: List[str],
        max_rows: int = 50,
    ) -> Dict[str, Any]:
        if mean_df is None or mean_df.empty or std_df is None or std_df.empty:
            return {"present": False}

        m = mean_df.copy()
        s = std_df.copy()

        alg_col_m = self._infer_alg_col(m)
        alg_col_s = self._infer_alg_col(s)
        if alg_col_m != "Algorithm":
            m = m.rename(columns={alg_col_m: "Algorithm"})
        if alg_col_s != "Algorithm":
            s = s.rename(columns={alg_col_s: "Algorithm"})

        highlight_metric = None
        for cand in highlight_metric_candidates:
            if cand in m.columns:
                highlight_metric = cand
                break

        common_metrics = [c for c in m.columns if c != "Algorithm" and c in s.columns]
        if not common_metrics:
            return {"present": False}

        merged = m[["Algorithm"] + common_metrics].merge(
            s[["Algorithm"] + common_metrics],
            on="Algorithm",
            how="inner",
            suffixes=("_mean", "_std"),
        )

        columns = ["Algorithm"] + common_metrics

        best_alg = None
        if highlight_metric and f"{highlight_metric}_mean" in merged.columns:
            try:
                best_idx = merged[f"{highlight_metric}_mean"].astype(float).idxmax()
                best_alg = str(merged.loc[best_idx, "Algorithm"])
            except Exception:
                best_alg = None

        rows = []
        for _, r in merged.head(max_rows).iterrows():
            cells = []
            for c in columns:
                if c == "Algorithm":
                    alg = str(r["Algorithm"])
                    cells.append({"value": alg, "best": bool(best_alg and alg == best_alg)})
                else:
                    mv = r.get(f"{c}_mean", None)
                    sv = r.get(f"{c}_std", None)
                    cells.append({"value": (mv, sv), "best": False})
            rows.append({"cells": cells})

        return {
            "present": True,
            "columns": columns,
            "rows": rows,
            "highlight_metric": highlight_metric,
            "best_algorithm": best_alg,
        }

    def _build_plain_table(self, df: Optional[pd.DataFrame], max_rows: int = 100) -> Dict[str, Any]:
        if df is None or df.empty:
            return {"present": False}
        df2 = df.head(max_rows).copy()
        df2 = df2.where(pd.notnull(df2), None)
        return {"present": True, "columns": list(df2.columns), "rows": df2.values.tolist()}

    # ============================================================
    # Data assembly with corrected paths + prefer originals
    # ============================================================

    def _collect_dataset_block(self, ds_dir: Path) -> Dict[str, Any]:
        name = ds_dir.name
        figs: Dict[str, str] = {}

        explore = ds_dir / "exploratory"
        data_process_summary = self._read_csv_if_exists(explore / "DataProcessSummary.csv")
        class_counts = self._read_csv_if_exists(explore / "ClassCounts.csv")
        missingness = self._read_csv_if_exists(explore / "DataMissingness.csv")
        univariate = self._read_csv_if_exists(explore / "univariate_analyses" / "Univariate_Significance.csv")
        univariate_top10 = univariate.head(10) if (univariate is not None and not univariate.empty) else None

        model_eval = ds_dir / "model_evaluation"
        summary_mean = self._read_csv_if_exists(model_eval / "Summary_performance_mean.csv")
        summary_std = self._read_csv_if_exists(model_eval / "Summary_performance_std.csv")
        summary_median = self._read_csv_if_exists(model_eval / "Summary_performance_median.csv")

        ens_eval = ds_dir / "ensemble_evaluation"
        ens_mean = self._read_csv_if_exists(ens_eval / "Ensembles_performance_mean.csv")
        ens_std = self._read_csv_if_exists(ens_eval / "Ensembles_performance_std.csv")
        ens_median = self._read_csv_if_exists(ens_eval / "Ensembles_performance_median.csv")

        feat_sel = self._read_csv_if_exists(ds_dir / "feature_selection" / "InformativeFeatureSummary.csv")
        runtimes = self._read_csv_if_exists(ds_dir / "runtimes.csv")

        # ------------------------------------------------
        # FIGURES: prefer originals, fallback to replot
        # ------------------------------------------------

        # Class balance
        figs["class_balance"] = self._figure_path_exploratory_class_balance(ds_dir) or ""
        if not figs["class_balance"]:
            if class_counts is not None and not class_counts.empty:
                try:
                    fig = self._plot_class_counts(class_counts, f"{name}: Class Balance")
                    out = self.paths.figures_dir / f"{name}_class_balance.png"
                    if _safe_plotly_to_png(fig, out):
                        figs["class_balance"] = str(out)
                except Exception as e:
                    logger.warning("Class balance plot failed for %s: %r", name, e)

        # Missingness (prefer any existing PNG, else fallback)
        figs["missingness"] = self._figure_path_exploratory_missingness(ds_dir) or ""
        if not figs["missingness"]:
            if missingness is not None and not missingness.empty:
                try:
                    fig = self._plot_missingness(missingness, f"{name}: Missingness (Top 25)")
                    out = self.paths.figures_dir / f"{name}_missingness_top25.png"
                    if _safe_plotly_to_png(fig, out):
                        figs["missingness"] = str(out)
                except Exception as e:
                    logger.warning("Missingness plot failed for %s: %r", name, e)

        # Models ROC/PRC summary (original Summary_ROC.png / Summary_PRC.png)
        roc_sum = self._figure_path_model_summary_roc_prc(ds_dir, "roc")
        prc_sum = self._figure_path_model_summary_roc_prc(ds_dir, "prc")
        if roc_sum:
            figs["models_roc_overlay"] = roc_sum
        if prc_sum:
            figs["models_prc_overlay"] = prc_sum

        # Model performance mean bar:
        # Prefer a precomputed boxplot image if available (Compare_Balanced Accuracy.png etc)
        # Otherwise, fallback to plotly bar from Summary_performance_mean.csv
        figs["models_mean_bar"] = ""
        figs["models_cv_box"] = ""

        chosen_metric = None
        if summary_mean is not None and not summary_mean.empty:
            for preferred in ["Balanced Accuracy", "ROC AUC", "PRC AUC", "Accuracy"]:
                if preferred in summary_mean.columns:
                    chosen_metric = preferred
                    break

        if chosen_metric:
            # Prefer original boxplot for CV distribution
            box = self._figure_path_model_metric_boxplot(ds_dir, chosen_metric)
            if box:
                figs["models_cv_box"] = box

            # Prefer also using same "Compare_*" as a strong visual for "mean performance"
            # (it’s not mean-bar, but it matches old visual reporting better)
            if box and not figs["models_mean_bar"]:
                figs["models_mean_bar"] = box

            # If still missing, fallback to plotly mean bar
            if not figs["models_mean_bar"]:
                try:
                    fig = self._plot_model_summary_bars(
                        summary_mean, chosen_metric, f"{name}: Mean {chosen_metric} (Models)"
                    )
                    out = self.paths.figures_dir / f"{name}_models_mean_{chosen_metric.replace(' ', '_')}.png"
                    if _safe_plotly_to_png(fig, out):
                        figs["models_mean_bar"] = str(out)
                except Exception as e:
                    logger.warning("Model mean bar plot failed for %s (%s): %r", name, chosen_metric, e)

        # Ensembles summaries (original)
        ens_roc = self._figure_path_ensemble_summary(ds_dir, "roc")
        ens_prc = self._figure_path_ensemble_summary(ds_dir, "prc")
        if ens_roc:
            figs["ensembles_roc"] = ens_roc
        if ens_prc:
            figs["ensembles_prc"] = ens_prc

        # Feature importance: prefer TopAverageScores.png from feature_importance/*
        fi_img = self._figure_path_model_fi_top(ds_dir)
        if fi_img:
            figs["fi_top20"] = fi_img
        else:
            # fallback: build from model_evaluation/feature_importance/*_FI.csv (as before)
            fi_dir = model_eval / "feature_importance"
            fi_files = sorted(fi_dir.glob("*_FI.csv")) if fi_dir.is_dir() else []
            if fi_files:
                try:
                    import plotly.express as px  # type: ignore

                    fi_df = pd.read_csv(fi_files[0])
                    cols = [c.lower() for c in fi_df.columns]
                    fcol = fi_df.columns[cols.index("feature")] if "feature" in cols else fi_df.columns[0]
                    icol = fi_df.columns[cols.index("importance")] if "importance" in cols else fi_df.columns[-1]
                    top = fi_df[[fcol, icol]].dropna()
                    top[icol] = pd.to_numeric(top[icol], errors="coerce")
                    top = top.dropna().sort_values(icol, ascending=False).head(20)
                    fig = px.bar(
                        top.iloc[::-1], x=icol, y=fcol, orientation="h",
                        title=f"{name}: Top 20 Feature Importance"
                    )
                    out = self.paths.figures_dir / f"{name}_fi_top20.png"
                    if _safe_plotly_to_png(fig, out):
                        figs["fi_top20"] = str(out)
                except Exception as e:
                    logger.warning("FI top20 plot failed for %s: %r", name, e)

        # Tables
        models_mean_std = self._build_mean_std_table(
            summary_mean, summary_std,
            highlight_metric_candidates=["Balanced Accuracy", "ROC AUC", "PRC AUC", "Accuracy"],
        )
        models_median = self._build_plain_table(summary_median, max_rows=100)

        ensembles_mean_std = self._build_mean_std_table(
            ens_mean, ens_std,
            highlight_metric_candidates=["Balanced Accuracy", "ROC AUC", "PRC AUC", "Accuracy"],
        )
        ensembles_median = self._build_plain_table(ens_median, max_rows=100)

        return {
            "dataset_name": name,
            "dataset_dir": str(ds_dir),
            "tables": {
                "data_process_summary": self._build_plain_table(data_process_summary, max_rows=50),
                "univariate_top10": self._build_plain_table(univariate_top10, max_rows=10),
                "informative_feature_summary": self._build_plain_table(feat_sel, max_rows=200),
                "runtimes": self._build_plain_table(runtimes, max_rows=500),
            },
            "perf": {
                "models_mean_std": models_mean_std,
                "models_median": models_median,
                "ensembles_mean_std": ensembles_mean_std,
                "ensembles_median": ensembles_median,
            },
            "figures": figs,
        }

    def _collect_dataset_comparisons_block(self) -> Dict[str, Any]:
        dc = self.exp_root / "DatasetComparisons"
        if not dc.is_dir():
            return {"present": False}

        best_kw = self._read_csv_if_exists(dc / "BestCompare_KruskalWallis.csv")
        best_mw = self._read_csv_if_exists(dc / "BestCompare_MannWhitney.csv")
        best_wx = self._read_csv_if_exists(dc / "BestCompare_WilcoxonRank.csv")

        figs: Dict[str, str] = {}

        # Prefer existing comparison plot(s)
        any_plot = self._figure_path_dataset_comparisons_any()
        if any_plot:
            figs["overview"] = any_plot

        # Keep prior KW p-value plot only as fallback
        if "overview" not in figs and best_kw is not None and not best_kw.empty:
            try:
                import plotly.express as px  # type: ignore

                df = best_kw.copy()
                low = {c.lower(): c for c in df.columns}
                pcol = None
                for cand in ["p-value", "p_value", "pvalue", "p"]:
                    if cand in low:
                        pcol = low[cand]
                        break
                if pcol is None and df.shape[1] == 1:
                    pcol = df.columns[0]

                if pcol is not None:
                    df = df[[pcol]].copy()
                    df[pcol] = pd.to_numeric(df[pcol], errors="coerce")
                    df = df.dropna().reset_index(drop=True)
                    df["Metric"] = [f"M{i+1}" for i in range(len(df))]
                    fig = px.bar(df, x="Metric", y=pcol, title="DatasetComparison: Kruskal-Wallis P-Values")
                    out = self.paths.figures_dir / "datasetcompare_kw_pvalues.png"
                    if _safe_plotly_to_png(fig, out):
                        figs["overview"] = str(out)
            except Exception as e:
                logger.warning("Dataset comparisons plot failed: %r", e)

        return {
            "present": True,
            "tables": {
                "best_kw": self._build_plain_table(best_kw, max_rows=200),
                "best_mw": self._build_plain_table(best_mw, max_rows=200),
                "best_wx": self._build_plain_table(best_wx, max_rows=200),
            },
            "figures": figs,
        }

    # ============================================================
    # PDF render
    # ============================================================

    def _write_pdf(self, report_data: Dict[str, Any]) -> None:
        """
        Render the full STREAMLINE-style PDF (FPDF).

        Requirements implemented:
        - No big "STREAMLINE Evaluation Report" header on every page:
        * Page 1 has a cover banner + cover header block.
        * Subsequent pages have no large header (only footer).
        - Metadata is rendered as a boxed KV panel and preserves ALL metadata keys.
        - Feature Importance always starts on a new page.
        - Runtimes always starts on a new page.
        """
        pdf = _StreamlinePDF(
            title=f"{report_data.get('experiment_name', '')} - STREAMLINE Report",
            streamline_version=str(report_data.get("streamline_version", "")),
            float_decimals=self.float_decimals,
        )
        pdf.alias_nb_pages()

        # -------------------------
        # COVER PAGE
        # -------------------------
        pdf.add_page()

        # Cover header block (boxed, reference-like)
        pdf.cover_header_block(
            experiment=str(report_data.get("experiment_name", "")),
            generated_at=str(report_data.get("generated_at", "")),
            version=str(report_data.get("streamline_version", "")),
        )

        # Metadata panel (keep ALL keys)
        meta = report_data.get("metadata", {}) or {}
        if meta:
            page_w = pdf.w - pdf.l_margin - pdf.r_margin
            x = pdf.l_margin
            y = pdf.get_y()
            # Single full-width panel; clean and scalable
            pdf.kv_panel(
                "Metadata",
                meta,
                x=x,
                y=y,
                w=page_w,
                key_w_ratio=0.34,  # matches your original Key/Value split
                row_h=4.8,
                title_h=6.0,
                font_size=9,
            )
            pdf.set_y(y + 6.0 + max(1, len(meta)) * 4.8 + 4.0)

        # -------------------------
        # DATASET PAGES
        # -------------------------
        for ds in report_data.get("datasets", []) or []:
            pdf.add_page()

            # Dataset title + path
            ds_name = str(ds.get("dataset_name", ""))
            ds_dir = str(ds.get("dataset_dir", ""))
            pdf.panel_title(f"Dataset: {ds_name}")
            pdf.set_font("Times", "", 10)
            if ds_dir:
                pdf.multi_cell(0, 5, ds_dir)
                pdf.ln(1.5)

            figs = ds.get("figures", {}) or {}
            perf = ds.get("perf", {}) or {}

            # --- Exploratory + Performance Summary (2x2)
            pdf.panel_title("Exploratory and Performance Summary")
            pdf.figure_grid_2x2(
                titles=[
                    "Class Balance",
                    "Missingness (Top 25)",
                    "Performance Summary / Boxplots",
                    "Models CV Distribution",
                ],
                paths=[
                    figs.get("class_balance") or None,
                    figs.get("missingness") or None,
                    figs.get("models_mean_bar") or None,
                    figs.get("models_cv_box") or None,
                ],
                cell_h=66.0,
                gap=4.0,
                title_h=6.0,
            )

            # --- Model Performance tables
            pdf.panel_title("Model Performance")

            ms = perf.get("models_mean_std", {}) or {}
            if ms.get("present"):
                pdf.subheader("Mean ± Std (Models)")
                pdf.draw_mean_std_table(ms)
            else:
                pdf.muted("Summary_performance_mean/std.csv not found.")

            med = perf.get("models_median", {}) or {}
            if med.get("present"):
                pdf.subheader("Median (Models)")
                pdf.draw_table(med.get("columns", []), med.get("rows", []), max_rows=100)
            else:
                pdf.muted("Summary_performance_median.csv not found.")

            es = perf.get("ensembles_mean_std", {}) or {}
            if es.get("present"):
                pdf.subheader("Mean ± Std (Ensembles)")
                pdf.draw_mean_std_table(es)

            em = perf.get("ensembles_median", {}) or {}
            if em.get("present"):
                pdf.subheader("Median (Ensembles)")
                pdf.draw_table(em.get("columns", []), em.get("rows", []), max_rows=100)

            # --- Curves (Summary)
            pdf.panel_title("Curves (Summary)")
            pdf.figure_row_2(
                titles=["Summary ROC", "Summary PRC"],
                paths=[
                    figs.get("models_roc_overlay") or None,
                    figs.get("models_prc_overlay") or None,
                ],
                h=78.0,
                gap=4.0,
                title_h=6.0,
            )

            if figs.get("ensembles_roc") or figs.get("ensembles_prc"):
                pdf.panel_title("Ensembles (Summary)")
                pdf.figure_row_2(
                    titles=["Ensembles ROC", "Ensembles PRC"],
                    paths=[
                        figs.get("ensembles_roc") or None,
                        figs.get("ensembles_prc") or None,
                    ],
                    h=78.0,
                    gap=4.0,
                    title_h=6.0,
                )

            # -------------------------
            # Feature Importance: ALWAYS on new page
            # -------------------------
            pdf.add_page()
            pdf.panel_title("Feature Importance")
            pdf.figure_single(
                "Top Scores / Importance",
                figs.get("fi_top20") or None,
                h=130.0,
                title_h=6.0,
            )

            # -------------------------
            # Remaining tables
            # -------------------------
            tables = ds.get("tables", {}) or {}

            pdf.panel_title("Univariate Significance (Top 10)")
            uni = tables.get("univariate_top10", {}) or {}
            if uni.get("present"):
                pdf.draw_table(uni.get("columns", []), uni.get("rows", []), max_rows=10)
            else:
                pdf.muted("Univariate_Significance.csv not found.")

            pdf.panel_title("Informative Feature Summary")
            inf = tables.get("informative_feature_summary", {}) or {}
            if inf.get("present"):
                pdf.draw_table(inf.get("columns", []), inf.get("rows", []), max_rows=200)
            else:
                pdf.muted("feature_selection/InformativeFeatureSummary.csv not found.")

            # -------------------------
            # Runtimes: ALWAYS on new page
            # -------------------------
            pdf.add_page()
            pdf.panel_title("Runtime Summary (runtimes.csv)")
            rt = tables.get("runtimes", {}) or {}
            if rt.get("present"):
                pdf.draw_table(rt.get("columns", []), rt.get("rows", []), max_rows=500)
            else:
                pdf.muted("runtimes.csv not found.")

        # -------------------------
        # DATASET COMPARISONS
        # -------------------------
        dc = report_data.get("dataset_comparisons", {}) or {}
        if dc.get("present"):
            pdf.add_page()
            pdf.panel_title("Dataset Comparisons")

            overview = (dc.get("figures", {}) or {}).get("overview")
            pdf.figure_single("Comparison Overview", overview or None, h=120.0, title_h=6.0)

            kw = (dc.get("tables", {}) or {}).get("best_kw", {}) or {}
            if kw.get("present"):
                pdf.panel_title("BestCompare_KruskalWallis.csv")
                pdf.draw_table(kw.get("columns", []), kw.get("rows", []), max_rows=200)

            mw = (dc.get("tables", {}) or {}).get("best_mw", {}) or {}
            if mw.get("present"):
                pdf.panel_title("BestCompare_MannWhitney.csv")
                pdf.draw_table(mw.get("columns", []), mw.get("rows", []), max_rows=200)

            wx = (dc.get("tables", {}) or {}).get("best_wx", {}) or {}
            if wx.get("present"):
                pdf.panel_title("BestCompare_WilcoxonRank.csv")
                pdf.draw_table(wx.get("columns", []), wx.get("rows", []), max_rows=200)

        pdf.output(str(self.paths.pdf))


    # ----------------------------
    # Runtime bookkeeping
    # ----------------------------
    def save_runtime(self):
        rt_dir = self.exp_root / "runtime"
        rt_dir.mkdir(exist_ok=True)
        (rt_dir / "runtime_report.txt").write_text(
            str(time.time() - (self.job_start_time or time.time()))
        )

    def run(self):
        self.job_start_time = time.time()

        datasets = self._list_datasets()
        if not datasets:
            raise RuntimeError(f"No dataset folders (with CVDatasets/) found under: {self.exp_root}")

        dataset_blocks = [self._collect_dataset_block(ds) for ds in datasets]
        dc_block = self._collect_dataset_comparisons_block()

        metadata = {
            "Experiment Root": str(self.exp_root),
            "Output Path": str(self.exp_root.parent),
            "Experiment Name": self.experiment_name,
        }
        if self.outcome_label:
            metadata["Outcome Label"] = self.outcome_label
        if self.instance_label:
            metadata["Instance Label"] = self.instance_label
        if self.outcome_type:
            metadata["Outcome Type"] = self.outcome_type

        report_data: Dict[str, Any] = {
            "title": self.title,
            "experiment_name": self.experiment_name,
            "experiment_root": str(self.exp_root),
            "generated_at": _now_iso_local(),
            "generated_at_epoch": int(time.time()),
            "streamline_version": _try_streamline_version(),
            "metadata": metadata,
            "datasets": dataset_blocks,
            "dataset_comparisons": dc_block,
        }

        self.paths.data_json.write_text(json.dumps(report_data, indent=2))

        if self.make_pdf:
            self._write_pdf(report_data)

        jc = self.exp_root / "jobsCompleted"
        jc.mkdir(exist_ok=True)
        (jc / "job_reporting.txt").write_text("complete")

        self.save_runtime()
        logger.info("Phase 11 reporting complete: %s", self.paths.pdf)


# ============================================================
# PDF renderer with consistent styling and layout for STREAMLINE reports
# ============================================================

class _StreamlinePDF(FPDF):
    def __init__(self, *, title: str, streamline_version: str, float_decimals: int = 3):
        super().__init__(orientation="P", unit="mm", format="A4")
        self._title = title
        self._streamline_version = streamline_version
        self._decimals = int(float_decimals)

        self.set_margins(10, 10, 10)
        self.set_auto_page_break(auto=True, margin=14)
        self.set_line_width(0.2)

        # Toggle: cover banner only on first page
        self._cover_done = False

        # Optional: small running header on non-cover pages
        self._use_running_header = False  # set True if you want a small header line

        # Table layout tuning
        self._tbl_pad_x = 1.2
        self._tbl_pad_y = 0.8
        self._tbl_line_h = 3.4

        # Spacing
        self._gap_after_cover = 4.0
        self._gap_after_section = 2.0
        # Vertical rhythm (global)
        # self._gap_after_header = 3.0
        # self._gap_after_section = 1.6
        self._gap_after_subheader = 0.8
        self._gap_after_textblock = 0.8
        self._gap_after_table = 1.2

        # Figure panel padding
        self._panel_pad = 2.0
        self._panel_title_text_pad_x = 1.4
        self._panel_title_text_pad_y = 1.4
        self._panel_content_pad_top = 2.2  # gap between title band and content

    # -----------------------------
    # Header / Footer
    # -----------------------------
    def header(self):
        # Big banner only on page 1 (cover-style)
        if self.page_no() == 1 and not self._cover_done:
            self._draw_cover_banner()
            self._cover_done = True
            return

        # No big header on subsequent pages
        if not self._use_running_header:
            return

        # Optional: small running header (thin line + short title)
        self.set_font("Times", "", 9)
        x = self.l_margin
        y = self.t_margin
        w = self.w - self.l_margin - self.r_margin
        self.set_xy(x, y - 2)
        self.line(x, y, x + w, y)
        self.set_xy(x, y)
        self.cell(w, 4, self._title, border=0, ln=1, align="L")
        self.ln(2)

    def footer(self):
        self.set_y(-10)
        self.set_font("Times", "I", 8)
        left = f"Generated with STREAMLINE ({self._streamline_version})"
        right = f"Page {self.page_no()}/{{nb}}"
        self.set_x(self.l_margin)
        self.cell(0, 5, left, border=0, ln=0, align="L")
        self.set_x(self.l_margin)
        self.cell(self.w - self.l_margin - self.r_margin, 5, right, border=0, ln=0, align="R")

    def _draw_cover_banner(self):
        """
        Mimics the reference report’s single top banner line. :contentReference[oaicite:2]{index=2}
        """
        x = self.l_margin
        y = self.t_margin
        w = self.w - self.l_margin - self.r_margin
        h = 9

        self.set_font("Times", "B", 12)
        self.set_xy(x, y)
        self.rect(x, y, w, h)
        self.cell(w, h, self._title, border=0, ln=1, align="L")
        self.ln(self._gap_after_cover)

    # -----------------------------
    # Boxed panels (STREAMLINE style)
    # -----------------------------
    def panel_title(self, title: str, *, h: float = 6.0):
        """
        Draw a boxed section heading line (like the reference panels). :contentReference[oaicite:3]{index=3}
        """
        w = self.w - self.l_margin - self.r_margin
        if self.get_y() + h + 2 > self.page_break_trigger:
            self.add_page()

        x = self.l_margin
        y = self.get_y()
        self.set_font("Times", "B", 10)
        self.rect(x, y, w, h)
        self.set_xy(x + 1.4, y + 1.4)
        self.cell(w - 2.8, h - 2.8, title, border=0, ln=1, align="L")
        self.ln(1.2)

    def kv_panel(
        self,
        title: str,
        mapping: Dict[str, Any],
        *,
        x: float,
        y: float,
        w: float,
        max_rows: Optional[int] = None,
        key_w_ratio: float = 0.55,
        row_h: float = 4.6,
        title_h: float = 6.0,
        font_size: int = 9,
    ) -> float:
        """
        Draw a boxed KV panel at (x,y) with a title band, returns panel height used.
        """
        items = [(str(k), "" if mapping.get(k) is None else str(mapping.get(k))) for k in mapping.keys()]
        if max_rows is not None:
            items = items[:max_rows]

        key_w = w * key_w_ratio
        val_w = w - key_w

        # Estimate height (title + rows)
        content_h = max(1, len(items)) * row_h
        h = title_h + content_h

        # Ensure fits page; if not, new page and reset y
        if y + h > self.page_break_trigger:
            self.add_page()
            x = self.l_margin
            y = self.get_y()
        self.set_xy(x, y)

        # Outer box + title band
        self.rect(x, y, w, h)
        self.rect(x, y, w, title_h)

        self.set_font("Times", "B", 9)
        self.set_xy(x + 1.2, y + 1.4)
        self.cell(w - 2.4, title_h - 2.8, title, border=0, ln=0, align="L")

        # Rows
        self.set_font("Times", "", font_size)
        cy = y + title_h
        for k, v in items:
            # row border
            self.rect(x, cy, w, row_h)
            # vertical split
            self.line(x + key_w, cy, x + key_w, cy + row_h)

            self.set_xy(x + 1.2, cy + 1.0)
            self.cell(key_w - 2.4, row_h - 2.0, k, border=0, ln=0, align="L")

            self.set_xy(x + key_w + 1.2, cy + 1.0)
            self.cell(val_w - 2.4, row_h - 2.0, v, border=0, ln=0, align="L")

            cy += row_h

        return h

    def cover_header_block(self, *, experiment: str, generated_at: str, version: str):
        """
        Second box on cover like the reference “report header block”. :contentReference[oaicite:4]{index=4}
        """
        w = self.w - self.l_margin - self.r_margin
        x = self.l_margin
        y = self.get_y()

        lines = [
            f"Experiment: {experiment}",
            f"Generated at: {generated_at}",
            f"Report Version: {version}",
        ]

        # height: title line + lines
        h = 8 + len(lines) * 5 + 3
        self.rect(x, y, w, h)

        self.set_font("Times", "B", 12)
        self.set_xy(x + 2, y + 2)
        self.cell(w - 4, 6, "STREAMLINE Evaluation Report", ln=1)

        self.set_font("Times", "", 10)
        self.set_x(x + 2)
        for ln_ in lines:
            self.multi_cell(w - 4, 5, ln_)
        self.ln(2.0)


    # -----------------------------
    # Blocks / Typography
    # -----------------------------
    def section_bar(self, text: str):
        w = self.w - self.l_margin - self.r_margin
        h = 6
        if self.get_y() + h + 2 > self.page_break_trigger:
            self.add_page()

        x = self.l_margin
        y = self.get_y()

        self.set_font("Times", "B", 10)
        self.rect(x, y, w, h)

        self.set_xy(x + 1.4, y + 1.4)
        self.cell(w - 2.8, h - 2.8, text, border=0, ln=1, align="L")
        self.ln(self._gap_after_section)

    def subheader(self, text: str):
        self.set_font("Times", "B", 10)
        self.multi_cell(0, 5, text)
        self.ln(self._gap_after_subheader)

    def muted(self, text: str):
        self.set_font("Times", "", 9)
        self.multi_cell(0, 4.5, text)
        self.ln(self._gap_after_textblock)

    def card_header(self, *, main_title: str, lines: Sequence[str]):
        w = self.w - self.l_margin - self.r_margin
        x = self.l_margin
        y = self.get_y()
        h = 8 + len(lines) * 5 + 4

        self.rect(x, y, w, h)
        self.set_font("Times", "B", 12)
        self.set_xy(x + 2, y + 2)
        self.cell(w - 4, 6, main_title, ln=1)

        self.set_font("Times", "", 10)
        self.set_x(x + 2)
        for ln_ in lines:
            self.multi_cell(w - 4, 5, ln_)
        self.ln(2.0)

    # -----------------------------
    # Formatting
    # -----------------------------
    def _cell_str(self, v: Any) -> str:
        return format_number(v, decimals=self._decimals)

    # -----------------------------
    # Wrapping (robust; prevents overflow artifacts)
    # -----------------------------
    def _split_lines(self, txt: str, width_mm: float, line_h: float) -> List[str]:
        """
        Get the exact wrapped lines that FPDF will use for multi_cell().
        Uses fpdf2's split_only=True when available.
        """
        s = "" if txt is None else str(txt)
        usable_w = max(1e-6, float(width_mm))

        try:
            lines = self.multi_cell(usable_w, line_h, s, border=0, align="L", split_only=True)
            return [ln if ln is not None else "" for ln in lines] or [""]
        except TypeError:
            # Fallback: conservative estimate (older fpdf)
            parts = s.splitlines() or [s]
            out: List[str] = []
            for part in parts:
                sw = self.get_string_width(part) if part else 0.0
                n = max(1, int(math.ceil(sw / usable_w)))
                out.extend([""] * n)
            return out or [""]

    def _wrapped_line_count(self, txt: str, width_mm: float, line_h: float) -> int:
        return len(self._split_lines(txt, width_mm, line_h))

    # -----------------------------
    # Tables
    # -----------------------------
    def draw_kv_table(self, mapping: Dict[str, Any]):
        cols = ["Key", "Value"]
        rows = [[k, mapping.get(k)] for k in mapping.keys()]
        table_w = self.w - self.l_margin - self.r_margin
        self.draw_table(
            cols,
            rows,
            col_widths=[table_w * 0.34, table_w * 0.66],
            font_size=9,
        )

    def _col_widths_model_perf(self, columns: Sequence[str], table_w: float) -> List[float]:
        """
        Tuned widths for wide model-performance tables:

        - Cap the first column ("Algorithm") so it doesn't steal half the page.
        - Allocate metric columns by weighted header length (long headers get more width).
        - Enforce a minimum metric width so mean±std values stay on one line.
        """
        n = len(columns)
        if n <= 1:
            return [table_w]

        # Algorithm column: readable, but capped
        first = min(max(38.0, 0.20 * table_w), 52.0)

        metric_cols = list(columns[1:])
        weights: List[float] = []
        for c in metric_cols:
            cl = c.lower()
            w = 1.0 + min(1.2, len(c) / 18.0)  # length influence
            if "sensitivity" in cl or "precision" in cl:
                w *= 1.35
            if "balanced" in cl:
                w *= 1.15
            if "brier" in cl:
                w *= 1.10
            weights.append(w)

        rest = max(0.0, table_w - first)
        sw = sum(weights) if sum(weights) > 0 else float(len(weights))
        raw = [rest * (w / sw) for w in weights]

        # Prevent very narrow metric columns
        min_metric = 16.0
        raw = [max(min_metric, r) for r in raw]

        # Renormalize to fit `rest`
        s2 = sum(raw)
        scale = (rest / s2) if s2 > 0 else 1.0
        metrics = [r * scale for r in raw]

        widths = [first] + metrics
        widths[-1] += (table_w - sum(widths))  # rounding fix
        return widths


    # Replace draw_mean_std_table with this version (inside _StreamlinePDF)

    def draw_mean_std_table(self, mean_std: Dict[str, Any]):
        cols = mean_std.get("columns", [])
        rows_out: List[List[Any]] = []

        for row in mean_std.get("rows", []) or []:
            out_row: List[Any] = []
            for cell in row.get("cells", []):
                val = cell.get("value")
                if isinstance(val, tuple) and len(val) == 2:
                    mv, sv = val
                    # Non-breaking spaces prevent wrap at " ± "
                    out_row.append(f"{self._cell_str(mv)}\u00A0±\u00A0{self._cell_str(sv)}")
                else:
                    out_row.append(val)
            rows_out.append(out_row)

        table_w = self.w - self.l_margin - self.r_margin

        # Use tuned widths for performance tables
        col_widths = self._col_widths_model_perf(cols, table_w)

        # Wide table: smaller font and slightly reduced horizontal padding
        old_pad = self._tbl_pad_x
        self._tbl_pad_x = 0.8

        ncol = len(cols)
        font_size = 6 if ncol >= 9 else 7

        self.draw_table(cols, rows_out, col_widths=col_widths, font_size=font_size)

        # restore padding
        self._tbl_pad_x = old_pad
        

    def _col_widths_wide_metrics(self, columns: Sequence[str], table_w: float) -> List[float]:
        """
        Wide metric tables: give first column more room, distribute rest evenly.
        """
        n = len(columns)
        if n <= 1:
            return [table_w]

        first = max(48.0, table_w * 0.24)
        rest = max(0.0, table_w - first)
        per = rest / (n - 1)

        widths = [first] + [per] * (n - 1)
        widths[-1] += (table_w - sum(widths))
        return widths

    def _auto_col_widths(self, columns: Sequence[str], rows: Sequence[Sequence[str]], table_w: float) -> List[float]:
        n = len(columns)
        if n == 1:
            return [table_w]

        weights: List[float] = []
        for j, c in enumerate(columns):
            w = max(3.0, float(len(str(c))))
            for r in rows[:15]:
                if j < len(r):
                    w = max(w, min(44.0, float(len(r[j]))))
            if j == 0:
                w *= 1.8
            weights.append(w)

        sw = sum(weights) if sum(weights) > 0 else float(n)
        raw = [table_w * (w / sw) for w in weights]

        min_w = 14.0 if n <= 4 else 10.0
        raw = [max(min_w, w) for w in raw]

        s2 = sum(raw)
        scale = (table_w / s2) if s2 > 0 else 1.0
        out = [w * scale for w in raw]
        out[-1] += (table_w - sum(out))
        return out

    def draw_table(
        self,
        columns: Sequence[str],
        rows: Sequence[Sequence[Any]],
        *,
        col_widths: Optional[Sequence[float]] = None,
        max_rows: Optional[int] = None,
        font_size: Optional[int] = None,
    ):
        if not columns:
            self.muted("No table columns.")
            return

        table_w = self.w - self.l_margin - self.r_margin
        ncol = len(columns)

        formatted_rows: List[List[str]] = [[self._cell_str(v) for v in r] for r in rows]
        if max_rows is not None:
            formatted_rows = formatted_rows[:max_rows]

        if font_size is None:
            font_size = 8 if ncol <= 5 else (7 if ncol <= 8 else (6 if ncol <= 11 else 5))
        self.set_font("Times", "", font_size)

        if col_widths is None:
            col_widths = self._auto_col_widths(columns, formatted_rows, table_w)
        col_widths = list(col_widths)

        header_h = 5.0
        line_h = self._tbl_line_h

        def draw_header():
            self.set_font("Times", "B", font_size)
            y0 = self.get_y()
            x0 = self.l_margin

            for j, col in enumerate(columns):
                wj = col_widths[j]
                self.rect(x0, y0, wj, header_h)
                self.set_xy(x0 + self._tbl_pad_x, y0 + self._tbl_pad_y)
                self.cell(
                    wj - 2 * self._tbl_pad_x,
                    header_h - 2 * self._tbl_pad_y,
                    str(col),
                    border=0,
                    ln=0,
                    align="C",
                )
                x0 += wj

            # Exact baseline; avoids header/body gaps
            self.set_xy(self.l_margin, y0 + header_h)
            self.set_font("Times", "", font_size)

        def row_height(cells: Sequence[str]) -> float:
            counts: List[int] = []
            for j, txt in enumerate(cells):
                usable_w = col_widths[j] - 2 * self._tbl_pad_x
                counts.append(self._wrapped_line_count(txt, usable_w, line_h))
            return max(counts) * line_h + 2 * self._tbl_pad_y

        if self.get_y() + header_h + 2 > self.page_break_trigger:
            self.add_page()

        draw_header()

        for cells in formatted_rows:
            rh = row_height(cells)
            if self.get_y() + rh > self.page_break_trigger:
                self.add_page()
                draw_header()

            y0 = self.get_y()
            x0 = self.l_margin

            for j, txt in enumerate(cells):
                wj = col_widths[j]
                self.rect(x0, y0, wj, rh)
                self.set_xy(x0 + self._tbl_pad_x, y0 + self._tbl_pad_y)

                # Use the same wrapping behavior we measured
                self.multi_cell(wj - 2 * self._tbl_pad_x, line_h, txt, border=0, align="L")

                x0 += wj
                self.set_xy(x0, y0)

            self.set_y(y0 + rh)

        self.ln(self._gap_after_table)

    # -----------------------------
    # Figures
    # -----------------------------
    def _image_panel(self, title: str, path: Optional[str], *, x: float, y: float, w: float, h: float, title_h: float):
        self.rect(x, y, w, h)
        self.rect(x, y, w, title_h)

        self.set_font("Times", "B", 8)
        self.set_xy(x + self._panel_title_text_pad_x, y + self._panel_title_text_pad_y)
        self.cell(
            w - 2 * self._panel_title_text_pad_x,
            title_h - 2 * self._panel_title_text_pad_y,
            title,
            border=0,
            ln=0,
            align="L",
        )

        inner_x = x + self._panel_pad
        inner_y = y + title_h + self._panel_content_pad_top
        inner_w = w - 2 * self._panel_pad
        inner_h = h - title_h - (self._panel_content_pad_top + self._panel_pad)

        if inner_w < 5 or inner_h < 5:
            self.set_font("Times", "", 8)
            self.set_xy(x + 1, y + title_h + 1)
            self.multi_cell(w - 2, 3.5, "Panel too small.", border=0)
            return

        if path and Path(path).exists():
            try:
                self.image(path, x=inner_x, y=inner_y, w=inner_w, h=inner_h)
            except Exception:
                self.set_font("Times", "", 8)
                self.set_xy(inner_x, inner_y)
                self.multi_cell(inner_w, 3.5, "Could not render image.", border=0)
        else:
            self.set_font("Times", "", 8)
            self.set_xy(inner_x, inner_y)
            self.multi_cell(inner_w, 3.5, "Figure missing.", border=0)

    def figure_grid_2x2(
        self,
        *,
        titles: Sequence[str],
        paths: Sequence[Optional[str]],
        cell_h: float = 66.0,
        gap: float = 4.0,
        title_h: float = 6.0,
    ):
        page_w = self.w - self.l_margin - self.r_margin
        cell_w = (page_w - gap) / 2.0

        x0 = self.l_margin
        y0 = self.get_y()

        needed_h = cell_h * 2 + gap + 2
        if y0 + needed_h > self.page_break_trigger:
            self.add_page()
            y0 = self.get_y()

        self._image_panel(titles[0], paths[0] if len(paths) > 0 else None, x=x0, y=y0, w=cell_w, h=cell_h, title_h=title_h)
        self._image_panel(titles[1], paths[1] if len(paths) > 1 else None, x=x0 + cell_w + gap, y=y0, w=cell_w, h=cell_h, title_h=title_h)

        y1 = y0 + cell_h + gap
        self._image_panel(titles[2], paths[2] if len(paths) > 2 else None, x=x0, y=y1, w=cell_w, h=cell_h, title_h=title_h)
        self._image_panel(titles[3], paths[3] if len(paths) > 3 else None, x=x0 + cell_w + gap, y=y1, w=cell_w, h=cell_h, title_h=title_h)

        self.set_y(y1 + cell_h + 2)

    def figure_row_2(
        self,
        *,
        titles: Sequence[str],
        paths: Sequence[Optional[str]],
        h: float = 80.0,
        gap: float = 4.0,
        title_h: float = 6.0,
    ):
        page_w = self.w - self.l_margin - self.r_margin
        cell_w = (page_w - gap) / 2.0
        y0 = self.get_y()
        if y0 + h + 2 > self.page_break_trigger:
            self.add_page()
            y0 = self.get_y()

        x0 = self.l_margin
        self._image_panel(titles[0], paths[0] if len(paths) > 0 else None, x=x0, y=y0, w=cell_w, h=h, title_h=title_h)
        self._image_panel(titles[1], paths[1] if len(paths) > 1 else None, x=x0 + cell_w + gap, y=y0, w=cell_w, h=h, title_h=title_h)
        self.set_y(y0 + h + 2)

    def figure_single(self, title: str, path: Optional[str], *, h: float = 90.0, title_h: float = 6.0):
        page_w = self.w - self.l_margin - self.r_margin
        y0 = self.get_y()
        if y0 + h + 2 > self.page_break_trigger:
            self.add_page()
            y0 = self.get_y()
        self._image_panel(title, path, x=self.l_margin, y=y0, w=page_w, h=h, title_h=title_h)
        self.set_y(y0 + h + 2)
