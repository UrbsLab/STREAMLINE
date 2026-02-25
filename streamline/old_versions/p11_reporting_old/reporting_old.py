from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML  # noqa: F401


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


def _rel_to_reporting(reporting_dir: Path, p: Path) -> str:
    """
    WeasyPrint base_url is reporting_dir, so all src= should be relative to it.
    """
    try:
        return str(p.relative_to(reporting_dir))
    except Exception:
        return str(p)


@dataclass
class ReportPaths:
    reporting_dir: Path
    data_json: Path
    html: Path
    pdf: Path
    figures_dir: Path


class ReportPhaseJob:
    """
    Phase 11: Reporting

    Generates a multi-page HTML report (Jinja2) and prints to PDF (WeasyPrint),
    with multi-image mosaics per page.

    IMPORTANT: This version does NOT try to locate “legacy” plots.
    It replots from available CSV/JSON artifacts and exports PNGs for the report.
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

        self.paths = self._init_paths()

    def _init_paths(self) -> ReportPaths:
        reporting_dir = self.exp_root / "reporting"
        figures_dir = reporting_dir / "figures"
        reporting_dir.mkdir(exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        return ReportPaths(
            reporting_dir=reporting_dir,
            data_json=reporting_dir / "report_data.json",
            html=reporting_dir / "report.html",
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

    # ----------------------------
    # Plot builders (replot everything)
    # ----------------------------
    def _plot_model_summary_bars(self, summary_mean: pd.DataFrame, metric: str, title: str):
        import plotly.express as px  # type: ignore

        df = summary_mean.copy()
        # rehydrate index if needed
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

    def _extract_xy_from_curve_blob(
        self, blob: Dict[str, Any], curve_kind: str
    ) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        if curve_kind == "roc":
            for a, b in [("fpr", "tpr"), ("x", "y")]:
                if a in blob and b in blob:
                    return list(map(float, blob[a])), list(map(float, blob[b]))
        else:
            for a, b in [("recall", "precision"), ("x", "y")]:
                if a in blob and b in blob:
                    return list(map(float, blob[a])), list(map(float, blob[b]))
        return None, None

    def _plot_roc_prc_from_curve_json(self, curves_dir: Path, curve_kind: str, title: str):
        import plotly.graph_objects as go  # type: ignore

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for fn in sorted(curves_dir.glob(f"*_{curve_kind}.json")):
            alg = fn.name.split("_CV_")[0]
            blob = self._read_json_if_exists(fn)
            if not blob:
                continue
            groups.setdefault(alg, []).append(blob)

        if not groups:
            raise RuntimeError(f"No curve json found for {curve_kind} under {curves_dir}")

        fig = go.Figure()
        for alg, blobs in groups.items():
            for i, b in enumerate(blobs):
                x, y = self._extract_xy_from_curve_blob(b, curve_kind)
                if x is None or y is None:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        name=f"{alg}",
                        showlegend=(i == 0),
                        opacity=0.30 if i > 0 else 0.90,
                    )
                )

        xlab = "False Positive Rate" if curve_kind == "roc" else "Recall"
        ylab = "True Positive Rate" if curve_kind == "roc" else "Precision"
        fig.update_layout(title=title, xaxis_title=xlab, yaxis_title=ylab)
        return fig

    def _plot_class_counts(self, class_counts: pd.DataFrame, title: str):
        """
        Best-effort class balance bar plot from exploratory/ClassCounts.csv
        """
        import plotly.express as px  # type: ignore

        df = class_counts.copy()
        # Common layouts vary; try to find label/count columns
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
            # if two columns, assume second is count
            count_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        df = df[[label_col, count_col]].dropna()
        df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
        df = df.dropna()
        fig = px.bar(df, x=label_col, y=count_col, title=title)
        fig.update_layout(xaxis_title="Class", yaxis_title="Count")
        return fig

    def _plot_missingness(self, missingness: pd.DataFrame, title: str):
        """
        Best-effort missingness bar plot from exploratory/DataMissingness.csv
        """
        import plotly.express as px  # type: ignore

        df = missingness.copy()
        # likely columns: Feature, MissingCount or MissingPercent
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

    # ----------------------------
    # Formatting helpers (tables)
    # ----------------------------
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
                    val = str(r["Algorithm"])
                    cells.append({"value": val, "best": bool(best_alg and val == best_alg)})
                else:
                    mv = r.get(f"{c}_mean", None)
                    sv = r.get(f"{c}_std", None)
                    try:
                        mvf = float(mv)
                        svf = float(sv) if sv is not None else float("nan")
                        val = f"{mvf:.3f} ± {svf:.3f}"
                    except Exception:
                        val = str(mv)
                    cells.append({"value": val, "best": False})
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
        cols = list(df.columns)
        rows = df.head(max_rows).fillna("").astype(str).values.tolist()
        return {"present": True, "columns": cols, "rows": rows}

    # ----------------------------
    # Data assembly (and plot export)
    # ----------------------------
    def _collect_dataset_block(self, ds_dir: Path) -> Dict[str, Any]:
        name = ds_dir.name
        figs: Dict[str, str] = {}

        # Phase 1 tables
        explore = ds_dir / "exploratory"
        data_process_summary = self._read_csv_if_exists(explore / "DataProcessSummary.csv")
        class_counts = self._read_csv_if_exists(explore / "ClassCounts.csv")
        missingness = self._read_csv_if_exists(explore / "DataMissingness.csv")
        univariate = self._read_csv_if_exists(explore / "univariate_analyses" / "Univariate_Significance.csv")
        univariate_top10 = univariate.head(10) if (univariate is not None and not univariate.empty) else None

        # Phase 8 model evaluation tables + sources for plots
        model_eval = ds_dir / "model_evaluation"
        summary_mean = self._read_csv_if_exists(model_eval / "Summary_performance_mean.csv")
        summary_std = self._read_csv_if_exists(model_eval / "Summary_performance_std.csv")
        summary_median = self._read_csv_if_exists(model_eval / "Summary_performance_median.csv")

        # Ensemble tables
        ens_eval = ds_dir / "ensemble_evaluation"
        ens_mean = self._read_csv_if_exists(ens_eval / "Ensembles_performance_mean.csv")
        ens_std = self._read_csv_if_exists(ens_eval / "Ensembles_performance_std.csv")
        ens_median = self._read_csv_if_exists(ens_eval / "Ensembles_performance_median.csv")

        # Feature selection
        feat_sel = self._read_csv_if_exists(ds_dir / "feature_selection" / "InformativeFeatureSummary.csv")

        # Runtimes
        runtimes = self._read_csv_if_exists(ds_dir / "runtimes.csv")

        # -----------------
        # Replot figures
        # -----------------
        # Exploratory: class counts + missingness
        if class_counts is not None and not class_counts.empty:
            try:
                fig = self._plot_class_counts(class_counts, f"{name}: Class Balance")
                out = self.paths.figures_dir / f"{name}_class_balance.png"
                if _safe_plotly_to_png(fig, out):
                    figs["class_balance"] = _rel_to_reporting(self.paths.reporting_dir, out)
            except Exception as e:
                logger.warning("Class balance plot failed for %s: %r", name, e)

        if missingness is not None and not missingness.empty:
            try:
                fig = self._plot_missingness(missingness, f"{name}: Missingness (Top 25)")
                out = self.paths.figures_dir / f"{name}_missingness_top25.png"
                if _safe_plotly_to_png(fig, out):
                    figs["missingness"] = _rel_to_reporting(self.paths.reporting_dir, out)
            except Exception as e:
                logger.warning("Missingness plot failed for %s: %r", name, e)

        # Model mean bar (pick best available metric)
        chosen_metric = None
        if summary_mean is not None and not summary_mean.empty:
            for preferred in ["Balanced Accuracy", "ROC AUC", "PRC AUC", "Accuracy"]:
                if preferred in summary_mean.columns:
                    chosen_metric = preferred
                    try:
                        fig = self._plot_model_summary_bars(
                            summary_mean, preferred, f"{name}: Mean {preferred} (Models)"
                        )
                        out = self.paths.figures_dir / f"{name}_models_mean_{preferred.replace(' ', '_')}.png"
                        if _safe_plotly_to_png(fig, out):
                            figs["models_mean_bar"] = _rel_to_reporting(self.paths.reporting_dir, out)
                            figs["models_mean_metric"] = preferred
                    except Exception as e:
                        logger.warning("Model mean bar plot failed for %s (%s): %r", name, preferred, e)
                    break

        # CV distribution boxplot
        mbc = model_eval / "metrics_by_cv"
        if mbc.is_dir():
            for preferred in ["Balanced Accuracy", "ROC AUC", "PRC AUC"]:
                try:
                    fig = self._plot_cv_metric_distribution(
                        mbc, preferred, f"{name}: CV Distribution ({preferred}) - Models"
                    )
                    out = self.paths.figures_dir / f"{name}_models_cv_box_{preferred.replace(' ', '_')}.png"
                    if _safe_plotly_to_png(fig, out):
                        figs["models_cv_box"] = _rel_to_reporting(self.paths.reporting_dir, out)
                        figs["models_cv_metric"] = preferred
                        break
                except Exception:
                    continue

        # ROC/PRC overlays from curves_by_cv
        curves = model_eval / "curves_by_cv"
        if curves.is_dir():
            for kind in ["roc", "prc"]:
                try:
                    fig = self._plot_roc_prc_from_curve_json(curves, kind, f"{name}: {kind.upper()} Curves (Models)")
                    out = self.paths.figures_dir / f"{name}_models_{kind}_overlay.png"
                    if _safe_plotly_to_png(fig, out):
                        figs[f"models_{kind}_overlay"] = _rel_to_reporting(self.paths.reporting_dir, out)
                except Exception:
                    pass

        # Feature importance top-20 (from *_FI.csv)
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
                    figs["fi_top20"] = _rel_to_reporting(self.paths.reporting_dir, out)
            except Exception as e:
                logger.warning("FI top20 plot failed for %s: %r", name, e)

        # -----------------
        # Tables (classic)
        # -----------------
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

        # Replot: KW p-values bar if possible
        if best_kw is not None and not best_kw.empty:
            try:
                import plotly.express as px  # type: ignore

                df = best_kw.copy()
                # try to find p-value column
                low = {c.lower(): c for c in df.columns}
                pcol = None
                for cand in ["p-value", "p_value", "pvalue", "p"]:
                    if cand in low:
                        pcol = low[cand]
                        break
                if pcol is None:
                    # if only one column, treat it as p
                    if df.shape[1] == 1:
                        pcol = df.columns[0]
                if pcol is not None:
                    df = df[[pcol]].copy()
                    df[pcol] = pd.to_numeric(df[pcol], errors="coerce")
                    df = df.dropna().reset_index(drop=True)
                    df["Metric"] = [f"M{i+1}" for i in range(len(df))]
                    fig = px.bar(df, x="Metric", y=pcol, title="DatasetComparison: Kruskal-Wallis P-Values")
                    out = self.paths.figures_dir / "datasetcompare_kw_pvalues.png"
                    if _safe_plotly_to_png(fig, out):
                        figs["kw_pvalues"] = _rel_to_reporting(self.paths.reporting_dir, out)
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

    # ----------------------------
    # Render
    # ----------------------------
    def _render_html(self, report_data: Dict[str, Any]) -> str:
        templates_dir = Path(__file__).with_name("templates")
        env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        tpl = env.get_template("report.html.j2")
        return tpl.render(**report_data)

    def _write_pdf(self, html_text: str):
        base_url = str(self.paths.reporting_dir)
        HTML(string=html_text, base_url=base_url).write_pdf(str(self.paths.pdf))

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

        html_text = self._render_html(report_data)
        self.paths.html.write_text(html_text, encoding="utf-8")

        if self.make_pdf:
            self._write_pdf(html_text)

        jc = self.exp_root / "jobsCompleted"
        jc.mkdir(exist_ok=True)
        (jc / "job_reporting.txt").write_text("complete")

        self.save_runtime()
        logger.info("Phase 11 reporting complete: %s", self.paths.pdf)
