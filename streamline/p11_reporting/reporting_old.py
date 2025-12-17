# streamline/p11_reporting/reporting.py
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Jinja2 + WeasyPrint
from jinja2 import Environment, FileSystemLoader, select_autoescape

from weasyprint import HTML  # noqa: F401

# Plotly (export to PNG for PDF)
def _safe_plotly_to_png(fig, out_path: Path, scale: int = 2) -> bool:
    """
    Try to export plotly fig to PNG (kaleido). Return True if success.
    """
    try:
        import plotly.io as pio  # type: ignore

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Use write_image to embed into HTML/PDF (WeasyPrint cannot render JS).
        pio.write_image(fig, str(out_path), format="png", scale=scale)
        return True
    except Exception as e:
        logger.warning("Plotly export failed for %s: %r", out_path, e)
        return False


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

    Generates a multi-page, page-broken HTML report (Jinja2) and prints to PDF
    (WeasyPrint), aggregating outputs from all phases.

    Expected experiment structure:
      <output_path>/<experiment_name>/
        <dataset_1>/
          exploratory/, impute_scale/, feature_learning/, feature_importance/,
          feature_selection/, model_evaluation/, ensemble_evaluation/, runtime/, ...
        <dataset_2>/
          ...
        DatasetComparisons/
        reporting/
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
        self.show_plots = False
        self.make_pdf = make_pdf
        self.job_start_time: Optional[float] = None

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
        """
        Dataset folders are directories that contain CVDatasets/.
        """
        ignore = {"jobs", "logs", "jobsCompleted", "dask_logs", "runtime", "DatasetComparisons", "reporting"}
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
    # Plot builders (NEW plots)
    # ----------------------------
    def _plot_model_summary_bars(
        self, summary_mean: pd.DataFrame, metric: str, title: str
    ):
        import plotly.express as px  # type: ignore

        df = summary_mean.copy()
        # expected layout: index col first; guard for both styles
        if df.columns[0].lower() in {"unnamed: 0", "model", "algorithm"}:
            # already has name column
            name_col = df.columns[0]
        else:
            # some Summary files are index-based saved with index_col=0; rehydrate
            if df.index.name is None or df.index.dtype != object:
                df = df.reset_index()
            name_col = df.columns[0]

        if metric not in df.columns:
            raise KeyError(metric)

        df = df[[name_col, metric]].dropna()
        df = df.sort_values(metric, ascending=False)

        fig = px.bar(df, x=name_col, y=metric, title=title)
        fig.update_layout(xaxis_title="Model", yaxis_title=metric)
        return fig

    def _plot_cv_metric_distribution(
        self, metrics_by_cv_dir: Path, metric: str, title: str
    ):
        """
        Reads per-CV JSON metrics and makes a boxplot over CV folds per model.
        """
        import plotly.express as px  # type: ignore

        rows = []
        for fn in sorted(metrics_by_cv_dir.glob("*.json")):
            blob = self._read_json_if_exists(fn)
            if not blob:
                continue
            # file pattern: <ALG>_CV_<k>.json
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

    def _plot_roc_prc_from_curve_json(self, curves_dir: Path, curve_kind: str, title: str):
        """
        curve_kind: "roc" or "prc"
        Expects files like: <ALG>_CV_<k>_roc.json or _prc.json
        The JSON schema varies by implementation; we do best-effort:
          - tries keys: fpr/tpr, recall/precision, x/y
        Produces mean curve across folds per algorithm (simple interpolation-free overlay + legend).
        """
        import plotly.graph_objects as go  # type: ignore

        # group by algorithm
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for fn in sorted(curves_dir.glob(f"*_{curve_kind}.json")):
            alg = fn.name.split("_CV_")[0]
            blob = self._read_json_if_exists(fn)
            if not blob:
                continue
            groups.setdefault(alg, []).append(blob)

        fig = go.Figure()
        for alg, blobs in groups.items():
            # overlay folds lightly, and add one bold line for the first fold (simple + robust)
            for i, b in enumerate(blobs):
                x, y = self._extract_xy_from_curve_blob(b, curve_kind)
                if x is None or y is None:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        name=f"{alg} (CV{i})",
                        showlegend=(i == 0),
                        opacity=0.35 if i > 0 else 0.9,
                    )
                )

        xlab = "False Positive Rate" if curve_kind == "roc" else "Recall"
        ylab = "True Positive Rate" if curve_kind == "roc" else "Precision"
        fig.update_layout(title=title, xaxis_title=xlab, yaxis_title=ylab)
        return fig

    def _extract_xy_from_curve_blob(self, blob: Dict[str, Any], curve_kind: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        # common options
        if curve_kind == "roc":
            for a, b in [("fpr", "tpr"), ("x", "y")]:
                if a in blob and b in blob:
                    return list(map(float, blob[a])), list(map(float, blob[b]))
        else:
            for a, b in [("recall", "precision"), ("x", "y")]:
                if a in blob and b in blob:
                    return list(map(float, blob[a])), list(map(float, blob[b]))
        return None, None

    # ----------------------------
    # Data assembly
    # ----------------------------
    def _collect_dataset_block(self, ds_dir: Path) -> Dict[str, Any]:
        name = ds_dir.name

        # Phase 8 summaries (regular models)
        model_eval = ds_dir / "model_evaluation"
        summary_mean = self._read_csv_if_exists(model_eval / "Summary_performance_mean.csv")
        summary_median = self._read_csv_if_exists(model_eval / "Summary_performance_median.csv")
        summary_std = self._read_csv_if_exists(model_eval / "Summary_performance_std.csv")

        # Phase 7 ensembles (optional)
        ens_eval = ds_dir / "ensemble_evaluation"
        ens_mean = self._read_csv_if_exists(ens_eval / "Ensembles_performance_mean.csv")
        ens_median = self._read_csv_if_exists(ens_eval / "Ensembles_performance_median.csv")
        ens_std = self._read_csv_if_exists(ens_eval / "Ensembles_performance_std.csv")

        # Feature selection summary (phase 5)
        feat_sel = self._read_csv_if_exists(ds_dir / "feature_selection" / "InformativeFeatureSummary.csv")

        # Exploratory summaries (phase 1)
        explore = ds_dir / "exploratory"
        missingness = self._read_csv_if_exists(explore / "DataMissingness.csv")
        class_counts = self._read_csv_if_exists(explore / "ClassCounts.csv")
        univariate = self._read_csv_if_exists(explore / "univariate_analyses" / "Univariate_Significance.csv")

        # FI (phase 6/8 outputs often in model_evaluation/feature_importance/*.csv)
        fi_dir = model_eval / "feature_importance"
        fi_files = sorted(fi_dir.glob("*_FI.csv")) if fi_dir.is_dir() else []

        # Runtimes (phase runtime files exist under ds_dir/runtime or exp_root/runtime)
        runtimes_csv = ds_dir / "runtimes.csv"
        runtimes = self._read_csv_if_exists(runtimes_csv)

        # Build new figures (saved as PNG)
        figs: Dict[str, str] = {}  # key->relative path

        # Regular model bar chart for a key metric if present
        if summary_mean is not None and not summary_mean.empty:
            for preferred in ["Balanced Accuracy", "ROC AUC", "PRC AUC", "Accuracy"]:
                if preferred in summary_mean.columns:
                    fig = self._plot_model_summary_bars(
                        summary_mean, preferred, f"{name}: Mean {preferred} (Models)"
                    )
                    out = self.paths.figures_dir / f"{name}_models_mean_{preferred.replace(' ', '_')}.png"
                    if _safe_plotly_to_png(fig, out):
                        figs[f"models_mean_{preferred}"] = str(out.relative_to(self.paths.reporting_dir))
                    break

            # distribution across folds via metrics_by_cv JSON (if present)
            mbc = model_eval / "metrics_by_cv"
            if mbc.is_dir():
                for preferred in ["Balanced Accuracy", "ROC AUC", "PRC AUC"]:
                    try:
                        fig = self._plot_cv_metric_distribution(
                            mbc, preferred, f"{name}: CV Distribution ({preferred}) - Models"
                        )
                        out = self.paths.figures_dir / f"{name}_models_cv_box_{preferred.replace(' ', '_')}.png"
                        if _safe_plotly_to_png(fig, out):
                            figs[f"models_cv_{preferred}"] = str(out.relative_to(self.paths.reporting_dir))
                            break
                    except Exception:
                        pass

            # ROC/PRC overlay (models) from curves_by_cv
            curves = model_eval / "curves_by_cv"
            if curves.is_dir():
                for kind in ["roc", "prc"]:
                    try:
                        fig = self._plot_roc_prc_from_curve_json(
                            curves, kind, f"{name}: {kind.upper()} Curves (Models)"
                        )
                        out = self.paths.figures_dir / f"{name}_models_{kind}_overlay.png"
                        if _safe_plotly_to_png(fig, out):
                            figs[f"models_{kind}_overlay"] = str(out.relative_to(self.paths.reporting_dir))
                    except Exception:
                        pass

        # Ensemble plots (if present)
        if ens_mean is not None and not ens_mean.empty:
            for preferred in ["Balanced Accuracy", "ROC AUC", "PRC AUC", "Accuracy"]:
                if preferred in ens_mean.columns:
                    fig = self._plot_model_summary_bars(
                        ens_mean, preferred, f"{name}: Mean {preferred} (Ensembles)"
                    )
                    out = self.paths.figures_dir / f"{name}_ensembles_mean_{preferred.replace(' ', '_')}.png"
                    if _safe_plotly_to_png(fig, out):
                        figs[f"ensembles_mean_{preferred}"] = str(out.relative_to(self.paths.reporting_dir))
                    break

            embc = ens_eval / "metrics_by_cv"
            if embc.is_dir():
                for preferred in ["Balanced Accuracy", "ROC AUC", "PRC AUC"]:
                    try:
                        fig = self._plot_cv_metric_distribution(
                            embc, preferred, f"{name}: CV Distribution ({preferred}) - Ensembles"
                        )
                        out = self.paths.figures_dir / f"{name}_ensembles_cv_box_{preferred.replace(' ', '_')}.png"
                        if _safe_plotly_to_png(fig, out):
                            figs[f"ensembles_cv_{preferred}"] = str(out.relative_to(self.paths.reporting_dir))
                            break
                    except Exception:
                        pass

            ecurves = ens_eval / "curves_by_cv"
            if ecurves.is_dir():
                for kind in ["roc", "prc"]:
                    try:
                        fig = self._plot_roc_prc_from_curve_json(
                            ecurves, kind, f"{name}: {kind.upper()} Curves (Ensembles)"
                        )
                        out = self.paths.figures_dir / f"{name}_ensembles_{kind}_overlay.png"
                        if _safe_plotly_to_png(fig, out):
                            figs[f"ensembles_{kind}_overlay"] = str(out.relative_to(self.paths.reporting_dir))
                    except Exception:
                        pass

        # Feature importance: build a new “top-20” bar for the first FI file (or aggregate later)
        if fi_files:
            try:
                import plotly.express as px  # type: ignore

                fi_df = pd.read_csv(fi_files[0])
                # best-effort: find columns
                # common: Feature, Importance
                cols = [c.lower() for c in fi_df.columns]
                fcol = fi_df.columns[cols.index("feature")] if "feature" in cols else fi_df.columns[0]
                icol = fi_df.columns[cols.index("importance")] if "importance" in cols else fi_df.columns[-1]
                top = fi_df[[fcol, icol]].dropna().sort_values(icol, ascending=False).head(20)
                fig = px.bar(top[::-1], x=icol, y=fcol, orientation="h", title=f"{name}: Top Feature Importance (sample)")
                out = self.paths.figures_dir / f"{name}_fi_top20.png"
                if _safe_plotly_to_png(fig, out):
                    figs["fi_top20"] = str(out.relative_to(self.paths.reporting_dir))
            except Exception:
                pass

        return {
            "dataset_name": name,
            "paths": {
                "dataset_dir": str(ds_dir),
            },
            "tables": {
                "summary_mean": summary_mean.to_dict(orient="records") if summary_mean is not None else None,
                "summary_median": summary_median.to_dict(orient="records") if summary_median is not None else None,
                "summary_std": summary_std.to_dict(orient="records") if summary_std is not None else None,
                "ensembles_mean": ens_mean.to_dict(orient="records") if ens_mean is not None else None,
                "ensembles_median": ens_median.to_dict(orient="records") if ens_median is not None else None,
                "ensembles_std": ens_std.to_dict(orient="records") if ens_std is not None else None,
                "feature_selection": feat_sel.to_dict(orient="records") if feat_sel is not None else None,
                "missingness": missingness.to_dict(orient="records") if missingness is not None else None,
                "class_counts": class_counts.to_dict(orient="records") if class_counts is not None else None,
                "univariate": univariate.to_dict(orient="records") if univariate is not None else None,
                "runtimes": runtimes.to_dict(orient="records") if runtimes is not None else None,
            },
            "figures": figs,
        }

    def _collect_dataset_comparisons_block(self) -> Dict[str, Any]:
        """
        Phase 9 outputs live in <exp_root>/DatasetComparisons
        """
        dc = self.exp_root / "DatasetComparisons"
        if not dc.is_dir():
            return {"present": False}

        best_kw = self._read_csv_if_exists(dc / "BestCompare_KruskalWallis.csv")
        best_mw = self._read_csv_if_exists(dc / "BestCompare_MannWhitney.csv")
        best_wx = self._read_csv_if_exists(dc / "BestCompare_WilcoxonRank.csv")
        mw_all = self._read_csv_if_exists(dc / "MannWhitney_all.csv")
        wx_all = self._read_csv_if_exists(dc / "WilcoxonRank_all.csv")

        # NEW figure: heatmap of best-compare p-values (KW) if present
        figs: Dict[str, str] = {}
        if best_kw is not None and not best_kw.empty and "P-Value" in best_kw.columns:
            try:
                import plotly.express as px  # type: ignore

                df = best_kw[["P-Value"]].copy()
                df["Metric"] = best_kw.index if best_kw.index.name is not None else range(len(best_kw))
                df = df.reset_index(drop=True)
                fig = px.bar(df, x="Metric", y="P-Value", title="DatasetComparison: Best-Model Kruskal-Wallis P-Values")
                out = self.paths.figures_dir / "datasetcompare_best_kw_pvalues.png"
                if _safe_plotly_to_png(fig, out):
                    figs["best_kw_pvalues"] = str(out.relative_to(self.paths.reporting_dir))
            except Exception:
                pass

        return {
            "present": True,
            "tables": {
                "best_kw": best_kw.to_dict(orient="records") if best_kw is not None else None,
                "best_mw": best_mw.to_dict(orient="records") if best_mw is not None else None,
                "best_wx": best_wx.to_dict(orient="records") if best_wx is not None else None,
                "mw_all": mw_all.to_dict(orient="records") if mw_all is not None else None,
                "wx_all": wx_all.to_dict(orient="records") if wx_all is not None else None,
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
        """
        Use base_url so relative image paths resolve.
        """
        base_url = str(self.paths.reporting_dir)
        HTML(string=html_text, base_url=base_url).write_pdf(str(self.paths.pdf))

    def save_runtime(self):
        rt_dir = self.exp_root / "runtime"
        rt_dir.mkdir(exist_ok=True)
        (rt_dir / "runtime_report.txt").write_text(str(time.time() - (self.job_start_time or time.time())))

    def run(self):
        self.job_start_time = time.time()

        datasets = self._list_datasets()
        if not datasets:
            raise RuntimeError(f"No dataset folders (with CVDatasets/) found under: {self.exp_root}")

        dataset_blocks = [self._collect_dataset_block(ds) for ds in datasets]
        dc_block = self._collect_dataset_comparisons_block()

        report_data: Dict[str, Any] = {
            "title": self.title,
            "experiment_name": self.experiment_name,
            "experiment_root": str(self.exp_root),
            "generated_at_epoch": int(time.time()),
            "datasets": dataset_blocks,
            "dataset_comparisons": dc_block,
        }

        # Persist data JSON
        self.paths.data_json.write_text(json.dumps(report_data, indent=2))

        # Render HTML
        html_text = self._render_html(report_data)
        self.paths.html.write_text(html_text, encoding="utf-8")

        # Write PDF
        self._write_pdf(html_text)

        # Mark completed
        jc = self.exp_root / "jobsCompleted"
        jc.mkdir(exist_ok=True)
        (jc / "job_reporting.txt").write_text("complete")

        self.save_runtime()
        logger.info("Phase 11 reporting complete: %s", self.paths.pdf)
