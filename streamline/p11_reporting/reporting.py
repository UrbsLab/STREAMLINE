from __future__ import annotations

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


try:
    # WeasyPrint is optional; the phase still runs without it.
    from weasyprint import HTML
    HAS_WEASYPRINT = True
except Exception:  # ImportError or runtime issues
    HAS_WEASYPRINT = False
    logger.warning(
        "WeasyPrint not available. PDF report will not be generated. "
        "Install weasyprint if you want PDF output."
    )


class ReportPhaseJob:
    """
    Phase 10: Experiment-level reporting.

    Responsibilities:
      * Aggregate outputs from all previous phases:
          - Phase 6: base model metrics & plots
          - Phase 7: ensemble metrics & plots (if present)
          - Phase 8: per-dataset statistics & summary plots
          - Phase 9: dataset comparison CSVs & plots
      * Write a JSON snapshot of the experiment.
      * Generate a static HTML report page.
      * Optionally generate a PDF via WeasyPrint (no Pango-specific features).
    """

    def __init__(
        self,
        output_path: str,
        experiment_name: str,
        outcome_label: str = "Class",
        outcome_type: str = "Binary",
        instance_label: Optional[str] = None,
        make_pdf: bool = True,
    ):
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.outcome_label = outcome_label
        self.outcome_type = outcome_type
        self.instance_label = instance_label
        self.make_pdf = bool(make_pdf)

        self.exp_root = Path(self.output_path) / self.experiment_name
        if not self.exp_root.is_dir():
            raise RuntimeError(f"Experiment folder not found: {self.exp_root}")

        # where we write report.json / report.html / report.pdf
        self.report_dir = self.exp_root / "reporting"
        self.report_dir.mkdir(exist_ok=True)

        self.job_start_time: float = 0.0

    # ------------------------------------------------------------------
    # PUBLIC ENTRY
    # ------------------------------------------------------------------
    def run(self) -> None:
        self.job_start_time = time.time()
        logger.info("Phase 11: building experiment report for %s", self.exp_root)

        summary = self.collect_summary()
        json_path = self.report_dir / "report_data.json"
        with json_path.open("w") as f:
            json.dump(summary, f, indent=2, default=self._json_default)
        logger.info("Wrote experiment summary JSON to %s", json_path)

        html = self.render_html(summary)
        html_path = self.report_dir / "report.html"
        with html_path.open("w", encoding="utf-8") as f:
            f.write(html)
        logger.info("Wrote HTML report to %s", html_path)

        if self.make_pdf and HAS_WEASYPRINT:
            try:
                pdf_path = self.report_dir / "report.pdf"
                # Use minimal CSS, no advanced Pango features required.
                HTML(filename=str(html_path)).write_pdf(str(pdf_path))
                logger.info("Wrote PDF report to %s", pdf_path)
            except Exception as e:
                raise e
                logger.warning("WeasyPrint PDF generation failed: %s", e)

        self._save_runtime()
        logger.info("Phase 10 complete.")

    # ------------------------------------------------------------------
    # SUMMARY COLLECTION (core for both HTML and Streamlit)
    # ------------------------------------------------------------------
    def collect_summary(self) -> Dict[str, Any]:
        """
        Walk the experiment folder, gather per-dataset and cross-dataset
        outputs into a single JSON-serializable object.
        """
        datasets = [
            d for d in sorted(self.exp_root.iterdir())
            if d.is_dir()
            and (d / "CVDatasets").is_dir()
            and d.name not in {"jobs", "logs", "jobsCompleted", "dask_logs", "runtime", "DatasetComparisons", "reporting"}
        ]

        if not datasets:
            logger.warning("No dataset directories found under %s", self.exp_root)

        ds_summaries: List[Dict[str, Any]] = []
        for ds in datasets:
            ds_summaries.append(self._collect_dataset_summary(ds))

        compare_summary = self._collect_dataset_comparisons()

        summary: Dict[str, Any] = {
            "experiment_name": self.experiment_name,
            "output_path": str(self.output_path),
            "outcome_label": self.outcome_label,
            "outcome_type": self.outcome_type,
            "instance_label": self.instance_label,
            "datasets": ds_summaries,
            "dataset_comparisons": compare_summary,
        }
        return summary

    def _collect_dataset_summary(self, ds_dir: Path) -> Dict[str, Any]:
        """
        One dataset: collect base model stats, ensemble stats, plots, runtimes.
        """
        name = ds_dir.name
        model_eval = ds_dir / "model_evaluation"
        ensemble_eval = ds_dir / "ensemble_evaluation"
        runtime_csv = ds_dir / "runtimes.csv"

        base_mean, base_med, base_std = None, None, None
        if (model_eval / "Summary_performance_mean.csv").is_file():
            base_mean = pd.read_csv(model_eval / "Summary_performance_mean.csv")
        if (model_eval / "Summary_performance_median.csv").is_file():
            base_med = pd.read_csv(model_eval / "Summary_performance_median.csv")
        if (model_eval / "Summary_performance_std.csv").is_file():
            base_std = pd.read_csv(model_eval / "Summary_performance_std.csv")

        ensembles_mean = ensembles_med = ensembles_std = None
        if ensemble_eval.is_dir():
            if (ensemble_eval / "Ensembles_performance_mean.csv").is_file():
                ensembles_mean = pd.read_csv(ensemble_eval / "Ensembles_performance_mean.csv")
            if (ensemble_eval / "Ensembles_performance_median.csv").is_file():
                ensembles_med = pd.read_csv(ensemble_eval / "Ensembles_performance_median.csv")
            if (ensemble_eval / "Ensembles_performance_std.csv").is_file():
                ensembles_std = pd.read_csv(ensemble_eval / "Ensembles_performance_std.csv")

        # gather available plots
        plots: Dict[str, Any] = {}
        # base model summary plots
        for fname, key in [
            ("Summary_ROC.png", "summary_roc"),
            ("Summary_PRC.png", "summary_prc"),
        ]:
            p = model_eval / fname
            if p.is_file():
                plots[key] = os.path.relpath(p, self.report_dir)

        fi_dir = model_eval / "feature_importance"
        if fi_dir.is_dir():
            fi_plots = [
                os.path.relpath(p, self.report_dir)
                for p in fi_dir.glob("*.png")
            ]
            if fi_plots:
                plots["feature_importance"] = sorted(fi_plots)

        # ensemble plots, if present
        if ensemble_eval.is_dir():
            ens_plots = {}
            p_roc = ensemble_eval / "Summary_ROC_ensembles.png"
            p_prc = ensemble_eval / "Summary_PRC_ensembles.png"
            if p_roc.is_file():
                ens_plots["summary_roc_ensembles"] = os.path.relpath(p_roc, self.report_dir)
            if p_prc.is_file():
                ens_plots["summary_prc_ensembles"] = os.path.relpath(p_prc, self.report_dir)
            if ens_plots:
                plots["ensembles"] = ens_plots

        runtimes = None
        if runtime_csv.is_file():
            try:
                runtimes = pd.read_csv(runtime_csv)
            except Exception:
                runtimes = None

        def df2dict(df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
            if df is None:
                return None
            # orient="index" gives {algorithm: {metric: value, ...}}
            return df.to_dict(orient="index")

        ds_summary: Dict[str, Any] = {
            "name": name,
            "relative_path": os.path.relpath(ds_dir, self.report_dir),
            "base_metrics_mean": df2dict(base_mean),
            "base_metrics_median": df2dict(base_med),
            "base_metrics_std": df2dict(base_std),
            "ensemble_metrics_mean": df2dict(ensembles_mean),
            "ensemble_metrics_median": df2dict(ensembles_med),
            "ensemble_metrics_std": df2dict(ensembles_std),
            "plots": plots,
            "runtimes": df2dict(runtimes),
        }
        return ds_summary

    def _collect_dataset_comparisons(self) -> Dict[str, Any]:
        """
        Phase 9 outputs under <exp_root>/DatasetComparisons.
        """
        comp_root = self.exp_root / "DatasetComparisons"
        if not comp_root.is_dir():
            return {}

        def rel(p: Path) -> str:
            return os.path.relpath(p, self.report_dir)

        csvs = {
            "kruskal_all": [],
            "wilcoxon_all": None,
            "mannwhitney_all": None,
            "best_kruskal": None,
            "best_wilcoxon": None,
            "best_mannwhitney": None,
        }

        # Kruskal per algorithm
        for p in comp_root.glob("KruskalWallis_*.csv"):
            csvs["kruskal_all"].append(rel(p))

        if (comp_root / "WilcoxonRank_all.csv").is_file():
            csvs["wilcoxon_all"] = rel(comp_root / "WilcoxonRank_all.csv")
        if (comp_root / "MannWhitney_all.csv").is_file():
            csvs["mannwhitney_all"] = rel(comp_root / "MannWhitney_all.csv")
        if (comp_root / "BestCompare_KruskalWallis.csv").is_file():
            csvs["best_kruskal"] = rel(comp_root / "BestCompare_KruskalWallis.csv")
        if (comp_root / "BestCompare_WilcoxonRank.csv").is_file():
            csvs["best_wilcoxon"] = rel(comp_root / "BestCompare_WilcoxonRank.csv")
        if (comp_root / "BestCompare_MannWhitney.csv").is_file():
            csvs["best_mannwhitney"] = rel(comp_root / "BestCompare_MannWhitney.csv")

        boxplot_dir = comp_root / "dataCompBoxplots"
        boxplots = []
        if boxplot_dir.is_dir():
            boxplots = [rel(p) for p in sorted(boxplot_dir.glob("*.png"))]

        return {
            "csvs": csvs,
            "boxplots": boxplots,
        }

    # ------------------------------------------------------------------
    # HTML RENDERING
    # ------------------------------------------------------------------
    def render_html(self, summary: Dict[str, Any]) -> str:
        """
        Simple, self-contained HTML (no JS) that can be:
          * viewed in a browser
          * fed to WeasyPrint for PDF rendering
        """
        exp_name = summary.get("experiment_name", "")
        outcome_label = summary.get("outcome_label", self.outcome_label)
        outcome_type = summary.get("outcome_type", self.outcome_type)

        # Minimal CSS: no webfonts, no advanced features, Pango not needed.
        css = """
        body { font-family: sans-serif; margin: 20px; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; margin-bottom: 1.5em; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 4px 8px; font-size: 12px; }
        th { background-color: #f5f5f5; text-align: left; }
        .dataset-section { border-top: 2px solid #999; margin-top: 2em; padding-top: 1em; }
        .plot-grid { display: flex; flex-wrap: wrap; gap: 16px; }
        .plot-grid img { max-width: 48%; border: 1px solid #ccc; padding: 4px; }
        .small-note { color: #777; font-size: 11px; }
        .badge { display: inline-block; padding: 2px 6px; border-radius: 4px;
                 background-color: #eee; font-size: 11px; margin-right: 4px; }
        """

        html_parts: List[str] = []
        html_parts.append("<!DOCTYPE html>")
        html_parts.append("<html><head><meta charset='utf-8'>")
        html_parts.append(f"<title>STREAMLINE Report - {exp_name}</title>")
        html_parts.append(f"<style>{css}</style></head><body>")

        html_parts.append(f"<h1>STREAMLINE Report – {exp_name}</h1>")
        html_parts.append("<p class='small-note'>"
                          "Generated from modeling, ensemble, statistics, and dataset comparison phases."
                          "</p>")
        html_parts.append("<hr/>")

        # ---------- Dataset sections ----------
        for ds in summary.get("datasets", []):
            html_parts.append(self._render_dataset_section(ds))

        # ---------- Dataset comparisons ----------
        html_parts.append("<div class='dataset-section'>")
        html_parts.append("<h2>Cross-dataset comparisons (Phase 9)</h2>")

        comp = summary.get("dataset_comparisons", {})
        csvs = comp.get("csvs", {})
        boxplots = comp.get("boxplots", [])

        if not comp:
            html_parts.append("<p><em>No dataset comparison outputs found.</em></p>")
        else:
            html_parts.append("<h3>Statistics CSVs</h3>")
            html_parts.append("<ul>")
            for label, path in csvs.items():
                if path is None:
                    continue
                if isinstance(path, list):
                    for p in path:
                        html_parts.append(f"<li><code>{label}</code>: {p}</li>")
                else:
                    html_parts.append(f"<li><code>{label}</code>: {path}</li>")
            html_parts.append("</ul>")

            if boxplots:
                html_parts.append("<h3>Dataset comparison boxplots</h3>")
                html_parts.append("<div class='plot-grid'>")
                for b in boxplots:
                    html_parts.append(f"<div><img src='{b}' alt='{b}'/></div>")
                html_parts.append("</div>")

        html_parts.append("</div>")

        # footer
        html_parts.append("<hr/>")
        html_parts.append("<p class='small-note'>"
                          f"Outcome label: <strong>{outcome_label}</strong> "
                          f"({outcome_type}). "
                          "This HTML can be converted to PDF with WeasyPrint."
                          "</p>")
        html_parts.append("</body></html>")
        return "\n".join(html_parts)

    def _render_dataset_section(self, ds: Dict[str, Any]) -> str:
        name = ds.get("name", "UnknownDataset")
        plots = ds.get("plots", {})

        html: List[str] = []
        html.append("<div class='dataset-section'>")
        html.append(f"<h2>Dataset: {name}</h2>")

        # ---- base metrics ----
        html.append("<h3>Base models – summary metrics</h3>")
        mean = ds.get("base_metrics_mean")
        median = ds.get("base_metrics_median")
        std = ds.get("base_metrics_std")

        if mean is None:
            html.append("<p><em>No base model summaries found.</em></p>")
        else:
            html.append(self._render_metric_table(mean, median, std, label="Base models"))

        # ---- ensemble metrics ----
        ens_mean = ds.get("ensemble_metrics_mean")
        if ens_mean is not None:
            html.append("<h3>Ensembles – summary metrics</h3>")
            html.append(self._render_metric_table(
                ens_mean,
                ds.get("ensemble_metrics_median"),
                ds.get("ensemble_metrics_std"),
                label="Ensembles")
            )

        # ---- plots ----
        html.append("<h3>Summary plots</h3>")
        html.append("<div class='plot-grid'>")

        for key in ["summary_roc", "summary_prc"]:
            p = plots.get(key)
            if p:
                title = "Base model ROC" if key == "summary_roc" else "Base model PRC"
                html.append(f"<div><div class='small-note'>{title}</div>"
                            f"<img src='{p}' alt='{title}'/></div>")

        fi_list = plots.get("feature_importance") or []
        for p in fi_list:
            html.append(f"<div><div class='small-note'>Feature importance</div>"
                        f"<img src='{p}' alt='Feature importance'/></div>")

        ens_plots = plots.get("ensembles", {})
        for key, p in ens_plots.items():
            label = "Ensemble ROC" if "roc" in key else "Ensemble PRC"
            html.append(f"<div><div class='small-note'>{label}</div>"
                        f"<img src='{p}' alt='{label}'/></div>")

        html.append("</div>")

        # ---- runtimes ----
        rt = ds.get("runtimes")
        if rt:
            html.append("<h3>Runtimes</h3>")
            html.append("<table><thead><tr><th>Component</th><th>Phase</th><th>Time (sec)</th></tr></thead><tbody>")
            # rt is {idx: {col: val}}
            for _, row in rt.items():
                comp = row.get("Pipeline Component", "")
                phase = row.get("Phase", "")
                t = row.get("Time (sec)", "")
                html.append(f"<tr><td>{comp}</td><td>{phase}</td><td>{t}</td></tr>")
            html.append("</tbody></table>")

        html.append("</div>")
        return "\n".join(html)

    def _render_metric_table(
        self,
        mean: Dict[str, Any],
        median: Optional[Dict[str, Any]],
        std: Optional[Dict[str, Any]],
        label: str,
    ) -> str:
        """
        mean/median/std are {algorithm: {metric: val}} dictionaries.
        """
        # determine metric columns
        if not mean:
            return "<p><em>No metrics.</em></p>"

        algs = sorted(mean.keys())
        metrics = sorted(next(iter(mean.values())).keys())

        html: List[str] = []
        html.append(f"<p class='small-note'>{label}</p>")
        html.append("<table><thead><tr><th>Model</th>")
        for m in metrics:
            html.append(f"<th>{m}</th>")
        html.append("</tr></thead><tbody>")

        for alg in algs:
            html.append(f"<tr><td><span class='badge'>{alg}</span></td>")
            for m in metrics:
                v_mean = mean.get(alg, {}).get(m, None)
                v_med = median.get(alg, {}).get(m, None) if median else None
                v_std = std.get(alg, {}).get(m, None) if std else None

                def fmt(x):
                    if x is None or (isinstance(x, float) and np.isnan(x)):
                        return "–"
                    try:
                        return f"{float(x):.3f}"
                    except Exception:
                        return str(x)

                parts = [fmt(v_mean)]
                if v_med is not None:
                    parts.append(f"med={fmt(v_med)}")
                if v_std is not None:
                    parts.append(f"sd={fmt(v_std)}")

                html.append(f"<td>{' / '.join(parts)}</td>")
            html.append("</tr>")
        html.append("</tbody></table>")
        return "\n".join(html)

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def _save_runtime(self) -> None:
        runtime_dir = self.exp_root / "runtime"
        runtime_dir.mkdir(exist_ok=True)
        out = runtime_dir / "runtime_report.txt"
        with out.open("w") as f:
            f.write(str(time.time() - self.job_start_time))

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return str(obj)
