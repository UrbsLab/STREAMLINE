from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

# WeasyPrint
from weasyprint import HTML  # keep usage minimal; no direct pango usage


EXCLUDE_DIRS = {
    ".DS_Store", ".idea", "__pycache__",
    "DatasetComparisons", "jobs", "jobsCompleted", "logs", "dask_logs",
    "KeyFileCopy", "reporting", "runtime",
    "metadata.pickle", "metadata.csv", "algInfo.pickle", "run_params.pickle",
}


def _file_uri(p: Path) -> str:
    # WeasyPrint resolves local resources reliably via file:// URIs.
    return p.resolve().as_uri()


def _safe_read_csv_records(path: Path, nrows: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    try:
        df = pd.read_csv(path)
        if nrows is not None:
            df = df.head(nrows)
        return df.to_dict(orient="records")
    except Exception:
        return None


def _pick_existing(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p is not None and p.exists():
            return p
    return None


@dataclass
class DatasetReportBundle:
    dataset_name: str
    paths: Dict[str, str]
    figures: Dict[str, str]
    tables: Dict[str, List[Dict[str, Any]]]


class ReportPhaseJob:
    """
    Phase 11 Reporting:
    - Reuses pre-generated figures/tables from earlier phases (PNG/CSV/JSON outputs).
    - Builds HTML via Jinja2 and renders PDF via WeasyPrint.
    - Supports multiple dataset folders under experiment root.
    """

    def __init__(
        self,
        output_path: str,
        experiment_name: str,
        experiment_path: Optional[str] = None,
        outcome_label: Optional[str] = None,
        outcome_type: Optional[str] = None,
        instance_label: Optional[str] = None,
        make_pdf: bool = True,
    ):
        template_name = "report.html.j2"
        title = None
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.exp_root = Path(experiment_path) if experiment_path else (Path(output_path) / experiment_name)
        if not self.exp_root.is_dir():
            raise FileNotFoundError(f"Experiment folder not found: {self.exp_root}")

        self.reporting_dir = self.exp_root / "reporting"
        self.reporting_dir.mkdir(parents=True, exist_ok=True)

        self.title = title or f"STREAMLINE Report"
        self.template_name = template_name

        templates_dir = Path(__file__).with_name("templates")
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def run(self) -> None:
        logging.info("P11 Reporting: building report bundles")
        datasets = self._discover_datasets()
        ds_bundles = [self._build_dataset_bundle(ds) for ds in datasets]

        dataset_comparisons = self._build_dataset_comparisons_bundle()

        context = {
            "title": self.title,
            "experiment_name": self.experiment_name,
            "experiment_root": str(self.exp_root),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "footer_text": "Generated with STREAMLINE",
            "datasets": [self._bundle_to_template_dict(b) for b in ds_bundles],
            "dataset_comparisons": dataset_comparisons,
        }

        html_out = self.reporting_dir / "report.html"
        pdf_out = self.reporting_dir / "report.pdf"
        json_out = self.reporting_dir / "report_data.json"

        template = self.jinja_env.get_template(self.template_name)
        rendered = template.render(**context)

        html_out.write_text(rendered, encoding="utf-8")
        json_out.write_text(json.dumps(context, indent=2), encoding="utf-8")

        logging.info("P11 Reporting: rendering PDF via WeasyPrint")
        # base_url ensures relative links resolve; we also prefer file:// URIs for images.
        HTML(string=rendered, base_url=str(self.exp_root)).write_pdf(str(pdf_out))

        logging.info(f"P11 Reporting complete: {pdf_out}")

    def _discover_datasets(self) -> List[str]:
        names = []
        for p in self.exp_root.iterdir():
            if not p.is_dir():
                continue
            if p.name in EXCLUDE_DIRS:
                continue
            names.append(p.name)
        return sorted(names)

    def _build_dataset_bundle(self, dataset_name: str) -> DatasetReportBundle:
        ds_dir = self.exp_root / dataset_name

        # ---- Figures (reuse pre-generated PNGs as in legacy report) ----
        # Models: model_evaluation/
        me = ds_dir / "model_evaluation"
        exp = ds_dir / "exploratory"
        fi_sel = ds_dir / "feature_importance"  # (newer runs)
        fi_legacy_sel = ds_dir / "feature_selection"  # (older runs)

        figures: Dict[str, str] = {}

        # Primary summary plots (legacy locations in your tree)
        # - Summary_ROC.png, Summary_PRC.png
        models_roc = _pick_existing(me / "Summary_ROC.png", me / "LR_ROC.png")
        models_prc = _pick_existing(me / "Summary_PRC.png", me / "LR_PRC.png")
        if models_roc:
            figures["models_roc_img"] = _file_uri(models_roc)
        if models_prc:
            figures["models_prc_img"] = _file_uri(models_prc)

        # Metric boxplots (choose common ones)
        # Example: model_evaluation/metricBoxplots/Compare_ROC AUC.png etc.
        mb = me / "metricBoxplots"
        figures_map = {
            "models_mean_ROC AUC": mb / "Compare_ROC AUC.png",
            "models_mean_PRC AUC": mb / "Compare_PRC AUC.png",
            "models_mean_Balanced Accuracy": mb / "Compare_Balanced Accuracy.png",
            "models_mean_Accuracy": mb / "Compare_Accuracy.png",
            "models_mean_F1 Score": mb / "Compare_F1 Score.png",
        }
        for k, p in figures_map.items():
            if p.exists():
                figures[k] = _file_uri(p)

        # Exploratory figures frequently used in old report
        cc = _pick_existing(exp / "ClassCountsBarPlot.png")
        if cc:
            figures["class_counts_bar"] = _file_uri(cc)

        fc = _pick_existing(exp / "FeatureCorrelations.png")
        if fc:
            figures["feature_correlations"] = _file_uri(fc)

        # Feature importance: legacy compare plot in model_evaluation/feature_importance
        fi_compare = _pick_existing(me / "feature_importance" / "Compare_FI_Norm_Weight.png")
        if fi_compare:
            figures["fi_compare_norm_weight"] = _file_uri(fi_compare)

        # Feature selection plots (older report used TopAverageScores.png under MI/MultiSURF)
        mi_top = _pick_existing(
            ds_dir / "feature_importance" / "mutualinformation" / "TopAverageScores.png",
            ds_dir / "feature_importance" / "mutual_information" / "TopAverageScores.png",
            ds_dir / "feature_selection" / "mutualinformation" / "TopAverageScores.png",
            ds_dir / "feature_selection" / "mutual_information" / "TopAverageScores.png",
        )
        ms_top = _pick_existing(
            ds_dir / "feature_importance" / "multisurf" / "TopAverageScores.png",
            ds_dir / "feature_selection" / "multisurf" / "TopAverageScores.png",
        )
        if mi_top:
            figures["fi_top_avg_mutualinfo"] = _file_uri(mi_top)
        if ms_top:
            figures["fi_top_avg_multisurf"] = _file_uri(ms_top)

        # ---- Tables (CSV to HTML tables in template) ----
        tables: Dict[str, List[Dict[str, Any]]] = {}

        dps = _safe_read_csv_records(exp / "DataProcessSummary.csv")
        if dps:
            tables["data_process_summary"] = dps

        univ_path = exp / "univariate_analyses" / "Univariate_Significance.csv"
        univ = _safe_read_csv_records(univ_path, nrows=10)
        if univ:
            tables["univariate_top10"] = univ

        return DatasetReportBundle(
            dataset_name=dataset_name,
            paths={"dataset_dir": str(ds_dir)},
            figures=figures,
            tables=tables,
        )

    def _build_dataset_comparisons_bundle(self) -> Dict[str, Any]:
        dc_dir = self.exp_root / "DatasetComparisons"
        if not dc_dir.exists():
            return {"present": False}

        figures: Dict[str, str] = {}
        tables: Dict[str, Any] = {}

        # Pre-generated comparison plots (per your tree)
        box_dir = dc_dir / "dataCompBoxplots"
        roc_all = _pick_existing(box_dir / "DataCompareAllModels_ROC AUC.png")
        prc_all = _pick_existing(box_dir / "DataCompareAllModels_PRC AUC.png")
        if roc_all:
            figures["compare_allmodels_roc_auc"] = _file_uri(roc_all)
        if prc_all:
            figures["compare_allmodels_prc_auc"] = _file_uri(prc_all)

        best_kw = _safe_read_csv_records(dc_dir / "BestCompare_KruskalWallis.csv")
        if best_kw:
            tables["best_kw"] = best_kw

        return {
            "present": True,
            "path": str(dc_dir),
            "figures": figures,
            "tables": tables,
        }

    @staticmethod
    def _bundle_to_template_dict(b: DatasetReportBundle) -> Dict[str, Any]:
        return {
            "dataset_name": b.dataset_name,
            "paths": b.paths,
            "figures": b.figures,
            "tables": b.tables,
        }
