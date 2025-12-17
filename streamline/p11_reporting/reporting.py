from __future__ import annotations

import os
import re
import csv
import math
import pickle
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

from streamline import __version__ as version


# -----------------------------
# Helpers
# -----------------------------
EXCLUDE_TOP_LEVEL = {
    ".DS_Store",
    ".idea",
    "jobs",
    "jobsCompleted",
    "logs",
    "dask_logs",
    "reporting",
    "runtime",
    "DatasetComparisons",
    "metadata.pickle",
    "metadata.csv",
    "algInfo.pickle",
}

# Keys where "lower is better" like the old FPDF logic
LOWER_IS_BETTER = {
    "FP", "FN", "LR-",
    "Max Error", "Mean Absolute Error", "Mean Squared Error", "Median Absolute Error",
}

# Common metric preference order for picking “headline” plots if multiple exist
METRIC_PREFERENCE = [
    "Balanced Accuracy",
    "ROC AUC",
    "PRC AUC",
    "Accuracy",
    "F1 Score",
]


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


def read_csv_records(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if limit is not None:
        df = df.head(limit)
    return df.to_dict(orient="records")


def read_csv_df(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def safe_relpath(path: Path, root: Path) -> str:
    """WeasyPrint is happiest with file:// URLs or absolute paths. We provide absolute POSIX paths."""
    return str(path.resolve().as_posix())


def list_datasets(experiment_root: Path) -> List[Path]:
    datasets = []
    for p in experiment_root.iterdir():
        if not p.is_dir():
            continue
        if p.name in EXCLUDE_TOP_LEVEL:
            continue
        # datasets are folders like hcc_demo, hcc_data_custom, etc.
        datasets.append(p)
    return sorted(datasets, key=lambda x: x.name)


def pick_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def find_first_matching(dir_path: Path, candidates: List[str]) -> Optional[Path]:
    """Return first existing file within dir_path matching any candidate names."""
    for name in candidates:
        p = dir_path / name
        if p.exists():
            return p
    return None


def compute_best_cells(mean_df: pd.DataFrame) -> Dict[Tuple[int, str], bool]:
    """
    Returns a mapping (row_index, column_name) -> True if that cell is "best" in its column,
    using LOWER_IS_BETTER for specific metric names, else max().

    Assumes:
      - First column is algorithm name or something non-numeric (we skip it)
      - Numeric columns are metrics
    """
    best_map: Dict[Tuple[int, str], bool] = {}
    if mean_df is None or mean_df.empty:
        return best_map

    df = mean_df.copy()

    # Try to coerce numeric columns; keep non-numeric as-is
    for col in df.columns:
        if col == df.columns[0]:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in df.columns[1:]:
        series = df[col]
        if series.dropna().empty:
            continue
        if col in LOWER_IS_BETTER:
            best_val = series.min(skipna=True)
        else:
            best_val = series.max(skipna=True)
        # Mark ties as best too
        for idx, val in series.items():
            if pd.isna(val):
                continue
            try:
                if float(val) == float(best_val):
                    best_map[(idx, col)] = True
            except Exception:
                pass

    return best_map


def merge_mean_std_tables(mean_df: Optional[pd.DataFrame], std_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Builds a single table where numeric metric columns become "mean ± std" strings.
    If std is missing, returns mean unchanged.
    """
    if mean_df is None:
        return None
    if std_df is None or std_df.empty:
        return mean_df

    m = mean_df.copy()
    s = std_df.copy()

    # Align rows by first column if possible, else by index
    key = m.columns[0]
    if key in s.columns:
        s_map = {str(r[key]): r for _, r in s.iterrows()}
        rows = []
        for _, r in m.iterrows():
            rr = r.copy()
            sid = str(r[key])
            sr = s_map.get(sid)
            for col in m.columns[1:]:
                mv = pd.to_numeric(r.get(col), errors="coerce")
                sv = pd.to_numeric(sr.get(col), errors="coerce") if sr is not None else math.nan
                if pd.isna(mv):
                    rr[col] = ""
                elif pd.isna(sv):
                    rr[col] = f"{mv:.3f}"
                else:
                    rr[col] = f"{mv:.3f} ± {sv:.3f}"
            rows.append(rr)
        return pd.DataFrame(rows, columns=m.columns)

    # Fallback: merge by index
    for col in m.columns[1:]:
        mv = pd.to_numeric(m[col], errors="coerce")
        sv = pd.to_numeric(s[col], errors="coerce") if col in s.columns else math.nan
        out = []
        for i in range(len(m)):
            a = mv.iloc[i] if i < len(mv) else math.nan
            b = sv.iloc[i] if hasattr(sv, "iloc") and i < len(sv) else math.nan
            if pd.isna(a):
                out.append("")
            elif pd.isna(b):
                out.append(f"{float(a):.3f}")
            else:
                out.append(f"{float(a):.3f} ± {float(b):.3f}")
        m[col] = out
    return m


@dataclass
class DatasetSection:
    dataset_name: str
    dataset_slug: str
    dataset_dir: str

    # figures (absolute paths)
    figures: Dict[str, Optional[str]]

    # tables (records)
    tables: Dict[str, List[Dict[str, Any]]]

    # performance tables (rendered as table matrix)
    perf: Dict[str, Any]


@dataclass
class DatasetComparisonsSection:
    present: bool
    figures: Dict[str, Optional[str]]
    tables: Dict[str, List[Dict[str, Any]]]


class ReportPhaseJob:
    """
    Generates a multi-dataset STREAMLINE PDF report using Jinja2 + WeasyPrint.
    Uses pre-generated figures/tables from the experiment directory tree.
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        experiment_name: Optional[str] = None,
        outcome_label: str = "Class",
        outcome_type: str = "Binary",
        instance_label: Optional[str] = None,
        make_pdf: bool = True,
        training: bool = True, # True for training phase, False for testing phase
    ):
        super().__init__()


        self.experiment_root = Path(output_path) / str(experiment_name)
        self.experiment_name = str(experiment_name)

        self.training = training
        self.template_dir = Path(__file__).parent / "templates"
        self.template_name = "report.html.j2"
        # self.out_pdf_name = "reporting/" + f"{self.experiment_name}_STREAMLINE_Report.pdf"

        # metadata
        self.metadata = self._load_pickle(self.experiment_root / "metadata.pickle") or {}
        # self.alg_info = self._load_pickle(self.experiment_root / "algInfo.pickle") or {}
        self.outcome_type = self.metadata.get("Outcome Type", "Unknown")

    def _load_pickle(self, path: Path) -> Any:
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logging.warning("Failed to load pickle %s: %s", path, e)
            return None

    def run(self):
        self.job()

    def job(self):
        logging.info("Starting WeasyPrint ReportPhaseJob for %s", self.experiment_root)

        generated_at = datetime.now()

        # Collect datasets
        ds_dirs = list_datasets(self.experiment_root)
        datasets: List[DatasetSection] = []
        for d in ds_dirs:
            datasets.append(self._build_dataset_section(d))

        # Dataset comparisons
        comparisons = self._build_dataset_comparisons_section()

        # Render HTML -> PDF
        env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = env.get_template(self.template_name)

        html_str = template.render(
            title=f"STREAMLINE Evaluation Report",
            experiment_name=self.experiment_name,
            experiment_root=str(self.experiment_root.resolve().as_posix()),
            generated_at=generated_at.strftime("%Y-%m-%d %H:%M:%S"),
            streamline_version=version,
            training=self.training,
            outcome_type=self.outcome_type,
            metadata=self.metadata,
            # alg_info=self.alg_info,
            datasets=datasets,
            dataset_comparisons=comparisons,
        )

        out_pdf = self.experiment_root / "reporting" / "report.pdf"
        HTML(string=html_str, base_url=str(self.experiment_root.resolve().as_uri())).write_pdf(str(out_pdf))
        logging.info("Wrote PDF: %s", out_pdf)

        # job completion marker (optional, mirror prior conventions)
        try:
            jc = self.experiment_root / "jobsCompleted"
            jc.mkdir(exist_ok=True, parents=True)
            (jc / "job_p11_reporting.txt").write_text("complete")
        except Exception:
            pass

    def _build_dataset_section(self, dataset_dir: Path) -> DatasetSection:
        ds_name = dataset_dir.name
        ds_slug = slugify(ds_name)

        # --- Figures: use pre-generated PNGs that already exist in old reports ---
        # Keep these paths aligned to your directory tree conventions.
        exploratory_dir = dataset_dir / "exploratory"
        model_eval_dir = dataset_dir / "model_evaluation"
        ensemble_eval_dir = dataset_dir / "ensemble_evaluation"
        fi_dir = model_eval_dir / "feature_importance"
        fs_mi_dir = dataset_dir / "feature_selection" / "mutual_information"
        fs_ms_dir = dataset_dir / "feature_selection" / "multisurf"

        # Summary plots (old report used these exact filenames)
        figures: Dict[str, Optional[str]] = {}

        figures["class_counts"] = safe_relpath(exploratory_dir / "ClassCountsBarPlot.png", self.experiment_root) if (exploratory_dir / "ClassCountsBarPlot.png").exists() else None
        figures["feature_correlations"] = safe_relpath(exploratory_dir / "FeatureCorrelations.png", self.experiment_root) if (exploratory_dir / "FeatureCorrelations.png").exists() else None

        figures["summary_roc"] = safe_relpath(model_eval_dir / "Summary_ROC.png", self.experiment_root) if (model_eval_dir / "Summary_ROC.png").exists() else None
        figures["summary_prc"] = safe_relpath(model_eval_dir / "Summary_PRC.png", self.experiment_root) if (model_eval_dir / "Summary_PRC.png").exists() else None

        # metric boxplots (these include spaces in filename)
        mb = model_eval_dir / "metricBoxplots"
        figures["compare_roc_auc"] = safe_relpath(mb / "Compare_ROC AUC.png", self.experiment_root) if (mb / "Compare_ROC AUC.png").exists() else None
        figures["compare_prc_auc"] = safe_relpath(mb / "Compare_PRC AUC.png", self.experiment_root) if (mb / "Compare_PRC AUC.png").exists() else None

        # Feature importance “composite” used in old PDF report
        figures["fi_norm_weight"] = safe_relpath(fi_dir / "Compare_FI_Norm_Weight.png", self.experiment_root) if (fi_dir / "Compare_FI_Norm_Weight.png").exists() else None

        # Feature selection figures
        figures["fs_mutual_information"] = safe_relpath(fs_mi_dir / "TopAverageScores.png", self.experiment_root) if (fs_mi_dir / "TopAverageScores.png").exists() else None
        figures["fs_multisurf"] = safe_relpath(fs_ms_dir / "TopAverageScores.png", self.experiment_root) if (fs_ms_dir / "TopAverageScores.png").exists() else None

        # --- Tables used in FPDF ---
        tables: Dict[str, List[Dict[str, Any]]] = {}

        dp_path = exploratory_dir / "DataProcessSummary.csv"
        tables["data_process_summary"] = read_csv_records(dp_path)

        uni_path = exploratory_dir / "univariate_analyses" / "Univariate_Significance.csv"
        tables["univariate_top10"] = read_csv_records(uni_path, limit=10)

        # --- Performance tables (new requirement) ---
        perf: Dict[str, Any] = {}

        # Models
        m_mean = read_csv_df(model_eval_dir / "Summary_performance_mean.csv")
        m_median = read_csv_df(model_eval_dir / "Summary_performance_median.csv")
        m_std = read_csv_df(model_eval_dir / "Summary_performance_std.csv")

        m_mean_std = merge_mean_std_tables(m_mean, m_std)
        m_best_map = compute_best_cells(m_mean) if m_mean is not None else {}

        perf["models_mean_std"] = _df_to_table_payload(m_mean_std, best_map=m_best_map)
        perf["models_median"] = _df_to_table_payload(m_median, best_map=None)

        # Ensembles
        e_mean = read_csv_df(ensemble_eval_dir / "Ensembles_performance_mean.csv")
        e_median = read_csv_df(ensemble_eval_dir / "Ensembles_performance_median.csv")
        e_std = read_csv_df(ensemble_eval_dir / "Ensembles_performance_std.csv")

        e_mean_std = merge_mean_std_tables(e_mean, e_std)
        e_best_map = compute_best_cells(e_mean) if e_mean is not None else {}

        perf["ensembles_mean_std"] = _df_to_table_payload(e_mean_std, best_map=e_best_map)
        perf["ensembles_median"] = _df_to_table_payload(e_median, best_map=None)

        # Runtimes table
        rt_path = dataset_dir / "runtimes.csv"
        tables["runtimes"] = read_csv_records(rt_path)

        return DatasetSection(
            dataset_name=ds_name,
            dataset_slug=ds_slug,
            dataset_dir=str(dataset_dir.resolve().as_posix()),
            figures=figures,
            tables=tables,
            perf=perf,
        )

    def _build_dataset_comparisons_section(self) -> DatasetComparisonsSection:
        dc_dir = self.experiment_root / "DatasetComparisons"
        if not dc_dir.exists():
            return DatasetComparisonsSection(present=False, figures={}, tables={})

        figures: Dict[str, Optional[str]] = {}
        tables: Dict[str, List[Dict[str, Any]]] = {}

        box_dir = dc_dir / "dataCompBoxplots"
        # Old report used these filenames:
        figures["allmodels_roc_auc"] = safe_relpath(box_dir / "DataCompareAllModels_ROC AUC.png", self.experiment_root) if (box_dir / "DataCompareAllModels_ROC AUC.png").exists() else None
        figures["allmodels_prc_auc"] = safe_relpath(box_dir / "DataCompareAllModels_PRC AUC.png", self.experiment_root) if (box_dir / "DataCompareAllModels_PRC AUC.png").exists() else None

        kw_csv = dc_dir / "BestCompare_KruskalWallis.csv"
        tables["best_kw"] = read_csv_records(kw_csv)

        return DatasetComparisonsSection(
            present=True,
            figures=figures,
            tables=tables,
        )


def _df_to_table_payload(df: Optional[pd.DataFrame], best_map: Optional[Dict[Tuple[int, str], bool]]) -> Dict[str, Any]:
    """
    Returns a payload that’s easy to render in Jinja:
      {
        "present": bool,
        "columns": [...],
        "rows": [
           {"cells":[{"value":..., "best":bool}, ...]}
        ]
      }
    """
    if df is None or df.empty:
        return {"present": False, "columns": [], "rows": []}

    columns = list(df.columns)
    rows = []
    for ridx in range(len(df)):
        cell_list = []
        for cidx, col in enumerate(columns):
            val = df.iloc[ridx, cidx]
            is_best = False
            if best_map is not None and cidx > 0:
                is_best = best_map.get((ridx, col), False)
            cell_list.append({"value": "" if pd.isna(val) else str(val), "best": bool(is_best)})
        rows.append({"cells": cell_list})
    return {"present": True, "columns": columns, "rows": rows}
