from __future__ import annotations

import logging
import math
import os
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

from streamline import __version__ as version


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
    # "algInfo.pickle",
    "run_params.pickle",
}

# Keys where "lower is better" (to decide best-cell highlighting)
LOWER_IS_BETTER = {
    "FP", "FN", "LR-",
    "Max Error", "Mean Absolute Error", "Mean Squared Error", "Median Absolute Error",
}

# A4-ish defaults (WeasyPrint @page handles margins, but we use these for plot sizing, etc.)
DEFAULT_DPI = 150


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


def safe_abs_posix(path: Path) -> str:
    return str(path.resolve().as_posix())


def list_datasets(experiment_root: Path) -> List[Path]:
    datasets: List[Path] = []
    for p in experiment_root.iterdir():
        if not p.is_dir():
            continue
        if p.name in EXCLUDE_TOP_LEVEL:
            continue
        datasets.append(p)
    return sorted(datasets, key=lambda x: x.name)


def read_csv_df(path: Path, index_col: Optional[int] = None) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, index_col=index_col)
    except Exception:
        return None


def read_csv_records(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    df = read_csv_df(path)
    if df is None:
        return []
    if limit is not None:
        df = df.head(limit)
    # ensure 3-decimal formatting consistently in records too
    df = format_df_3dp(df)
    return df.to_dict(orient="records")


def format_df_3dp(df: pd.DataFrame) -> pd.DataFrame:
    """Format all numeric columns to 3 decimals (as strings)."""
    out = df.copy()
    for col in out.columns:
        # don't break non-numeric columns
        series = pd.to_numeric(out[col], errors="coerce")
        if series.notna().any():
            out[col] = [
                "" if pd.isna(v) else f"{float(v):.3f}"
                for v in series
            ]
    return out


def compute_best_cells(mean_df: pd.DataFrame) -> Dict[Tuple[int, str], bool]:
    """
    Returns mapping (row_index, column_name) -> True if best in column (ties included).
    Expects first col = label (model/ensemble name), others numeric.
    """
    best_map: Dict[Tuple[int, str], bool] = {}
    if mean_df is None or mean_df.empty:
        return best_map

    df = mean_df.copy()
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in df.columns[1:]:
        s = df[col]
        if s.dropna().empty:
            continue
        best_val = s.min(skipna=True) if col in LOWER_IS_BETTER else s.max(skipna=True)
        for idx, val in s.items():
            if pd.isna(val):
                continue
            if float(val) == float(best_val):
                best_map[(idx, col)] = True
    return best_map


def merge_mean_std_tables(mean_df: Optional[pd.DataFrame], std_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Builds a table where numeric cells are "mean ± std" (3dp).
    If std missing: numeric cells are "mean" (3dp).
    """
    if mean_df is None or mean_df.empty:
        return None

    m = mean_df.copy()
    s = std_df.copy() if (std_df is not None and not std_df.empty) else None

    key = m.columns[0]
    if s is None:
        # just format mean to 3dp
        return format_df_3dp(m)

    # Align by key if possible
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
                    rr[col] = f"{float(mv):.3f}"
                else:
                    rr[col] = f"{float(mv):.3f} ± {float(sv):.3f}"
            rows.append(rr)
        return pd.DataFrame(rows, columns=m.columns)

    # Fallback index alignment
    out = m.copy()
    for col in out.columns[1:]:
        mv = pd.to_numeric(m[col], errors="coerce")
        sv = pd.to_numeric(s[col], errors="coerce") if col in s.columns else pd.Series([math.nan] * len(m))
        col_out: List[str] = []
        for i in range(len(out)):
            a = mv.iloc[i] if i < len(mv) else math.nan
            b = sv.iloc[i] if i < len(sv) else math.nan
            if pd.isna(a):
                col_out.append("")
            elif pd.isna(b):
                col_out.append(f"{float(a):.3f}")
            else:
                col_out.append(f"{float(a):.3f} ± {float(b):.3f}")
        out[col] = col_out
    return out


def _df_to_table_payload(
    df: Optional[pd.DataFrame],
    best_map: Optional[Dict[Tuple[int, str], bool]] = None,
) -> Dict[str, Any]:
    """
    Payload for Jinja rendering:
      {
        "present": bool,
        "columns": [...],
        "rows": [{"cells":[{"value":..., "best":bool}, ...]}, ...]
      }
    """
    if df is None or df.empty:
        return {"present": False, "columns": [], "rows": []}

    columns = list(df.columns)
    rows: List[Dict[str, Any]] = []

    for ridx in range(len(df)):
        cells: List[Dict[str, Any]] = []
        for cidx, col in enumerate(columns):
            val = df.iloc[ridx, cidx]
            v = "" if pd.isna(val) else str(val)
            is_best = False
            if best_map is not None and cidx > 0:
                is_best = best_map.get((ridx, col), False)
            cells.append({"value": v, "best": bool(is_best)})
        rows.append({"cells": cells})

    return {"present": True, "columns": columns, "rows": rows}


# -----------------------------
# Plot generation (only if missing)
# -----------------------------
def ensure_class_counts_plot(exploratory_dir: Path) -> Optional[Path]:
    png = exploratory_dir / "ClassCountsBarPlot.png"
    if png.exists():
        return png

    csv_path = exploratory_dir / "ClassCounts.csv"
    if not csv_path.exists():
        return None

    try:
        import matplotlib.pyplot as plt

        df = pd.read_csv(csv_path)
        # try common patterns: either columns [Class, Count] or first two columns
        if df.shape[1] >= 2:
            x = df.iloc[:, 0].astype(str).tolist()
            y = pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0).tolist()
        else:
            return None

        plt.figure()
        plt.bar(x, y)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(png, dpi=DEFAULT_DPI)
        plt.close()
        return png
    except Exception as e:
        logging.warning("Failed to generate ClassCountsBarPlot.png: %s", e)
        return None


def ensure_feature_correlation_plot(exploratory_dir: Path) -> Optional[Path]:
    png = exploratory_dir / "FeatureCorrelations.png"
    if png.exists():
        return png

    csv_path = exploratory_dir / "FeatureCorrelations.csv"
    if not csv_path.exists():
        return None

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        df = pd.read_csv(csv_path, index_col=0)
        mat = df.values.astype(float)
        plt.figure()
        plt.imshow(mat, aspect="auto")
        plt.colorbar()
        # keep labels small; big matrices will be unreadable anyway
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(png, dpi=DEFAULT_DPI)
        plt.close()
        return png
    except Exception as e:
        logging.warning("Failed to generate FeatureCorrelations.png: %s", e)
        return None


class ReportPhaseJob:
    """
    Generates a multi-dataset STREAMLINE PDF report using Jinja2 + WeasyPrint.

    - Corrects filenames to match your tree
    - Adds model + ensemble performance tables (mean±std, median)
    - Applies 3dp formatting everywhere
    - Highlights best cells like old FPDF
    - Improves pagination via CSS in template
    - Generates a couple plots from CSV if missing
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        experiment_name: Optional[str] = None,
        experiment_path: Optional[str] = None,
        training: bool = True,
        make_pdf: bool = True,
        template_name: str = "report.html.j2",
    ):
        if experiment_path:
            self.experiment_root = Path(experiment_path)
            self.experiment_name = self.experiment_root.name
        else:
            if output_path is None or experiment_name is None:
                raise ValueError("Provide either experiment_path OR (output_path and experiment_name).")
            self.experiment_root = Path(output_path) / str(experiment_name)
            self.experiment_name = str(experiment_name)

        self.training = bool(training)
        self.make_pdf = bool(make_pdf)

        self.template_dir = Path(__file__).parent / "templates"
        self.template_name = template_name

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
        logging.info("Starting ReportPhaseJob for %s", self.experiment_root)

        generated_at = datetime.now()

        ds_dirs = list_datasets(self.experiment_root)
        datasets: List[Dict[str, Any]] = []
        for ds in ds_dirs:
            datasets.append(self._build_dataset_section(ds))

        comparisons = self._build_dataset_comparisons_section()

        env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = env.get_template(self.template_name)

        html_str = template.render(
            title="STREAMLINE Evaluation Report",
            experiment_name=self.experiment_name,
            experiment_root=safe_abs_posix(self.experiment_root),
            generated_at=generated_at.strftime("%Y-%m-%d %H:%M:%S"),
            streamline_version=version,
            training=self.training,
            outcome_type=self.outcome_type,
            metadata=self.metadata,
            # alg_info=self.alg_info,
            datasets=datasets,
            dataset_comparisons=comparisons,
        )

        out_dir = self.experiment_root / "reporting"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = out_dir / f"{self.experiment_name}_STREAMLINE_Report.pdf"

        if self.make_pdf:
            HTML(string=html_str, base_url=str(self.experiment_root.resolve().as_uri())).write_pdf(str(out_pdf))
            logging.info("Wrote PDF: %s", out_pdf)

        # completion marker
        try:
            jc = self.experiment_root / "jobsCompleted"
            jc.mkdir(exist_ok=True, parents=True)
            (jc / "job_p11_reporting.txt").write_text("complete")
        except Exception:
            pass

    def _build_dataset_section(self, dataset_dir: Path) -> Dict[str, Any]:
        ds_name = dataset_dir.name
        ds_slug = slugify(ds_name)

        exploratory_dir = dataset_dir / "exploratory"
        model_eval_dir = dataset_dir / "model_evaluation"
        ensemble_eval_dir = dataset_dir / "ensemble_evaluation"

        # generate a couple plots from CSV if missing
        if exploratory_dir.exists():
            ensure_class_counts_plot(exploratory_dir)
            ensure_feature_correlation_plot(exploratory_dir)

        figures: Dict[str, Optional[str]] = {}

        # Exploratory
        figures["class_counts"] = safe_abs_posix(exploratory_dir / "ClassCountsBarPlot.png") if (exploratory_dir / "ClassCountsBarPlot.png").exists() else None
        figures["feature_correlations"] = safe_abs_posix(exploratory_dir / "FeatureCorrelations.png") if (exploratory_dir / "FeatureCorrelations.png").exists() else None

        # Model evaluation summaries
        figures["summary_roc"] = safe_abs_posix(model_eval_dir / "Summary_ROC.png") if (model_eval_dir / "Summary_ROC.png").exists() else None
        figures["summary_prc"] = safe_abs_posix(model_eval_dir / "Summary_PRC.png") if (model_eval_dir / "Summary_PRC.png").exists() else None

        mb = model_eval_dir / "metricBoxplots"
        figures["compare_roc_auc"] = safe_abs_posix(mb / "Compare_ROC AUC.png") if (mb / "Compare_ROC AUC.png").exists() else None
        figures["compare_prc_auc"] = safe_abs_posix(mb / "Compare_PRC AUC.png") if (mb / "Compare_PRC AUC.png").exists() else None

        # Ensemble evaluation summaries (your tree: Summary_*_ensembles.png)
        figures["summary_roc_ensembles"] = safe_abs_posix(ensemble_eval_dir / "Summary_ROC_ensembles.png") if (ensemble_eval_dir / "Summary_ROC_ensembles.png").exists() else None
        figures["summary_prc_ensembles"] = safe_abs_posix(ensemble_eval_dir / "Summary_PRC_ensembles.png") if (ensemble_eval_dir / "Summary_PRC_ensembles.png").exists() else None

        # Feature importance plots (your tree uses: feature_importance/multisurf and feature_importance/mutualinformation)
        fi_mi = dataset_dir / "feature_importance" / "mutualinformation" / "TopAverageScores.png"
        fi_ms = dataset_dir / "feature_importance" / "multisurf" / "TopAverageScores.png"
        figures["fi_mutualinformation"] = safe_abs_posix(fi_mi) if fi_mi.exists() else None
        figures["fi_multisurf"] = safe_abs_posix(fi_ms) if fi_ms.exists() else None

        # Tables
        tables: Dict[str, Any] = {}
        tables["data_process_summary"] = read_csv_records(exploratory_dir / "DataProcessSummary.csv")
        tables["univariate_top10"] = read_csv_records(exploratory_dir / "univariate_analyses" / "Univariate_Significance.csv", limit=10)
        tables["informative_feature_summary"] = read_csv_records(dataset_dir / "feature_selection" / "InformativeFeatureSummary.csv")
        tables["runtimes"] = read_csv_records(dataset_dir / "runtimes.csv")

        # Performance tables (models + ensembles): mean±std together; median separate
        perf: Dict[str, Any] = {}

        # Models
        m_mean = read_csv_df(model_eval_dir / "Summary_performance_mean.csv", index_col=0)
        m_median = read_csv_df(model_eval_dir / "Summary_performance_median.csv", index_col=0)
        m_std = read_csv_df(model_eval_dir / "Summary_performance_std.csv", index_col=0)

        # reset to include algorithm column again
        if m_mean is not None:
            m_mean = m_mean.reset_index()
        if m_median is not None:
            m_median = m_median.reset_index()
        if m_std is not None:
            m_std = m_std.reset_index()

        m_best_map = compute_best_cells(m_mean) if m_mean is not None else {}
        m_mean_std = merge_mean_std_tables(m_mean, m_std)
        if m_median is not None:
            m_median = format_df_3dp(m_median)

        perf["models_mean_std"] = _df_to_table_payload(m_mean_std, best_map=m_best_map)
        perf["models_median"] = _df_to_table_payload(m_median, best_map=None)

        # Ensembles
        e_mean = read_csv_df(ensemble_eval_dir / "Ensembles_performance_mean.csv", index_col=0)
        e_median = read_csv_df(ensemble_eval_dir / "Ensembles_performance_median.csv", index_col=0)
        e_std = read_csv_df(ensemble_eval_dir / "Ensembles_performance_std.csv", index_col=0)

        if e_mean is not None:
            e_mean = e_mean.reset_index()
        if e_median is not None:
            e_median = e_median.reset_index()
        if e_std is not None:
            e_std = e_std.reset_index()

        e_best_map = compute_best_cells(e_mean) if e_mean is not None else {}
        e_mean_std = merge_mean_std_tables(e_mean, e_std)
        if e_median is not None:
            e_median = format_df_3dp(e_median)

        perf["ensembles_mean_std"] = _df_to_table_payload(e_mean_std, best_map=e_best_map)
        perf["ensembles_median"] = _df_to_table_payload(e_median, best_map=None)

        return {
            "dataset_name": ds_name,
            "dataset_slug": ds_slug,
            "dataset_dir": safe_abs_posix(dataset_dir),
            "figures": figures,
            "tables": tables,
            "perf": perf,
        }

    def _build_dataset_comparisons_section(self) -> Dict[str, Any]:
        dc_dir = self.experiment_root / "DatasetComparisons"
        if not dc_dir.exists():
            return {"present": False, "figures": {}, "tables": {}}

        figures: Dict[str, Optional[str]] = {}
        tables: Dict[str, Any] = {}

        box_dir = dc_dir / "dataCompBoxplots"
        figures["allmodels_roc_auc"] = safe_abs_posix(box_dir / "DataCompareAllModels_ROC AUC.png") if (box_dir / "DataCompareAllModels_ROC AUC.png").exists() else None
        figures["allmodels_prc_auc"] = safe_abs_posix(box_dir / "DataCompareAllModels_PRC AUC.png") if (box_dir / "DataCompareAllModels_PRC AUC.png").exists() else None

        # Table(s)
        tables["best_kw"] = read_csv_records(dc_dir / "BestCompare_KruskalWallis.csv")

        return {"present": True, "figures": figures, "tables": tables}
