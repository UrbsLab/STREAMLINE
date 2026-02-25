from __future__ import annotations

import json
import logging
import math
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from fpdf import FPDF

logger = logging.getLogger(__name__)

Number = Union[int, float]


# ============================================================
# Plot export helper (fallback-only; prefer precomputed PNGs)
# ============================================================

def _safe_plotly_to_png(fig, out_path: Path, scale: int = 2) -> bool:
    """
    Export a Plotly figure to PNG using kaleido.

    This is a fallback path only: the report prefers precomputed PNGs
    already present in the experiment output tree.
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
    Canonical formatting for numeric table cells.

    - ints -> "42"
    - floats -> fixed decimals unless abs(x) < sci_small (and non-zero), then scientific
    - numeric strings -> parsed & formatted
    - other strings -> unchanged
    """
    if x is None or _is_nan(x):
        return ""

    if isinstance(x, bool):
        return "True" if x else "False"

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

class ReportPaths:
    def __init__(self, reporting_dir: Path, data_json: Path, pdf: Path, figures_dir: Path, metadata_txt: Path):
        self.reporting_dir = reporting_dir
        self.data_json = data_json
        self.pdf = pdf
        self.figures_dir = figures_dir
        self.metadata_txt = metadata_txt


# ============================================================
# Main job
# ============================================================

class ReportPhaseJob:
    """
    Phase 11: Reporting (FPDF)

    Layout rules:
      - Cover page: legacy-style two-column parameter boxes.
      - Metadata also written to reporting/metadata.txt (plain text).

      - Per-dataset order:
          1) EDA - Page 1
             - Univariate (only if informative)
             - Class Balance + Missingness (grid)
             - Cleaning (C) and Engineering (E) Elements (text box)
          1b) Correlation Matrix (full page, if present)
          2) Feature Learning (all FI methods)
          3) Performance (combined models + ensembles; ensemble rows renamed with suffix; no wrap)
          4) Evaluation Results (summary ROC/PRC only; no per-algorithm ROC/PRC pages)
          5) Runtime Summary

    Rendering rules:
      - Prefer precomputed PNGs in the experiment output tree.
      - Fallback to plotting into reporting/figures only if originals are missing.
      - Keep Times / Times New Roman core fonts; sanitize text to ASCII-safe equivalents.
      - Do not render any placeholder panels for missing figures (skip silently).
      - No em-dashes and no ellipsis in outputs.
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

        self.title = f"STREAMLINE Testing Data Evaluation Report: {_now_iso_local()}"

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
            metadata_txt=reporting_dir / "metadata.txt",
        )

    # ----------------------------
    # File readers
    # ----------------------------
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

    def _read_pickle_if_exists(self, path: Path) -> Optional[Any]:
        try:
            if path.is_file():
                with path.open("rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning("Failed reading pickle %s: %r", path, e)
        return None

    # ----------------------------
    # Utilities: normalize/flatten/merge/pretty print
    # ----------------------------
    def _flatten_mapping(
        self,
        obj: Any,
        *,
        prefix: str = "",
        sep: str = " · ",
        max_depth: int = 5,
        _depth: int = 0,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if _depth > max_depth:
            if prefix:
                out[prefix] = str(obj)
            return out

        if isinstance(obj, dict):
            for k, v in obj.items():
                kk = str(k)
                new_prefix = f"{prefix}{sep}{kk}" if prefix else kk
                out.update(self._flatten_mapping(v, prefix=new_prefix, sep=sep, max_depth=max_depth, _depth=_depth + 1))
            return out

        if isinstance(obj, (list, tuple)):
            if prefix:
                out[prefix] = ", ".join(str(x) for x in obj)
            return out

        if prefix:
            out[prefix] = obj
        return out

    def _merge_union_dicts(self, *dicts: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for d in dicts:
            for k, v in (d or {}).items():
                if k not in out:
                    out[k] = v
                else:
                    if str(out[k]) != str(v):
                        alt_k = f"{k} (alt)"
                        i = 2
                        while alt_k in out:
                            alt_k = f"{k} (alt {i})"
                            i += 1
                        out[alt_k] = v
        return out

    def _normalize_bool(self, v: Any) -> Optional[bool]:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)) and v in (0, 1):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"true", "t", "yes", "y", "1"}:
                return True
            if s in {"false", "f", "no", "n", "0"}:
                return False
        return None

    def _as_compact_str(self, v: Any) -> str:
        if v is None:
            return "None"
        b = self._normalize_bool(v)
        if b is not None:
            return "True" if b else "False"
        if isinstance(v, (int, float)):
            return format_number(v, decimals=self.float_decimals)
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(str(x) for x in v) + "]"
        return str(v)

    def _is_timestampy_key(self, k: str) -> bool:
        s = k.lower()
        if any(x in s for x in ["generated_at", "generated at", "epoch", "timestamp", "time stamp"]):
            return True
        if re.search(r"\b20\d{2}[-_/]\d{2}[-_/]\d{2}\b", s):
            return True
        if re.search(r"\b\d{2}:\d{2}:\d{2}\b", s):
            return True
        return False

    def _pretty_cover_key(self, k: str) -> str:
        parts = [p.strip() for p in str(k).split("·")]
        last = parts[-1] if parts else str(k)
        last = last.strip().replace("_", " ")
        last = re.sub(r"\s+", " ", last).strip()
        return last

    def _should_drop_identifier_code(self, k: str) -> bool:
        s = k.lower()
        return any(x in s for x in ["identifier code", "identifier_code", "run id", "run_id", "uuid", "guid", "hash", "job id", "job_id"])

    def _write_metadata_text(self, *, cover_boxes: Dict[str, List[Tuple[str, str]]], enriched_meta: Dict[str, Any]) -> None:
        lines: List[str] = []
        lines.append(self.title)
        lines.append(f"Experiment Root: {self.exp_root}")
        lines.append(f"Experiment Name: {self.experiment_name}")
        lines.append(f"Generated At: {_now_iso_local()}")
        lines.append(f"STREAMLINE Version: {_try_streamline_version()}")
        lines.append("")

        lines.append("=== COVER PARAMETERS (grouped) ===")
        for section, items in (cover_boxes or {}).items():
            lines.append(f"[{section}]")
            for k, v in items:
                lines.append(f"- {k}: {v}")
            lines.append("")

        lines.append("=== FULL METADATA UNION (flattened) ===")
        for k in sorted(enriched_meta.keys(), key=lambda s: str(s).lower()):
            lines.append(f"{k}: {self._as_compact_str(enriched_meta.get(k))}")

        self.paths.metadata_txt.write_text("\n".join(lines))

    # ----------------------------
    # Dataset discovery
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
        ds: List[Path] = []
        for p in sorted(self.exp_root.iterdir()):
            if not p.is_dir():
                continue
            if p.name in ignore:
                continue
            if (p / "CVDatasets").is_dir():
                ds.append(p)
        return ds

    # ============================================================
    # Prefer-original figure resolution
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
        return self._first_existing([
            ds_dir / "exploratory" / "ClassCountsBarPlot.png",
            ds_dir / "exploratory" / "ClassCountsBarplot.png",
            ds_dir / "exploratory" / "ClassCounts.png",
        ])

    def _figure_path_exploratory_missingness(self, ds_dir: Path) -> Optional[str]:
        return self._first_existing([
            ds_dir / "exploratory" / "DataMissingness.png",
            ds_dir / "exploratory" / "Missingness.png",
            ds_dir / "exploratory" / "MissingnessTop25.png",
        ])

    def _figure_path_exploratory_correlation_matrix(self, ds_dir: Path) -> Optional[str]:
        return self._first_existing([
            ds_dir / "exploratory" / "FeatureCorrelationMatrix.png",
            ds_dir / "exploratory" / "FeatureCorrelation.png",
            ds_dir / "exploratory" / "CorrelationMatrix.png",
            ds_dir / "exploratory" / "CorrelationHeatmap.png",
            ds_dir / "exploratory" / "feature_correlation" / "CorrelationMatrix.png",
            ds_dir / "exploratory" / "feature_correlation" / "FeatureCorrelationMatrix.png",
            ds_dir / "exploratory" / "FeatureCorrelation" / "CorrelationMatrix.png",
        ])

    def _figure_path_model_summary_roc_prc(self, ds_dir: Path, kind: str) -> Optional[str]:
        if kind.lower() == "roc":
            return self._first_existing([ds_dir / "model_evaluation" / "Summary_ROC.png"])
        return self._first_existing([ds_dir / "model_evaluation" / "Summary_PRC.png"])

    def _figure_path_model_metric_boxplot(self, ds_dir: Path, preferred_metric: str) -> Optional[str]:
        return self._first_existing([
            ds_dir / "model_evaluation" / "metricBoxplots" / f"Compare_{preferred_metric}.png",
        ])

    def _figure_path_ensemble_summary(self, ds_dir: Path, kind: str) -> Optional[str]:
        if kind.lower() == "roc":
            return self._first_existing([ds_dir / "ensemble_evaluation" / "Summary_ROC_ensembles.png"])
        return self._first_existing([ds_dir / "ensemble_evaluation" / "Summary_PRC_ensembles.png"])

    def _figure_paths_fs_top_scores(self, ds_dir: Path) -> List[Dict[str, str]]:
        base = ds_dir / "feature_importance"
        if not base.is_dir():
            return []
        hits = sorted(base.glob("*/TopAverageScores.png"), key=lambda p: p.parent.name.lower())
        out: List[Dict[str, str]] = []
        for p in hits:
            try:
                if p.is_file():
                    out.append({"method": p.parent.name, "path": str(p)})
            except Exception:
                continue
        return out

    def _figure_path_dataset_comparisons_any(self) -> Optional[str]:
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

    def _plot_correlation_matrix_from_csv(self, corr_df: pd.DataFrame, title: str):
        import plotly.express as px  # type: ignore

        df = corr_df.copy()
        if df.shape[1] > 1 and str(df.columns[0]).lower() in {"unnamed: 0", "feature", "var", "variable"}:
            df = df.set_index(df.columns[0])
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

        fig = px.imshow(df, aspect="auto", title=title, color_continuous_scale="RdBu", zmin=-1, zmax=1)
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
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

    def _univariate_is_informative(self, uni: Optional[pd.DataFrame]) -> bool:
        if uni is None or uni.empty:
            return False
        if len(uni) < 3:
            return False

        low = {c.lower(): c for c in uni.columns}
        pcol = None
        for cand in ["p", "p-value", "p_value", "pvalue", "pval"]:
            if cand in low:
                pcol = low[cand]
                break
        if pcol is None:
            return uni.shape[1] >= 2 and len(uni) >= 5

        try:
            pv = pd.to_numeric(uni[pcol], errors="coerce").dropna()
            if pv.empty:
                return False
            return bool((pv < 0.10).any())
        except Exception:
            return False

    # ============================================================
    # Combine model + ensemble performance (Mean +/- SD)
    #   - NO new column
    #   - ensemble rows renamed: "<Algorithm> - Ensemble"
    # ============================================================

    def _combine_model_and_ensemble_perf(
        self,
        models_mean_std: Dict[str, Any],
        ensembles_mean_std: Dict[str, Any],
        *,
        drop_if_needed: Sequence[str] = ("Brier Score",),
        ensemble_suffix: str = " - Ensemble",
    ) -> Dict[str, Any]:
        ms = models_mean_std or {}
        es = ensembles_mean_std or {}

        if not ms.get("present") and not es.get("present"):
            return {"present": False}

        ms_cols = list(ms.get("columns") or []) if ms.get("present") else []
        es_cols = list(es.get("columns") or []) if es.get("present") else []

        ms_metrics = [c for c in ms_cols if c != "Algorithm"]
        es_metrics = [c for c in es_cols if c != "Algorithm"]

        metrics: List[str] = []
        for c in ms_metrics + es_metrics:
            if c not in metrics:
                metrics.append(c)

        # Too wide? columns are Algorithm + metrics
        too_wide = (1 + len(metrics)) >= 11
        if too_wide:
            for drop_col in drop_if_needed:
                if drop_col in metrics:
                    metrics.remove(drop_col)
                    break

        def _row_map(mean_std_tbl: Dict[str, Any]) -> Dict[str, Dict[str, Tuple[Any, Any]]]:
            out: Dict[str, Dict[str, Tuple[Any, Any]]] = {}
            if not mean_std_tbl.get("present"):
                return out
            cols = list(mean_std_tbl.get("columns") or [])
            for row in mean_std_tbl.get("rows") or []:
                cells = row.get("cells") or []
                if not cells:
                    continue
                alg = str((cells[0] or {}).get("value", "")).strip()
                if not alg:
                    continue
                m: Dict[str, Tuple[Any, Any]] = {}
                for j, col in enumerate(cols):
                    if col == "Algorithm":
                        continue
                    if j >= len(cells):
                        continue
                    v = (cells[j] or {}).get("value")
                    if isinstance(v, tuple) and len(v) == 2:
                        m[col] = (v[0], v[1])
                out[alg] = m
            return out

        ms_map = _row_map(ms)
        es_map = _row_map(es)

        columns = ["Algorithm"] + metrics
        rows: List[Dict[str, Any]] = []

        # Models
        for alg in sorted(ms_map.keys(), key=lambda s: s.lower()):
            metric_map = ms_map[alg]
            cells: List[Dict[str, Any]] = [{"value": alg, "best": False}]
            for met in metrics:
                if met in metric_map:
                    mv, sv = metric_map[met]
                    cells.append({"value": (mv, sv), "best": False})
                else:
                    cells.append({"value": ("", ""), "best": False})
            rows.append({"cells": cells})

        # Ensembles (renamed)
        for alg in sorted(es_map.keys(), key=lambda s: s.lower()):
            alg2 = f"{alg}{ensemble_suffix}"
            metric_map = es_map[alg]
            cells = [{"value": alg2, "best": False}]
            for met in metrics:
                if met in metric_map:
                    mv, sv = metric_map[met]
                    cells.append({"value": (mv, sv), "best": False})
                else:
                    cells.append({"value": ("", ""), "best": False})
            rows.append({"cells": cells})

        highlight_metric = None
        for cand in ["Balanced Accuracy", "ROC AUC", "PRC AUC", "Accuracy"]:
            if cand in metrics:
                highlight_metric = cand
                break

        return {
            "present": True,
            "columns": columns,
            "rows": rows,
            "highlight_metric": highlight_metric,
            "best_algorithm": None,
        }

    # ============================================================
    # Combine model + ensemble medians (same renaming)
    # ============================================================

    def _combine_model_and_ensemble_median(
        self,
        models_median: Dict[str, Any],
        ensembles_median: Dict[str, Any],
        *,
        drop_if_needed: Sequence[str] = ("Brier Score",),
        ensemble_suffix: str = " - Ensemble",
    ) -> Dict[str, Any]:
        mm = models_median or {}
        em = ensembles_median or {}

        if not mm.get("present") and not em.get("present"):
            return {"present": False}

        mm_cols = list(mm.get("columns") or []) if mm.get("present") else []
        em_cols = list(em.get("columns") or []) if em.get("present") else []

        if not mm_cols and not em_cols:
            return {"present": False}

        def infer_alg_col(columns: List[str]) -> str:
            low = {c.lower(): c for c in columns}
            for key in ["ml algorithm", "ml_algorithm", "algorithm", "model"]:
                if key in low:
                    return low[key]
            return columns[0] if columns else "Algorithm"

        mm_alg = infer_alg_col(mm_cols) if mm_cols else "Algorithm"
        em_alg = infer_alg_col(em_cols) if em_cols else "Algorithm"

        mm_metrics = [c for c in mm_cols if c != mm_alg]
        em_metrics = [c for c in em_cols if c != em_alg]

        metrics: List[str] = []
        for c in mm_metrics + em_metrics:
            if c not in metrics:
                metrics.append(c)

        too_wide = (1 + len(metrics)) >= 11
        if too_wide:
            for drop_col in drop_if_needed:
                if drop_col in metrics:
                    metrics.remove(drop_col)
                    break

        def to_map(tbl: Dict[str, Any], alg_col: str) -> Dict[str, Dict[str, Any]]:
            out: Dict[str, Dict[str, Any]] = {}
            if not tbl.get("present"):
                return out
            cols = list(tbl.get("columns") or [])
            rows = list(tbl.get("rows") or [])
            if not cols or not rows:
                return out
            idx = {c: i for i, c in enumerate(cols)}
            alg_i = idx.get(alg_col, 0)

            for r in rows:
                if r is None:
                    continue
                rr = list(r)
                if alg_i >= len(rr):
                    continue
                alg = str(rr[alg_i]).strip()
                if not alg:
                    continue
                m: Dict[str, Any] = {}
                for met in metrics:
                    j = idx.get(met)
                    if j is None or j >= len(rr):
                        continue
                    m[met] = rr[j]
                out[alg] = m
            return out

        mm_map = to_map(mm, mm_alg)
        em_map = to_map(em, em_alg)

        columns = ["Algorithm"] + metrics
        out_rows: List[List[Any]] = []

        for alg in sorted(mm_map.keys(), key=lambda s: s.lower()):
            m = mm_map[alg]
            out_rows.append([alg] + [m.get(met, "") for met in metrics])

        for alg in sorted(em_map.keys(), key=lambda s: s.lower()):
            alg2 = f"{alg}{ensemble_suffix}"
            m = em_map[alg]
            out_rows.append([alg2] + [m.get(met, "") for met in metrics])

        return {"present": True, "columns": columns, "rows": out_rows}

    # ============================================================
    # Cover page grouping (legacy categories)
    # ============================================================

    def _categorize_cover_items(
        self,
        meta_pickle_flat: Dict[str, Any],
        run_params_flat: Dict[str, Any],
        *,
        exp_root: Path,
    ) -> Dict[str, List[Tuple[str, str]]]:
        combined = self._merge_union_dicts(run_params_flat or {}, meta_pickle_flat or {})

        def add(cat: str, k: str, v: Any):
            if self._is_timestampy_key(k):
                return
            if self._should_drop_identifier_code(k):
                return
            pretty_k = self._pretty_cover_key(k)
            if pretty_k.strip() == "":
                return
            out.setdefault(cat, []).append((pretty_k, self._as_compact_str(v)))

        out: Dict[str, List[Tuple[str, str]]] = {}

        ds_names = [p.name for p in self._list_datasets()]
        if ds_names:
            items = [(f"D{i+1}", nm) for i, nm in enumerate(ds_names)]
            out["Target Dataset(s):"] = [(f"{k}", f"= {v}") for k, v in items]

        for key in [
            "data path", "input path", "dataset path",
            "output path",
            "experiment name",
            "class label",
            "instance label",
            "match label",
            "ignored features",
            "specified categorical features",
            "specified quantitative features",
        ]:
            for kk in list(combined.keys()):
                if str(kk).strip().lower() == key:
                    add("Target Data Settings:", kk, combined[kk])

        for kk, vv in combined.items():
            k = str(kk).lower()
            if any(x in k for x in ["cv", "partition", "seed", "random", "categorical cutoff", "statistical", "significance", "notebook"]):
                if any(x in k for x in ["plot", "export", "roc", "prc", "figure", "boxplot"]):
                    continue
                add("General Pipeline Settings:", kk, vv)

        for kk, vv in combined.items():
            k = str(kk).lower()
            if any(x in k for x in ["missing", "imput", "scale", "correlation", "describe", "univariate", "eda", "processing", "clean"]):
                if any(x in k for x in ["feature importance", "feature_selection", "feature selection", "multisurf", "mutual", "turf"]):
                    continue
                add("EDA and Processing Settings:", kk, vv)

        for kk, vv in combined.items():
            k = str(kk).lower()
            if any(x in k for x in ["feature importance", "feature selection", "multisurf", "mutual", "turf", "max features", "top features", "filter poor"]):
                add("Feature Importance/Selection Settings:", kk, vv)

        algo_items: List[Tuple[str, str]] = []
        algo_keys = [
            "logistic", "naive", "bayes", "random forest", "svm", "support vector",
            "xgboost", "gradient boosting", "lightgbm", "catboost",
            "decision tree", "elastic", "knn", "k-nearest",
            "ann", "neural", "exstracs", "xcs", "elcs", "genetic programming",
        ]
        for kk, vv in combined.items():
            k = str(kk).lower()
            if any(a in k for a in algo_keys):
                b = self._normalize_bool(vv)
                if b is None:
                    continue
                algo_items.append((self._pretty_cover_key(kk), "True" if b else "False"))
        if algo_items:
            algo_items.sort(key=lambda t: (t[1] != "True", t[0].lower()))
            out["ML Modeling Algorithms:"] = algo_items

        for kk, vv in combined.items():
            k = str(kk).lower()
            if any(x in k for x in ["primary metric", "hyperparameter", "trials", "timeout", "subsample", "uniform feature importance"]):
                add("Modeling Settings:", kk, vv)

        for kk, vv in combined.items():
            k = str(kk).lower()
            if any(x in k for x in ["lcs", "xcs", "elcs", "exstracs", "rule population", "training iterations", "nu"]):
                add("LCS Settings (eLCS, XCS, ExSTraCS):", kk, vv)

        for kk, vv in combined.items():
            k = str(kk).lower()
            if any(x in k for x in ["export", "roc", "prc", "boxplot", "figure", "plot", "correlation", "top model features", "metric weighting"]):
                add("Stats and Figure Settings:", kk, vv)

        for cat, items in list(out.items()):
            seen = set()
            dedup: List[Tuple[str, str]] = []
            for k, v in items:
                if k in seen:
                    continue
                seen.add(k)
                dedup.append((k, v))
            out[cat] = dedup

        ordered = [
            "General Pipeline Settings:",
            "EDA and Processing Settings:",
            "Feature Importance/Selection Settings:",
            "ML Modeling Algorithms:",
            "Modeling Settings:",
            "LCS Settings (eLCS, XCS, ExSTraCS):",
            "Stats and Figure Settings:",
            "Target Dataset(s):",
            "Target Data Settings:",
        ]
        out2: Dict[str, List[Tuple[str, str]]] = {}
        for cat in ordered:
            if cat in out and out[cat]:
                out2[cat] = out[cat]
        for cat, items in out.items():
            if cat not in out2 and items:
                out2[cat] = items
        return out2

    # ============================================================
    # Data assembly (tables + figs)
    # ============================================================

    def _collect_dataset_block(self, ds_dir: Path) -> Dict[str, Any]:
        name = ds_dir.name
        figs: Dict[str, Any] = {}

        explore = ds_dir / "exploratory"
        class_counts = self._read_csv_if_exists(explore / "ClassCounts.csv")
        missingness = self._read_csv_if_exists(explore / "DataMissingness.csv")
        univariate = self._read_csv_if_exists(explore / "univariate_analyses" / "Univariate_Significance.csv")

        # Correlation matrix
        corr_png = self._figure_path_exploratory_correlation_matrix(ds_dir) or ""
        if not corr_png:
            corr_csv = self._first_existing([
                explore / "FeatureCorrelationMatrix.csv",
                explore / "CorrelationMatrix.csv",
                explore / "FeatureCorrelation.csv",
                explore / "feature_correlation" / "CorrelationMatrix.csv",
                explore / "FeatureCorrelation" / "CorrelationMatrix.csv",
            ])
            if corr_csv:
                try:
                    corr_df = pd.read_csv(corr_csv)
                    fig = self._plot_correlation_matrix_from_csv(corr_df, f"{name}: Feature Correlation Matrix")
                    out = self.paths.figures_dir / f"{name}_corr_matrix.png"
                    if _safe_plotly_to_png(fig, out):
                        corr_png = str(out)
                except Exception as e:
                    logger.warning("Correlation matrix fallback plot failed for %s: %r", name, e)
        figs["correlation_matrix"] = corr_png

        # Univariate (optional)
        uni_use = self._univariate_is_informative(univariate)
        univariate_top10 = univariate.head(10) if (uni_use and univariate is not None and not univariate.empty) else None

        # Performance tables
        model_eval = ds_dir / "model_evaluation"
        summary_mean = self._read_csv_if_exists(model_eval / "Summary_performance_mean.csv")
        summary_std = self._read_csv_if_exists(model_eval / "Summary_performance_std.csv")
        summary_median = self._read_csv_if_exists(model_eval / "Summary_performance_median.csv")

        ens_eval = ds_dir / "ensemble_evaluation"
        ens_mean = self._read_csv_if_exists(ens_eval / "Ensembles_performance_mean.csv")
        ens_std = self._read_csv_if_exists(ens_eval / "Ensembles_performance_std.csv")
        ens_median = self._read_csv_if_exists(ens_eval / "Ensembles_performance_median.csv")

        # Feature learning / selection tables
        feat_sel = self._read_csv_if_exists(ds_dir / "feature_selection" / "InformativeFeatureSummary.csv")
        runtimes = self._read_csv_if_exists(ds_dir / "runtimes.csv")

        # Figures (EDA)
        figs["class_balance"] = self._figure_path_exploratory_class_balance(ds_dir) or ""
        if not figs["class_balance"] and class_counts is not None and not class_counts.empty:
            try:
                fig = self._plot_class_counts(class_counts, f"{name}: Class Balance")
                out = self.paths.figures_dir / f"{name}_class_balance.png"
                if _safe_plotly_to_png(fig, out):
                    figs["class_balance"] = str(out)
            except Exception as e:
                logger.warning("Class balance plot failed for %s: %r", name, e)

        figs["missingness"] = self._figure_path_exploratory_missingness(ds_dir) or ""
        if not figs["missingness"] and missingness is not None and not missingness.empty:
            try:
                fig = self._plot_missingness(missingness, f"{name}: Missingness (Top 25 Features)")
                out = self.paths.figures_dir / f"{name}_missingness_top25.png"
                if _safe_plotly_to_png(fig, out):
                    figs["missingness"] = str(out)
            except Exception as e:
                logger.warning("Missingness plot failed for %s: %r", name, e)

        # CV distribution boxplot
        figs["models_cv_box"] = ""
        chosen_metric = None
        if summary_mean is not None and not summary_mean.empty:
            for preferred in ["Balanced Accuracy", "ROC AUC", "PRC AUC", "Accuracy"]:
                if preferred in summary_mean.columns:
                    chosen_metric = preferred
                    break
        if chosen_metric:
            box = self._figure_path_model_metric_boxplot(ds_dir, chosen_metric)
            if box:
                figs["models_cv_box"] = box

        # Summary curves only
        figs["models_roc_overlay"] = self._figure_path_model_summary_roc_prc(ds_dir, "roc") or ""
        figs["models_prc_overlay"] = self._figure_path_model_summary_roc_prc(ds_dir, "prc") or ""
        figs["ensembles_roc"] = self._figure_path_ensemble_summary(ds_dir, "roc") or ""
        figs["ensembles_prc"] = self._figure_path_ensemble_summary(ds_dir, "prc") or ""

        # Feature learning figures (all methods)
        figs["fi_top_scores"] = self._figure_paths_fs_top_scores(ds_dir)

        # Tables (mean/std + median)
        models_mean_std = self._build_mean_std_table(
            summary_mean,
            summary_std,
            highlight_metric_candidates=["Balanced Accuracy", "ROC AUC", "PRC AUC", "Accuracy"],
        )
        ensembles_mean_std = self._build_mean_std_table(
            ens_mean,
            ens_std,
            highlight_metric_candidates=["Balanced Accuracy", "ROC AUC", "PRC AUC", "Accuracy"],
        )
        combined_mean_std = self._combine_model_and_ensemble_perf(
            models_mean_std,
            ensembles_mean_std,
            drop_if_needed=("Brier Score",),
            ensemble_suffix=" - Ensemble",
        )

        models_median = self._build_plain_table(summary_median, max_rows=200)
        ensembles_median = self._build_plain_table(ens_median, max_rows=200)
        combined_median = self._combine_model_and_ensemble_median(
            models_median,
            ensembles_median,
            drop_if_needed=("Brier Score",),
            ensemble_suffix=" - Ensemble",
        )

        return {
            "dataset_name": name,
            "dataset_dir": str(ds_dir),
            "tables": {
                "univariate_top10": self._build_plain_table(univariate_top10, max_rows=10),
                "informative_feature_summary": self._build_plain_table(feat_sel, max_rows=200),
                "runtimes": self._build_plain_table(runtimes, max_rows=500),
            },
            "perf": {
                "combined_mean_std": combined_mean_std,
                "combined_median": combined_median,
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
        any_plot = self._figure_path_dataset_comparisons_any()
        if any_plot:
            figs["overview"] = any_plot

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
        pdf = _StreamlinePDF(
            title=str(report_data.get("title", "")),
            streamline_version=str(report_data.get("streamline_version", "")),
            float_decimals=self.float_decimals,
        )
        pdf.alias_nb_pages()

        # COVER
        pdf.add_page()
        pdf.cover_banner_title(str(report_data.get("title", "")))
        pdf.cover_two_column_boxes(report_data.get("cover_boxes", {}) or {})

        # DATASETS
        for ds in report_data.get("datasets", []) or []:
            ds_name = str(ds.get("dataset_name", ""))
            ds_dir = str(ds.get("dataset_dir", ""))

            figs = ds.get("figures", {}) or {}
            tables = ds.get("tables", {}) or {}
            perf = ds.get("perf", {}) or {}

            # -------------------------
            # EDA - PAGE 1
            # -------------------------
            pdf.add_page()
            pdf.panel_title(f"Dataset: {ds_name}")
            pdf.set_font("Times", "", 10)
            if ds_dir:
                pdf.multi_cell(0, 5, pdf.s(ds_dir))
                pdf.ln(1.0)

            pdf.panel_title("EDA")

            # Univariate (optional)
            uni = tables.get("univariate_top10", {}) or {}
            if uni.get("present"):
                pdf.subheader("Univariate Analysis (Top 10)")
                pdf.draw_table(uni.get("columns", []), uni.get("rows", []), max_rows=10, no_wrap=True)

            # EDA grid (skip missing panels silently)
            pdf.figure_grid_2x2(
                titles=["Class Balance", "Missingness (Top 25 Features)", "", ""],
                paths=[figs.get("class_balance") or None, figs.get("missingness") or None, None, None],
                cell_h=66.0,
                gap=4.0,
                title_h=6.0,
                keep_aspect=True,
            )

            # Cleaning/Engineering elements
            pdf.panel_title("Cleaning (C) and Engineering (E) Elements")
            pdf.cleaning_engineering_box(
                [
                    "C1 - Remove instances with no outcome and features to ignore",
                    "E1 - Add missingness features",
                    "C2 - Remove features with invariance or high missingness",
                    "C3 - Remove instances with high missingness",
                    "E2 - Add one-hot-encoding of categorical features",
                    "C4 - Remove highly correlated features",
                ]
            )

            # Correlation matrix full page (if present)
            corr = figs.get("correlation_matrix") or ""
            if corr and Path(corr).exists():
                pdf.add_page()
                pdf.panel_title(f"Dataset: {ds_name}")
                pdf.panel_title("Feature Correlation Matrix")
                pdf.figure_single("", corr, h=175.0, title_h=0.0, keep_aspect=True)

            # -------------------------
            # Feature Learning
            # -------------------------
            pdf.add_page()
            pdf.panel_title(f"Dataset: {ds_name}")
            pdf.panel_title("Feature Learning")

            fi_list = figs.get("fi_top_scores") or []
            norm: List[Dict[str, str]] = []
            for item in fi_list:
                if isinstance(item, dict) and "path" in item and item.get("path"):
                    p = str(item["path"])
                    if Path(p).exists():
                        norm.append({"method": str(item.get("method") or "method"), "path": p})

            if norm:
                if len(norm) == 1:
                    pdf.figure_single(
                        f"Top Scores ({norm[0]['method']})", norm[0]["path"], h=130.0, title_h=6.0, keep_aspect=True
                    )
                elif len(norm) == 2:
                    pdf.figure_row_2(
                        titles=[f"Top Scores ({norm[0]['method']})", f"Top Scores ({norm[1]['method']})"],
                        paths=[norm[0]["path"], norm[1]["path"]],
                        h=95.0,
                        gap=4.0,
                        title_h=6.0,
                        keep_aspect=True,
                    )
                else:
                    for i in range(0, len(norm), 4):
                        chunk = norm[i : i + 4]
                        titles = [f"Top Scores ({c['method']})" for c in chunk]
                        paths = [c["path"] for c in chunk]
                        while len(titles) < 4:
                            titles.append("")
                            paths.append(None)
                        pdf.figure_grid_2x2(
                            titles=titles,
                            paths=paths,
                            cell_h=66.0,
                            gap=4.0,
                            title_h=6.0,
                            keep_aspect=True,
                        )
                        if i + 4 < len(norm):
                            pdf.add_page()
                            pdf.panel_title(f"Dataset: {ds_name}")
                            pdf.panel_title("Feature Learning")

            inf = tables.get("informative_feature_summary", {}) or {}
            if inf.get("present"):
                pdf.panel_title("Informative Feature Summary")
                pdf.draw_table(inf.get("columns", []), inf.get("rows", []), max_rows=200, no_wrap=True)

            # -------------------------
            # Performance (combined mean/std + combined median)
            # -------------------------
            pdf.add_page()
            pdf.panel_title(f"Dataset: {ds_name}")
            pdf.panel_title("Performance (Cross-Validation)")

            cmb = perf.get("combined_mean_std", {}) or {}
            if cmb.get("present"):
                pdf.subheader("Model and Ensemble Performance (Mean plus SD)")
                pdf.draw_mean_std_table(cmb, no_wrap=True)

            med = perf.get("combined_median", {}) or {}
            if med.get("present"):
                pdf.subheader("Median (Combined)")
                pdf.draw_table(med.get("columns", []), med.get("rows", []), max_rows=200, no_wrap=True)

            cv_box = figs.get("models_cv_box") or ""
            if cv_box and Path(cv_box).exists():
                pdf.panel_title("Performance Distribution")
                pdf.figure_single("Model Comparison", cv_box, h=110.0, title_h=6.0, keep_aspect=True)

            # -------------------------
            # Evaluation results (summary only)
            # -------------------------
            pdf.add_page()
            pdf.panel_title(f"Dataset: {ds_name}")
            pdf.panel_title("Evaluation Results (Curves)")

            roc = figs.get("models_roc_overlay") or ""
            prc = figs.get("models_prc_overlay") or ""
            eroc = figs.get("ensembles_roc") or ""
            eprc = figs.get("ensembles_prc") or ""

            if (roc and Path(roc).exists()) or (prc and Path(prc).exists()):
                pdf.figure_row_2(
                    titles=["Summary ROC", "Summary PRC"],
                    paths=[
                        roc if (roc and Path(roc).exists()) else None,
                        prc if (prc and Path(prc).exists()) else None,
                    ],
                    h=85.0,
                    gap=4.0,
                    title_h=6.0,
                    keep_aspect=True,
                )

            if (eroc and Path(eroc).exists()) or (eprc and Path(eprc).exists()):
                pdf.panel_title("Ensembles (Summary Curves)")
                pdf.figure_row_2(
                    titles=["Ensembles ROC", "Ensembles PRC"],
                    paths=[
                        eroc if (eroc and Path(eroc).exists()) else None,
                        eprc if (eprc and Path(eprc).exists()) else None,
                    ],
                    h=85.0,
                    gap=4.0,
                    title_h=6.0,
                    keep_aspect=True,
                )

            # -------------------------
            # Runtimes
            # -------------------------
            pdf.add_page()
            pdf.panel_title(f"Dataset: {ds_name}")
            pdf.panel_title("Runtime Summary")
            rt = tables.get("runtimes", {}) or {}
            if rt.get("present"):
                pdf.draw_table(rt.get("columns", []), rt.get("rows", []), max_rows=500, no_wrap=True)

        # DATASET COMPARISONS
        dc = report_data.get("dataset_comparisons", {}) or {}
        if dc.get("present"):
            pdf.add_page()
            pdf.panel_title("Dataset Comparisons")

            overview = (dc.get("figures", {}) or {}).get("overview") or ""
            if overview and Path(overview).exists():
                pdf.figure_single("Comparison Overview (All Datasets)", overview, h=120.0, title_h=6.0, keep_aspect=True)

            kw = (dc.get("tables", {}) or {}).get("best_kw", {}) or {}
            if kw.get("present"):
                pdf.panel_title("Best Comparisons - Kruskal-Wallis")
                pdf.draw_table(kw.get("columns", []), kw.get("rows", []), max_rows=200, no_wrap=True)

            mw = (dc.get("tables", {}) or {}).get("best_mw", {}) or {}
            if mw.get("present"):
                pdf.panel_title("Best Comparisons - Mann-Whitney U")
                pdf.draw_table(mw.get("columns", []), mw.get("rows", []), max_rows=200, no_wrap=True)

            wx = (dc.get("tables", {}) or {}).get("best_wx", {}) or {}
            if wx.get("present"):
                pdf.panel_title("Best Comparisons - Wilcoxon Rank-Sum")
                pdf.draw_table(wx.get("columns", []), wx.get("rows", []), max_rows=200, no_wrap=True)

        pdf.output(str(self.paths.pdf))

    # ----------------------------
    # Runtime bookkeeping
    # ----------------------------
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

        meta_pickle_obj = self._read_pickle_if_exists(self.exp_root / "metadata.pickle")
        run_params_obj = self._read_pickle_if_exists(self.exp_root / "run_params.pickle")

        meta_pickle_dict: Dict[str, Any] = meta_pickle_obj if isinstance(meta_pickle_obj, dict) else (
            {"metadata.pickle": str(meta_pickle_obj)} if meta_pickle_obj is not None else {}
        )
        run_params_dict: Dict[str, Any] = run_params_obj if isinstance(run_params_obj, dict) else (
            {"run_params.pickle": str(run_params_obj)} if run_params_obj is not None else {}
        )

        meta_flat = self._flatten_mapping(meta_pickle_dict, sep=" · ", max_depth=6) if meta_pickle_dict else {}
        params_flat = self._flatten_mapping(run_params_dict, sep=" · ", max_depth=6) if run_params_dict else {}

        cover_boxes = self._categorize_cover_items(meta_flat, params_flat, exp_root=self.exp_root)

        base_meta: Dict[str, Any] = {
            "Experiment Root": str(self.exp_root),
            "Output Path": str(self.exp_root.parent),
            "Experiment Name": self.experiment_name,
            "Datasets Found": len(datasets),
            "Generated At": _now_iso_local(),
        }
        if self.outcome_label:
            base_meta["Outcome Label"] = self.outcome_label
        if self.instance_label:
            base_meta["Instance Label"] = self.instance_label
        if self.outcome_type:
            base_meta["Outcome Type"] = self.outcome_type

        enriched_meta = self._merge_union_dicts(base_meta, meta_flat, params_flat)

        report_data: Dict[str, Any] = {
            "title": self.title,
            "experiment_name": self.experiment_name,
            "experiment_root": str(self.exp_root),
            "generated_at": _now_iso_local(),
            "generated_at_epoch": int(time.time()),
            "streamline_version": _try_streamline_version(),
            "metadata": enriched_meta,
            "metadata_pickle_flat": meta_flat,
            "run_params_flat": params_flat,
            "cover_boxes": cover_boxes,
            "datasets": dataset_blocks,
            "dataset_comparisons": dc_block,
        }

        self.paths.data_json.write_text(json.dumps(report_data, indent=2))
        self._write_metadata_text(cover_boxes=cover_boxes, enriched_meta=enriched_meta)

        if self.make_pdf:
            self._write_pdf(report_data)

        jc = self.exp_root / "jobsCompleted"
        jc.mkdir(exist_ok=True)
        (jc / "job_reporting.txt").write_text("complete")

        self.save_runtime()
        logger.info("Phase 11 reporting complete: %s", self.paths.pdf)


# ============================================================
# PDF renderer (Times; ASCII sanitization; skip missing figures)
# ============================================================

class _StreamlinePDF(FPDF):
    """
    Keep core Times font, but sanitize smart punctuation to ASCII-safe equivalents.

    Also:
      - Do not draw placeholder panels for missing figures (skip silently).
      - No em-dashes and no ellipsis in rendered strings.
      - Performance tables: no wrapping; use truncation and tuned column widths.
    """

    def __init__(self, *, title: str, streamline_version: str, float_decimals: int = 3):
        super().__init__(orientation="P", unit="mm", format="A4")
        self._title = title
        self._streamline_version = streamline_version
        self._decimals = int(float_decimals)

        self.set_margins(10, 10, 10)
        self.set_auto_page_break(auto=True, margin=14)
        self.set_line_width(0.2)

        self._use_running_header = False

        # table layout
        self._tbl_pad_x = 1.2
        self._tbl_pad_y = 0.8
        self._tbl_line_h = 3.4
        self._gap_after_table = 1.2

        # figure panel padding
        self._panel_pad = 2.0
        self._panel_title_text_pad_x = 1.4
        self._panel_title_text_pad_y = 1.4
        self._panel_content_pad_top = 2.2

        self.set_font("Times", "", 10)

    # -----------------------------
    # Sanitization (no smart punctuation, no em-dash, no ellipsis)
    # -----------------------------
    def s(self, txt: Any) -> str:
        if txt is None:
            return ""
        t = str(txt)

        repl = {
            "\u201c": '"',
            "\u201d": '"',
            "\u2018": "'",
            "\u2019": "'",
            "\u2013": "-",
            "\u2014": "-",
            "\u2022": "-",
            "\u00A0": " ",
            "\u2026": "",
        }
        for k, v in repl.items():
            t = t.replace(k, v)

        t = t.encode("latin-1", "ignore").decode("latin-1")
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\s+\n", "\n", t)
        return t.strip()

    # -----------------------------
    # Header / Footer
    # -----------------------------
    def header(self):
        if not self._use_running_header:
            return
        self.set_font("Times", "", 9)
        x = self.l_margin
        y = self.t_margin
        w = self.w - self.l_margin - self.r_margin
        self.set_xy(x, y - 2)
        self.line(x, y, x + w, y)
        self.set_xy(x, y)
        self.cell(w, 4, self.s(self._title), border=0, ln=1, align="L")
        self.ln(2)

    def footer(self):
        self.set_y(-10)
        self.set_font("Times", "I", 8)
        left = self.s(f"Generated with STREAMLINE ({self._streamline_version})")
        right = self.s(f"Page {self.page_no()}/{{nb}}")
        self.set_x(self.l_margin)
        self.cell(0, 5, left, border=0, ln=0, align="L")
        self.set_x(self.l_margin)
        self.cell(self.w - self.l_margin - self.r_margin, 5, right, border=0, ln=0, align="R")

    # -----------------------------
    # Cover
    # -----------------------------
    def cover_banner_title(self, title: str):
        x = self.l_margin
        y = self.t_margin
        w = self.w - self.l_margin - self.r_margin
        h = 12

        self.set_font("Times", "B", 14)
        self.set_xy(x, y)
        self.rect(x, y, w, h)
        self.set_xy(x + 2, y + 3.2)
        self.cell(w - 4, 6, self.s(title), border=0, ln=1, align="L")
        self.ln(3)

    def _cover_section_box(
        self,
        title: str,
        items: Sequence[Tuple[str, str]],
        *,
        x: float,
        y: float,
        w: float,
        max_items: Optional[int] = None,
        title_h: float = 6.0,
        row_h: float = 4.6,
        font_size: int = 9,
    ) -> float:
        if max_items is not None:
            items = items[:max_items]

        content_h = max(1, len(items)) * row_h
        h = title_h + content_h + 1.4

        if y + h > (self.h - self.b_margin - 2):
            self.add_page()
            y = self.get_y()

        self.rect(x, y, w, h)
        self.rect(x, y, w, title_h)

        self.set_font("Times", "B", 11)
        self.set_xy(x + 1.6, y + 1.6)
        self.cell(w - 3.2, title_h - 3.2, self.s(title), border=0, ln=0, align="L")

        self.set_font("Times", "", font_size)
        cy = y + title_h + 0.6
        for k, v in items:
            line = self.s(f"{k}: {v}")
            self.set_xy(x + 1.8, cy)
            self.multi_cell(w - 3.6, row_h, line, border=0)
            cy += row_h

        return h

    def cover_two_column_boxes(self, boxes: Dict[str, List[Tuple[str, str]]]):
        page_w = self.w - self.l_margin - self.r_margin
        gap = 4.0
        col_w = (page_w - gap) / 2.0

        xL = self.l_margin
        xR = self.l_margin + col_w + gap

        left_order = [
            "General Pipeline Settings:",
            "Feature Importance/Selection Settings:",
            "ML Modeling Algorithms:",
            "Modeling Settings:",
            "LCS Settings (eLCS, XCS, ExSTraCS):",
            "Stats and Figure Settings:",
        ]
        right_order = [
            "EDA and Processing Settings:",
            "Target Dataset(s):",
            "Target Data Settings:",
        ]

        y_start = self.get_y()
        yL = y_start
        yR = y_start

        for title in left_order:
            items = boxes.get(title) or []
            if not items:
                continue
            h = self._cover_section_box(title, items, x=xL, y=yL, w=col_w, title_h=6.0, row_h=4.6, font_size=9)
            yL += h + 2.0

        for title in right_order:
            items = boxes.get(title) or []
            if not items:
                continue
            font_size = 8 if title in {"EDA and Processing Settings:", "Target Data Settings:"} else 9
            row_h = 4.4 if font_size == 8 else 4.6
            h = self._cover_section_box(title, items, x=xR, y=yR, w=col_w, title_h=6.0, row_h=row_h, font_size=font_size)
            yR += h + 2.0

        self.set_y(max(yL, yR) + 1.0)

    # -----------------------------
    # Section typography
    # -----------------------------
    def panel_title(self, title: str, *, h: float = 6.0):
        w = self.w - self.l_margin - self.r_margin
        if self.get_y() + h + 2 > self.page_break_trigger:
            self.add_page()

        x = self.l_margin
        y = self.get_y()
        self.set_font("Times", "B", 10)
        self.rect(x, y, w, h)
        self.set_xy(x + 1.4, y + 1.4)
        self.cell(w - 2.8, h - 2.8, self.s(title), border=0, ln=1, align="L")
        self.ln(1.2)

    def subheader(self, text: str):
        self.set_font("Times", "B", 10)
        self.multi_cell(0, 5, self.s(text))
        self.ln(0.8)

    def cleaning_engineering_box(self, lines: Sequence[str]):
        w = self.w - self.l_margin - self.r_margin
        x = self.l_margin
        y = self.get_y()

        line_h = 5.0
        pad = 2.0
        h = pad + len(lines) * line_h + pad

        if y + h > self.page_break_trigger:
            self.add_page()
            x = self.l_margin
            y = self.get_y()

        self.rect(x, y, w, h)
        self.set_font("Times", "", 10)
        cy = y + pad
        for ln in lines:
            self.set_xy(x + pad, cy)
            self.multi_cell(w - 2 * pad, line_h, self.s(ln), border=0)
            cy += line_h
        self.ln(2.0)

    # -----------------------------
    # Formatting helpers
    # -----------------------------
    def _cell_str(self, v: Any) -> str:
        return self.s(format_number(v, decimals=self._decimals))

    def _truncate_to_width(self, s: str, w_mm: float) -> str:
        if s is None:
            return ""
        s = self.s(s)
        if self.get_string_width(s) <= w_mm:
            return s
        if w_mm <= 0:
            return ""
        lo, hi = 0, len(s)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = s[:mid]
            if self.get_string_width(cand) <= w_mm:
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    # -----------------------------
    # Tables
    # -----------------------------
    def _col_widths_model_perf(self, columns: Sequence[str], table_w: float) -> List[float]:
        """
        Widths for wide performance tables with:
          columns = ["Algorithm", metric1, metric2, ...]
        No wrapping: we rely on truncation to fit each cell.
        """
        n = len(columns)
        if n <= 1:
            return [table_w]

        alg_w = min(max(36.0, 0.20 * table_w), 52.0)
        rest = max(0.0, table_w - alg_w)

        metric_cols = list(columns[1:])
        weights: List[float] = []
        for c in metric_cols:
            cl = str(c).lower()
            w = 1.0 + min(1.6, len(str(c)) / 16.0)
            if any(x in cl for x in ["sensitivity", "precision", "specificity"]):
                w *= 1.25
            if "balanced" in cl:
                w *= 1.10
            if "roc" in cl or "prc" in cl:
                w *= 1.10
            weights.append(w)

        sw = sum(weights) if sum(weights) > 0 else float(max(1, len(weights)))
        raw = [rest * (w / sw) for w in weights]

        min_metric = 18.0
        raw = [max(min_metric, r) for r in raw]

        s2 = sum(raw)
        scale = (rest / s2) if s2 > 0 else 1.0
        metrics = [r * scale for r in raw]

        widths = [alg_w] + metrics
        widths[-1] += (table_w - sum(widths))
        return widths

    def draw_mean_std_table(self, mean_std: Dict[str, Any], *, no_wrap: bool = True):
        cols = mean_std.get("columns", [])
        rows_out: List[List[Any]] = []

        for row in mean_std.get("rows", []) or []:
            out_row: List[Any] = []
            for cell in row.get("cells", []):
                val = cell.get("value")
                if isinstance(val, tuple) and len(val) == 2:
                    mv, sv = val
                    out_row.append(self.s(f"{format_number(mv, decimals=self._decimals)} +/- {format_number(sv, decimals=self._decimals)}"))
                else:
                    out_row.append(self.s(val))
            rows_out.append(out_row)

        table_w = self.w - self.l_margin - self.r_margin
        col_widths = self._col_widths_model_perf(cols, table_w)

        ncol = len(cols)
        font_size = 6 if ncol >= 10 else 7

        self.draw_table(cols, rows_out, col_widths=col_widths, font_size=font_size, no_wrap=no_wrap)

    def _auto_col_widths(self, columns: Sequence[str], rows: Sequence[Sequence[str]], table_w: float) -> List[float]:
        n = len(columns)
        if n == 1:
            return [table_w]

        weights: List[float] = []
        for j, c in enumerate(columns):
            w = max(3.0, float(len(self.s(c))))
            for r in rows[:15]:
                if j < len(r):
                    w = max(w, min(44.0, float(len(self.s(r[j])))))
            if j == 0:
                w *= 1.6
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
        no_wrap: bool = False,
    ):
        if not columns:
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
                txt = self._truncate_to_width(str(col), wj - 2 * self._tbl_pad_x) if no_wrap else self.s(col)
                self.cell(wj - 2 * self._tbl_pad_x, header_h - 2 * self._tbl_pad_y, txt, border=0, ln=0, align="C")
                x0 += wj
            self.set_xy(self.l_margin, y0 + header_h)
            self.set_font("Times", "", font_size)

        if self.get_y() + header_h + 2 > self.page_break_trigger:
            self.add_page()
        draw_header()

        row_h = max(4.4, line_h + 2 * self._tbl_pad_y)
        for cells in formatted_rows:
            if self.get_y() + row_h > self.page_break_trigger:
                self.add_page()
                draw_header()

            y0 = self.get_y()
            x0 = self.l_margin
            for j, txt in enumerate(cells):
                wj = col_widths[j]
                self.rect(x0, y0, wj, row_h)
                self.set_xy(x0 + self._tbl_pad_x, y0 + self._tbl_pad_y)
                t = self._truncate_to_width(txt, wj - 2 * self._tbl_pad_x) if no_wrap else self.s(txt)
                self.cell(wj - 2 * self._tbl_pad_x, line_h, t, border=0, ln=0, align="L")
                x0 += wj
            self.set_y(y0 + row_h)

        self.ln(self._gap_after_table)

    # -----------------------------
    # Figures (skip missing entirely; keep aspect if supported)
    # -----------------------------
    def _image_panel(
        self,
        title: str,
        path: Optional[str],
        *,
        x: float,
        y: float,
        w: float,
        h: float,
        title_h: float,
        keep_aspect: bool = True,
    ) -> bool:
        if not path:
            return False
        p = Path(path)
        if not p.exists():
            return False

        self.rect(x, y, w, h)
        if title_h > 0:
            self.rect(x, y, w, title_h)
            self.set_font("Times", "B", 8)
            self.set_xy(x + self._panel_title_text_pad_x, y + self._panel_title_text_pad_y)
            self.cell(
                w - 2 * self._panel_title_text_pad_x,
                title_h - 2 * self._panel_title_text_pad_y,
                self.s(title),
                border=0,
                ln=0,
                align="L",
            )

        inner_x = x + self._panel_pad
        inner_y = y + title_h + (self._panel_content_pad_top if title_h > 0 else self._panel_pad)
        inner_w = w - 2 * self._panel_pad
        inner_h = h - title_h - ((self._panel_content_pad_top if title_h > 0 else 0) + self._panel_pad)

        try:
            if keep_aspect:
                try:
                    self.image(str(p), x=inner_x, y=inner_y, w=inner_w, h=inner_h, keep_aspect_ratio=True)  # type: ignore
                except TypeError:
                    self.image(str(p), x=inner_x, y=inner_y, w=inner_w, h=inner_h)
            else:
                self.image(str(p), x=inner_x, y=inner_y, w=inner_w, h=inner_h)
        except Exception:
            return False

        return True

    def figure_grid_2x2(
        self,
        titles: Sequence[str],
        paths: Sequence[Optional[str]],
        *,
        cell_h: float = 66.0,
        gap: float = 4.0,
        title_h: float = 6.0,
        keep_aspect: bool = True,
    ):
        page_w = self.w - self.l_margin - self.r_margin
        cell_w = (page_w - gap) / 2.0

        x0 = self.l_margin
        y0 = self.get_y()

        needed_h = cell_h * 2 + gap + 2
        if y0 + needed_h > self.page_break_trigger:
            self.add_page()
            y0 = self.get_y()

        self._image_panel(titles[0], paths[0] if len(paths) > 0 else None, x=x0, y=y0, w=cell_w, h=cell_h, title_h=title_h, keep_aspect=keep_aspect)
        self._image_panel(titles[1], paths[1] if len(paths) > 1 else None, x=x0 + cell_w + gap, y=y0, w=cell_w, h=cell_h, title_h=title_h, keep_aspect=keep_aspect)

        y1 = y0 + cell_h + gap
        self._image_panel(titles[2], paths[2] if len(paths) > 2 else None, x=x0, y=y1, w=cell_w, h=cell_h, title_h=title_h, keep_aspect=keep_aspect)
        self._image_panel(titles[3], paths[3] if len(paths) > 3 else None, x=x0 + cell_w + gap, y=y1, w=cell_w, h=cell_h, title_h=title_h, keep_aspect=keep_aspect)

        self.set_y(y1 + cell_h + 2)

    def figure_row_2(
        self,
        titles: Sequence[str],
        paths: Sequence[Optional[str]],
        *,
        h: float = 80.0,
        gap: float = 4.0,
        title_h: float = 6.0,
        keep_aspect: bool = True,
    ):
        page_w = self.w - self.l_margin - self.r_margin
        cell_w = (page_w - gap) / 2.0
        y0 = self.get_y()
        if y0 + h + 2 > self.page_break_trigger:
            self.add_page()
            y0 = self.get_y()

        x0 = self.l_margin
        self._image_panel(titles[0], paths[0] if len(paths) > 0 else None, x=x0, y=y0, w=cell_w, h=h, title_h=title_h, keep_aspect=keep_aspect)
        self._image_panel(titles[1], paths[1] if len(paths) > 1 else None, x=x0 + cell_w + gap, y=y0, w=cell_w, h=h, title_h=title_h, keep_aspect=keep_aspect)
        self.set_y(y0 + h + 2)

    def figure_single(
        self,
        title: str,
        path: Optional[str],
        *,
        h: float = 90.0,
        title_h: float = 6.0,
        keep_aspect: bool = True,
    ):
        if not path:
            return
        p = Path(path)
        if not p.exists():
            return

        page_w = self.w - self.l_margin - self.r_margin
        y0 = self.get_y()
        if y0 + h + 2 > self.page_break_trigger:
            self.add_page()
            y0 = self.get_y()

        self._image_panel(title, str(p), x=self.l_margin, y=y0, w=page_w, h=h, title_h=title_h, keep_aspect=keep_aspect)
        self.set_y(y0 + h + 2)
