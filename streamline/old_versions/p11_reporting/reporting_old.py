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

    Layout changes (per your request):
      - Cover page looks like the legacy report (two-column parameter boxes).
      - Metadata is also written as a plain text file (reporting/metadata.txt).
      - Per-dataset order:
          1) EDA page (Page 1 for the dataset): class balance, missingness, (optional) univariate
          2) Feature learning page: feature importance/selection (all methods) + informative features table
          3) Performance page: combined model + ensemble performance tables (no wrapping; widths + truncation)
          4) Evaluation results page: ROC/PRC summary with original aspect ratio + (optional) per-model curves
      - Univariate appears in EDA; automatically omitted if not informative.

    Data rules:
      - Prefer precomputed PNGs already present in experiment outputs.
      - Fallback to plotting/exporting into reporting/figures only if originals are missing.
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

        # Match legacy cover title style
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
        # e.g., 2023-09-26 12:57:00.921630 or ISO-ish fragments in keys
        if re.search(r"\b20\d{2}[-_/]\d{2}[-_/]\d{2}\b", s):
            return True
        if re.search(r"\b\d{2}:\d{2}:\d{2}\b", s):
            return True
        return False

    def _pretty_cover_key(self, k: str) -> str:
        """
        Remove identifier-like prefixes and flatten separators for the cover page.

        - Drops leading "run_params ·" / "metadata ·" segments by taking last segment.
        - Replaces underscores with spaces.
        - Strips common "identifier_code"/"identifier" style keys.
        """
        parts = [p.strip() for p in str(k).split("·")]
        last = parts[-1] if parts else str(k)
        last = last.strip()
        last = last.replace("_", " ")
        last = re.sub(r"\s+", " ", last).strip()
        return last

    def _should_drop_identifier_code(self, k: str) -> bool:
        s = k.lower()
        # user: "remove the identifier code (code like variable name)"
        # heuristics: keys that look like internal IDs / hashes / uid fields
        if any(x in s for x in ["identifier code", "identifier_code", "run id", "run_id", "uuid", "guid", "hash", "job id", "job_id"]):
            return True
        # If key itself looks like a variable name but value looks like a hash/uid
        return False

    def _write_metadata_text(self, *, cover_boxes: Dict[str, List[Tuple[str, str]]], enriched_meta: Dict[str, Any]) -> None:
        """
        Write a plain text metadata file for easy grepping/logging.
        """
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
        """
        Return ALL existing TopAverageScores.png figures under feature_importance/*.
        """
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

    def _figure_paths_model_curves(self, ds_dir: Path) -> Dict[str, List[str]]:
        """
        Collect per-model ROC/PRC curves if present, e.g. LR_ROC.png, LR_PRC.png.
        """
        out: Dict[str, List[str]] = {"roc": [], "prc": []}
        me = ds_dir / "model_evaluation"
        if not me.is_dir():
            return out

        for p in sorted(me.glob("*_ROC.png")):
            if p.name.lower().startswith("summary_"):
                continue
            out["roc"].append(str(p))

        for p in sorted(me.glob("*_PRC.png")):
            if p.name.lower().startswith("summary_"):
                continue
            out["prc"].append(str(p))

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

    def _univariate_is_informative(self, uni: Optional[pd.DataFrame]) -> bool:
        """
        Drop univariate if it doesn't add much:
          - missing/empty -> False
          - <3 rows -> False
          - if a p-value column exists, require at least one p < 0.10
        """
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
            # If no p-values, still consider it useful if it has at least 2+ columns and non-trivial rows
            return uni.shape[1] >= 2 and len(uni) >= 5

        try:
            pv = pd.to_numeric(uni[pcol], errors="coerce").dropna()
            if pv.empty:
                return False
            return bool((pv < 0.10).any())
        except Exception:
            return False

    # ============================================================
    # Cover page grouping (match legacy report categories)
    # ============================================================

    def _categorize_cover_items(
        self,
        meta_pickle_flat: Dict[str, Any],
        run_params_flat: Dict[str, Any],
        *,
        exp_root: Path,
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Create the cover page boxes in the same spirit as the legacy report,
        while removing duplicate timestamped params and identifier-code-ish fields.
        """
        combined = self._merge_union_dicts(run_params_flat or {}, meta_pickle_flat or {})

        def add(cat: str, k: str, v: Any):
            # Clean and filter keys
            if self._is_timestampy_key(k):
                return
            if self._should_drop_identifier_code(k):
                return

            pretty_k = self._pretty_cover_key(k)

            # If key is still ugly variable-ish, keep it but at least space it.
            if pretty_k.strip() == "":
                return

            out.setdefault(cat, []).append((pretty_k, self._as_compact_str(v)))

        out: Dict[str, List[Tuple[str, str]]] = {}

        # --- Target Dataset(s): infer from folder structure if not present
        ds_names = [p.name for p in self._list_datasets()]
        if ds_names:
            items = [(f"D{i+1}", nm) for i, nm in enumerate(ds_names)]
            out["Target Dataset(s):"] = [(f"{k}", f"= {v}") for k, v in items]

        # --- Target Data Settings (paths/labels)
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

        # --- General Pipeline Settings
        for kk, vv in combined.items():
            k = str(kk).lower()
            if any(x in k for x in ["cv", "partition", "seed", "random", "categorical cutoff", "statistical", "significance", "notebook"]):
                if any(x in k for x in ["plot", "export", "roc", "prc", "figure", "boxplot"]):
                    continue
                add("General Pipeline Settings:", kk, vv)

        # --- EDA and Processing Settings
        for kk, vv in combined.items():
            k = str(kk).lower()
            if any(x in k for x in ["missing", "imput", "scale", "correlation", "describe", "univariate", "eda", "processing", "clean"]):
                if any(x in k for x in ["feature importance", "feature_selection", "feature selection", "multisurf", "mutual", "turf"]):
                    continue
                add("EDA and Processing Settings:", kk, vv)

        # --- Feature Importance/Selection Settings
        for kk, vv in combined.items():
            k = str(kk).lower()
            if any(x in k for x in ["feature importance", "feature selection", "multisurf", "mutual", "turf", "max features", "top features", "filter poor"]):
                add("Feature Importance/Selection Settings:", kk, vv)

        # --- ML Modeling Algorithms (boolean toggles)
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

        # --- Modeling Settings
        for kk, vv in combined.items():
            k = str(kk).lower()
            if any(x in k for x in ["primary metric", "hyperparameter", "trials", "timeout", "subsample", "uniform feature importance"]):
                add("Modeling Settings:", kk, vv)

        # --- LCS Settings
        for kk, vv in combined.items():
            k = str(kk).lower()
            if any(x in k for x in ["lcs", "xcs", "elcs", "exstracs", "rule population", "training iterations", "nu"]):
                add("LCS Settings (eLCS, XCS, ExSTraCS):", kk, vv)

        # --- Stats and Figure Settings
        for kk, vv in combined.items():
            k = str(kk).lower()
            if any(x in k for x in ["export", "roc", "prc", "boxplot", "figure", "plot", "correlation", "top model features", "metric weighting"]):
                add("Stats and Figure Settings:", kk, vv)

        # Deduplicate by key within sections (keep first)
        for cat, items in list(out.items()):
            seen = set()
            dedup: List[Tuple[str, str]] = []
            for k, v in items:
                if k in seen:
                    continue
                seen.add(k)
                dedup.append((k, v))
            out[cat] = dedup

        # Ensure legacy order
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
        data_process_summary = self._read_csv_if_exists(explore / "DataProcessSummary.csv")
        class_counts = self._read_csv_if_exists(explore / "ClassCounts.csv")
        missingness = self._read_csv_if_exists(explore / "DataMissingness.csv")
        univariate = self._read_csv_if_exists(explore / "univariate_analyses" / "Univariate_Significance.csv")

        # Univariate: keep only if informative; show Top 10
        uni_use = self._univariate_is_informative(univariate)
        univariate_top10 = univariate.head(10) if (uni_use and univariate is not None and not univariate.empty) else None

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

        # Figures: EDA (prefer originals)
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

        # Performance: prefer summary/boxplots
        figs["models_mean_bar"] = ""
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
            if box and not figs["models_mean_bar"]:
                figs["models_mean_bar"] = box
            if not figs["models_mean_bar"]:
                try:
                    fig = self._plot_model_summary_bars(summary_mean, chosen_metric, f"{name}: Mean {chosen_metric} (Models)")
                    out = self.paths.figures_dir / f"{name}_models_mean_{chosen_metric.replace(' ', '_')}.png"
                    if _safe_plotly_to_png(fig, out):
                        figs["models_mean_bar"] = str(out)
                except Exception as e:
                    logger.warning("Model mean bar plot failed for %s (%s): %r", name, chosen_metric, e)

        # Evaluation results: ROC/PRC summaries (and per-model curves)
        figs["models_roc_overlay"] = self._figure_path_model_summary_roc_prc(ds_dir, "roc") or ""
        figs["models_prc_overlay"] = self._figure_path_model_summary_roc_prc(ds_dir, "prc") or ""

        figs["ensembles_roc"] = self._figure_path_ensemble_summary(ds_dir, "roc") or ""
        figs["ensembles_prc"] = self._figure_path_ensemble_summary(ds_dir, "prc") or ""

        figs["model_curves"] = self._figure_paths_model_curves(ds_dir)  # {"roc":[...], "prc":[...]}

        # Feature learning: ALL FI methods
        fi_methods = self._figure_paths_fs_top_scores(ds_dir)
        figs["fi_top_scores"] = fi_methods

        # Tables
        models_mean_std = self._build_mean_std_table(
            summary_mean,
            summary_std,
            highlight_metric_candidates=["Balanced Accuracy", "ROC AUC", "PRC AUC", "Accuracy"],
        )
        models_median = self._build_plain_table(summary_median, max_rows=100)

        ensembles_mean_std = self._build_mean_std_table(
            ens_mean,
            ens_std,
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

        # COVER (legacy-like)
        pdf.add_page()
        pdf.cover_banner_title(str(report_data.get("title", "")))
        pdf.cover_two_column_boxes(report_data.get("cover_boxes", {}) or {})

        # DATASETS (per-dataset page order you requested)
        for ds in report_data.get("datasets", []) or []:
            ds_name = str(ds.get("dataset_name", ""))
            ds_dir = str(ds.get("dataset_dir", ""))

            figs = ds.get("figures", {}) or {}
            tables = ds.get("tables", {}) or {}
            perf = ds.get("perf", {}) or {}

            # -------------------------
            # EDA - PAGE 1 (per dataset)
            # -------------------------
            pdf.add_page()
            pdf.panel_title(f"Dataset: {ds_name}")
            pdf.set_font("Times", "", 10)
            if ds_dir:
                pdf.multi_cell(0, 5, ds_dir)
                pdf.ln(1.0)

            pdf.panel_title("EDA")
            pdf.figure_row_2(
                titles=["Class Balance", "Missingness (Top 25 Features)"],
                paths=[figs.get("class_balance") or None, figs.get("missingness") or None],
                h=82.0,
                gap=4.0,
                title_h=6.0,
            )

            # Univariate first (and drop if not useful; already filtered)
            uni = tables.get("univariate_top10", {}) or {}
            if uni.get("present"):
                pdf.panel_title("Univariate Analysis (Top 10)")
                pdf.draw_table(uni.get("columns", []), uni.get("rows", []), max_rows=10)

            # Optional: data process summary fits naturally under EDA
            dps = tables.get("data_process_summary", {}) or {}
            if dps.get("present"):
                pdf.panel_title("Data Processing Summary")
                pdf.draw_table(dps.get("columns", []), dps.get("rows", []), max_rows=50)

            # -------------------------
            # Feature Learning
            # -------------------------
            pdf.add_page()
            pdf.panel_title(f"Dataset: {ds_name}")
            pdf.panel_title("Feature Learning")

            fi_list = figs.get("fi_top_scores") or []
            norm: List[Dict[str, str]] = []
            for item in fi_list:
                if isinstance(item, dict) and "path" in item:
                    norm.append({"method": str(item.get("method") or "method"), "path": str(item["path"])})
                elif isinstance(item, str):
                    norm.append({"method": "method", "path": item})

            if not norm:
                pdf.muted("Missing: feature_importance/*/TopAverageScores.png")
            elif len(norm) == 1:
                pdf.figure_single(f"Top Scores ({norm[0]['method']})", norm[0]["path"], h=130.0, title_h=6.0)
            elif len(norm) == 2:
                pdf.figure_row_2(
                    titles=[f"Top Scores ({norm[0]['method']})", f"Top Scores ({norm[1]['method']})"],
                    paths=[norm[0]["path"], norm[1]["path"]],
                    h=95.0,
                    gap=4.0,
                    title_h=6.0,
                )
            else:
                # 2x2 pages for many methods
                for i in range(0, len(norm), 4):
                    chunk = norm[i: i + 4]
                    titles = [f"Top Scores ({c['method']})" for c in chunk]
                    paths = [c["path"] for c in chunk]
                    while len(titles) < 4:
                        titles.append("")
                        paths.append(None)
                    pdf.figure_grid_2x2(titles=titles, paths=paths, cell_h=66.0, gap=4.0, title_h=6.0)
                    if i + 4 < len(norm):
                        pdf.add_page()
                        pdf.panel_title(f"Dataset: {ds_name}")
                        pdf.panel_title("Feature Learning")

            # Informative feature summary (selection output)
            inf = tables.get("informative_feature_summary", {}) or {}
            if inf.get("present"):
                pdf.panel_title("Informative Feature Summary")
                pdf.draw_table(inf.get("columns", []), inf.get("rows", []), max_rows=200)
            else:
                pdf.muted("Missing: feature_selection/InformativeFeatureSummary.csv")

            # -------------------------
            # Performance (CV-based): combine model + ensemble tables
            # - no wrapping (truncate)
            # - column widths driven by tuned allocation
            # -------------------------
            pdf.add_page()
            pdf.panel_title(f"Dataset: {ds_name}")
            pdf.panel_title("Performance (Cross-Validation)")

            ms = perf.get("models_mean_std", {}) or {}
            es = perf.get("ensembles_mean_std", {}) or {}
            mm = perf.get("models_median", {}) or {}
            em = perf.get("ensembles_median", {}) or {}

            pdf.subheader("Model + Ensemble Performance (Mean ± SD)")
            if ms.get("present"):
                pdf.draw_mean_std_table(ms, no_wrap=True)
            else:
                pdf.muted("Missing: model_evaluation/Summary_performance_mean.csv and/or Summary_performance_std.csv")

            if es.get("present"):
                pdf.draw_mean_std_table(es, no_wrap=True)

            # Optional: median tables (often redundant). Keep only if present and not huge.
            if mm.get("present") or em.get("present"):
                pdf.subheader("Median (optional)")
                if mm.get("present"):
                    pdf.draw_table(mm.get("columns", []), mm.get("rows", []), max_rows=80, no_wrap=True)
                if em.get("present"):
                    pdf.draw_table(em.get("columns", []), em.get("rows", []), max_rows=80, no_wrap=True)

            # If a boxplot exists, show it here as the “distribution” companion
            if figs.get("models_cv_box") or figs.get("models_mean_bar"):
                pdf.panel_title("Performance Distribution / Comparison Plot")
                pdf.figure_single(
                    "Model Comparison / Boxplot",
                    figs.get("models_cv_box") or figs.get("models_mean_bar") or None,
                    h=110.0,
                    title_h=6.0,
                    keep_aspect=True,
                )

            # -------------------------
            # Evaluation results (post-CV): ROC/PRC summary + per-model curves
            # Use original aspect ratio where possible.
            # -------------------------
            pdf.add_page()
            pdf.panel_title(f"Dataset: {ds_name}")
            pdf.panel_title("Evaluation Results (Curves)")

            pdf.figure_row_2(
                titles=["Summary ROC", "Summary PRC"],
                paths=[figs.get("models_roc_overlay") or None, figs.get("models_prc_overlay") or None],
                h=85.0,
                gap=4.0,
                title_h=6.0,
                keep_aspect=True,
            )

            if figs.get("ensembles_roc") or figs.get("ensembles_prc"):
                pdf.panel_title("Ensembles (Summary Curves)")
                pdf.figure_row_2(
                    titles=["Ensembles ROC", "Ensembles PRC"],
                    paths=[figs.get("ensembles_roc") or None, figs.get("ensembles_prc") or None],
                    h=85.0,
                    gap=4.0,
                    title_h=6.0,
                    keep_aspect=True,
                )

            # Per-model ROC/PRC (show a compact selection)
            curves = figs.get("model_curves") or {}
            roc_list = list(curves.get("roc") or [])
            prc_list = list(curves.get("prc") or [])
            if roc_list or prc_list:
                # Show up to 4 ROC + 4 PRC over pages
                def take_chunks(xs: List[str], n: int) -> List[List[str]]:
                    return [xs[i:i + n] for i in range(0, len(xs), n)]

                roc_chunks = take_chunks(roc_list[:8], 4)
                prc_chunks = take_chunks(prc_list[:8], 4)

                # Interleave ROC then PRC pages
                for chunk in roc_chunks:
                    pdf.panel_title("Per-Model ROC Curves (sample)")
                    titles = [Path(p).stem for p in chunk]
                    paths = chunk
                    while len(titles) < 4:
                        titles.append("")
                        paths.append(None)
                    pdf.figure_grid_2x2(titles=titles, paths=paths, cell_h=66.0, gap=4.0, title_h=6.0, keep_aspect=True)
                    if chunk is not roc_chunks[-1] or prc_chunks:
                        pdf.add_page()
                        pdf.panel_title(f"Dataset: {ds_name}")
                        pdf.panel_title("Evaluation Results (Curves)")

                for chunk in prc_chunks:
                    pdf.panel_title("Per-Model PRC Curves (sample)")
                    titles = [Path(p).stem for p in chunk]
                    paths = chunk
                    while len(titles) < 4:
                        titles.append("")
                        paths.append(None)
                    pdf.figure_grid_2x2(titles=titles, paths=paths, cell_h=66.0, gap=4.0, title_h=6.0, keep_aspect=True)
                    if chunk is not prc_chunks[-1]:
                        pdf.add_page()
                        pdf.panel_title(f"Dataset: {ds_name}")
                        pdf.panel_title("Evaluation Results (Curves)")

            # -------------------------
            # Runtimes (still its own page)
            # -------------------------
            pdf.add_page()
            pdf.panel_title(f"Dataset: {ds_name}")
            pdf.panel_title("Runtime Summary")
            rt = tables.get("runtimes", {}) or {}
            if rt.get("present"):
                pdf.draw_table(rt.get("columns", []), rt.get("rows", []), max_rows=500, no_wrap=True)
            else:
                pdf.muted("Missing: runtimes.csv")

        # DATASET COMPARISONS (global)
        dc = report_data.get("dataset_comparisons", {}) or {}
        if dc.get("present"):
            pdf.add_page()
            pdf.panel_title("Dataset Comparisons")

            overview = (dc.get("figures", {}) or {}).get("overview")
            pdf.figure_single("Comparison Overview (All Datasets)", overview or None, h=120.0, title_h=6.0, keep_aspect=True)

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

        # Pickles for cover metadata/params
        meta_pickle_obj = self._read_pickle_if_exists(self.exp_root / "metadata.pickle")
        run_params_obj = self._read_pickle_if_exists(self.exp_root / "run_params.pickle")

        meta_pickle_dict: Dict[str, Any] = meta_pickle_obj if isinstance(meta_pickle_obj, dict) else ({"metadata.pickle": str(meta_pickle_obj)} if meta_pickle_obj is not None else {})
        run_params_dict: Dict[str, Any] = run_params_obj if isinstance(run_params_obj, dict) else ({"run_params.pickle": str(run_params_obj)} if run_params_obj is not None else {})

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

        # JSON for downstream debugging (kept)
        self.paths.data_json.write_text(json.dumps(report_data, indent=2))

        # Metadata as text file (requested)
        self._write_metadata_text(cover_boxes=cover_boxes, enriched_meta=enriched_meta)

        if self.make_pdf:
            self._write_pdf(report_data)

        jc = self.exp_root / "jobsCompleted"
        jc.mkdir(exist_ok=True)
        (jc / "job_reporting.txt").write_text("complete")

        self.save_runtime()
        logger.info("Phase 11 reporting complete: %s", self.paths.pdf)


# ============================================================
# PDF renderer (legacy-like cover + boxed sections)
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

    # -----------------------------
    # Legacy-like cover
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
        self.cell(w - 4, 6, title, border=0, ln=1, align="L")
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
        self.cell(w - 3.2, title_h - 3.2, title, border=0, ln=0, align="L")

        self.set_font("Times", "", font_size)
        cy = y + title_h + 0.6
        for k, v in items:
            line = f"{k}: {v}"
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
        self.cell(w - 2.8, h - 2.8, title, border=0, ln=1, align="L")
        self.ln(1.2)

    def subheader(self, text: str):
        self.set_font("Times", "B", 10)
        self.multi_cell(0, 5, text)
        self.ln(0.8)

    def muted(self, text: str):
        self.set_font("Times", "", 9)
        self.multi_cell(0, 4.5, text)
        self.ln(0.8)

    def _cell_str(self, v: Any) -> str:
        return format_number(v, decimals=self._decimals)

    # -----------------------------
    # No-wrap table rendering (truncate to fit)
    # -----------------------------
    def _truncate_to_width(self, s: str, w_mm: float) -> str:
        """
        Truncate string to fit inside width (mm) using current font metrics.
        """
        if s is None:
            return ""
        s = str(s)
        if self.get_string_width(s) <= w_mm:
            return s
        if w_mm <= self.get_string_width("..."):
            return ""
        # binary-ish shrink
        ell = "..."
        lo, hi = 0, len(s)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = s[:mid] + ell
            if self.get_string_width(cand) <= w_mm:
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1
        return best or ell

    # -----------------------------
    # Tables
    # -----------------------------
    def _col_widths_model_perf(self, columns: Sequence[str], table_w: float) -> List[float]:
        """
        Performance tables: prioritize metric columns to reduce need for wrapping.
        """
        n = len(columns)
        if n <= 1:
            return [table_w]

        # shrink Algorithm column more than before
        first = min(max(32.0, 0.16 * table_w), 46.0)

        metric_cols = list(columns[1:])
        weights: List[float] = []
        for c in metric_cols:
            cl = c.lower()
            w = 1.0 + min(1.6, len(c) / 16.0)
            if "sensitivity" in cl or "precision" in cl or "specificity" in cl:
                w *= 1.25
            if "balanced" in cl:
                w *= 1.10
            if "roc" in cl or "prc" in cl:
                w *= 1.10
            weights.append(w)

        rest = max(0.0, table_w - first)
        sw = sum(weights) if sum(weights) > 0 else float(len(weights))
        raw = [rest * (w / sw) for w in weights]

        # enforce larger minimum metric width to avoid wraps
        min_metric = 18.0
        raw = [max(min_metric, r) for r in raw]

        s2 = sum(raw)
        scale = (rest / s2) if s2 > 0 else 1.0
        metrics = [r * scale for r in raw]

        widths = [first] + metrics
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
                    out_row.append(f"{self._cell_str(mv)} ± {self._cell_str(sv)}")
                else:
                    out_row.append(val)
            rows_out.append(out_row)

        table_w = self.w - self.l_margin - self.r_margin
        col_widths = self._col_widths_model_perf(cols, table_w)

        ncol = len(cols)
        font_size = 6 if ncol >= 9 else 7

        self.draw_table(cols, rows_out, col_widths=col_widths, font_size=font_size, no_wrap=no_wrap)

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
                txt = self._truncate_to_width(str(col), wj - 2 * self._tbl_pad_x) if no_wrap else str(col)
                self.cell(wj - 2 * self._tbl_pad_x, header_h - 2 * self._tbl_pad_y, txt, border=0, ln=0, align="C")
                x0 += wj
            self.set_xy(self.l_margin, y0 + header_h)
            self.set_font("Times", "", font_size)

        if self.get_y() + header_h + 2 > self.page_break_trigger:
            self.add_page()

        draw_header()

        if no_wrap:
            # Fixed row height (single line)
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
                    t = self._truncate_to_width(txt, wj - 2 * self._tbl_pad_x)
                    self.cell(wj - 2 * self._tbl_pad_x, line_h, t, border=0, ln=0, align="L")
                    x0 += wj
                self.set_y(y0 + row_h)

            self.ln(self._gap_after_table)
            return

        # Wrapping version (kept for non-performance tables)
        def split_lines(txt: str, width_mm: float) -> List[str]:
            s = "" if txt is None else str(txt)
            usable_w = max(1e-6, float(width_mm))
            try:
                lines = self.multi_cell(usable_w, line_h, s, border=0, align="L", split_only=True)
                return [ln if ln is not None else "" for ln in lines] or [""]
            except TypeError:
                parts = s.splitlines() or [s]
                out: List[str] = []
                for part in parts:
                    sw = self.get_string_width(part) if part else 0.0
                    n = max(1, int(math.ceil(sw / usable_w)))
                    out.extend([""] * n)
                return out or [""]

        def row_height(cells: Sequence[str]) -> float:
            counts: List[int] = []
            for j, txt in enumerate(cells):
                usable_w = col_widths[j] - 2 * self._tbl_pad_x
                counts.append(len(split_lines(txt, usable_w)))
            return max(counts) * line_h + 2 * self._tbl_pad_y

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
                self.multi_cell(wj - 2 * self._tbl_pad_x, line_h, txt, border=0, align="L")
                x0 += wj
                self.set_xy(x0, y0)

            self.set_y(y0 + rh)

        self.ln(self._gap_after_table)

    # -----------------------------
    # Figures (aspect ratio control)
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
        keep_aspect: bool = False,
    ):
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
                # Prefer keeping aspect ratio (fpdf2 supports keep_aspect_ratio in image()).
                if keep_aspect:
                    try:
                        self.image(path, x=inner_x, y=inner_y, w=inner_w, h=inner_h, keep_aspect_ratio=True)  # type: ignore
                    except TypeError:
                        # fallback: let FPDF decide; still uses both w/h
                        self.image(path, x=inner_x, y=inner_y, w=inner_w, h=inner_h)
                else:
                    self.image(path, x=inner_x, y=inner_y, w=inner_w, h=inner_h)
            except Exception:
                self.set_font("Times", "", 8)
                self.set_xy(inner_x, inner_y)
                self.multi_cell(inner_w, 3.5, "Unable to render figure.", border=0)
        else:
            self.set_font("Times", "", 8)
            self.set_xy(inner_x, inner_y)
            self.multi_cell(inner_w, 3.5, "Figure not found.", border=0)

    def figure_grid_2x2(
        self,
        titles: Sequence[str],
        paths: Sequence[Optional[str]],
        *,
        cell_h: float = 66.0,
        gap: float = 4.0,
        title_h: float = 6.0,
        keep_aspect: bool = False,
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
        keep_aspect: bool = False,
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

    def figure_single(self, title: str, path: Optional[str], *, h: float = 90.0, title_h: float = 6.0, keep_aspect: bool = False):
        page_w = self.w - self.l_margin - self.r_margin
        y0 = self.get_y()
        if y0 + h + 2 > self.page_break_trigger:
            self.add_page()
            y0 = self.get_y()
        self._image_panel(title, path, x=self.l_margin, y=y0, w=page_w, h=h, title_h=title_h, keep_aspect=keep_aspect)
        self.set_y(y0 + h + 2)
