from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import logging
import math
import os
import pickle
import re
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from streamline.p6_modeling.utils.categorical import NATIVE_CATEGORICAL_MODELS_DEFAULT
from streamline.utils.run_commands import RUN_COMMANDS_FILENAME, load_run_commands

logger = logging.getLogger(__name__)


try:
    from fpdf import FPDF  # type: ignore
except Exception:  # pragma: no cover
    FPDF = None  # type: ignore


PHASE_LABELS = {
    "1": "EDA / Exploratory Analysis",
    "2": "Scale and Impute",
    "3": "Feature Learning",
    "4": "Feature Selection",
    "5": "Modeling",
    "8": "Stats Summary",
}

RUN_COMMAND_PHASE_ORDER = [
    "p1_data_process",
    "p2_impute_scale",
    "p3_feature_learning",
    "p4_feature_importance",
    "p5_feature_selection",
    "p6_modeling",
    "p7_ensembles",
    "p8_summary_statistics",
    "p9_compare_datasets",
    "p10_replication",
    "p11_reporting",
]

RUN_COMMAND_PHASE_LABELS = {
    "p1_data_process": "P1 Data Processing",
    "p2_impute_scale": "P2 Impute / Scale",
    "p3_feature_learning": "P3 Feature Learning",
    "p4_feature_importance": "P4 Feature Importance",
    "p5_feature_selection": "P5 Feature Selection",
    "p6_modeling": "P6 Modeling",
    "p7_ensembles": "P7 Ensembles",
    "p8_summary_statistics": "P8 Summary Statistics",
    "p9_compare_datasets": "P9 Dataset Comparison",
    "p10_replication": "P10 Replication",
    "p11_reporting": "P11 Reporting",
}

LEGACY_NOT_RECORDED = "Not recorded in legacy output"
NOT_RUN = "Not run"
NO_VALUE = object()

ENSEMBLE_SMALL_NAME_TO_ID = {
    "HEV": "hard_voting",
    "SEV": "soft_voting",
    "STK_LR": "stack_lr",
    "STK_DT": "stack_dt",
    "STK_RF": "stack_rf",
}

CLASSIFICATION_METRICS = [
    "Balanced Accuracy",
    "Accuracy",
    "F1 Score",
    "Sensitivity (Recall)",
    "Precision (PPV)",
    "Brier Score",
    "ROC AUC",
    "PRC AUC",
    "PRC APS",
]

REGRESSION_METRICS = [
    "Explained Variance",
    "Pearson Correlation",
    "Mean Absolute Error",
    "Mean Squared Error",
    "Median Absolute Error",
    "Max Error",
]

REGRESSION_DEFAULT_METRIC = "explained_variance"

CLASSIFICATION_ONLY_METRIC_KEYS = {
    "balanced_accuracy",
    "accuracy",
    "f1",
    "f1_macro",
    "recall",
    "recall_macro",
    "precision",
    "precision_macro",
    "roc_auc",
    "roc_auc_macro",
    "average_precision",
    "average_precision_macro",
    "prc_auc",
    "prc_aps",
}

METRIC_DIRECTION_HIGHER_IS_BETTER = {
    "Balanced Accuracy": True,
    "Accuracy": True,
    "F1 Score": True,
    "Sensitivity (Recall)": True,
    "Precision (PPV)": True,
    "ROC AUC": True,
    "PRC AUC": True,
    "PRC APS": True,
    "Brier Score": False,
    "Explained Variance": True,
    "Pearson Correlation": True,
    "Mean Absolute Error": False,
    "Mean Squared Error": False,
    "Median Absolute Error": False,
    "Max Error": False,
}

METRIC_JSON_KEYS = {
    "Balanced Accuracy": "balanced_accuracy",
    "Accuracy": "accuracy",
    "F1 Score": "f1",
    "Sensitivity (Recall)": "recall_macro",
    "Precision (PPV)": "precision_macro",
    "Brier Score": "brier_score",
    "ROC AUC": "roc_auc_macro",
    "PRC AUC": "average_precision_macro",
    "PRC APS": "average_precision_macro",
    "Explained Variance": "explained_variance",
    "Pearson Correlation": "pearson_correlation",
    "Mean Absolute Error": "mean_absolute_error",
    "Mean Squared Error": "mean_squared_error",
    "Median Absolute Error": "median_absolute_error",
    "Max Error": "max_error",
}

FEATURE_LEARNING_METHODS: List[Dict[str, Any]] = [
    {
        "key": "mutual_info",
        "label": "Mutual Information",
        "dir_aliases": ["mutualinformation", "mutual_information"],
        "score_patterns": ["mutualinformation_scores_cv_*.csv", "mutual_information_scores_cv_*.csv"],
    },
    {
        "key": "multisurf",
        "label": "MultiSURF",
        "dir_aliases": ["multisurf", "multi_surf"],
        "score_patterns": ["multisurf_scores_cv_*.csv", "multi_surf_scores_cv_*.csv"],
    },
    {
        "key": "multisurfstar",
        "label": "MultiSURFstar",
        "dir_aliases": ["multisurfstar", "multisurf_star", "multi_surfstar", "multi_surf_star"],
        "score_patterns": ["multisurfstar_scores_cv_*.csv", "multisurf_star_scores_cv_*.csv"],
    },
]


def _now_iso_local() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _try_streamline_version() -> str:
    try:
        return importlib.metadata.version("streamline")
    except Exception:
        return "unknown"


def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.is_file():
            return p
    return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None


def _format_number(value: Any, *, is_pvalue: bool = False) -> str:
    f = _safe_float(value)
    if f is None:
        return str(value) if value is not None else ""
    if is_pvalue and abs(f) < 0.001 and f != 0:
        return f"{f:.2e}"
    rounded = round(f, 3)
    if abs(rounded) < 0.0005:
        return "0"
    if abs(rounded - round(rounded)) < 1e-12:
        return str(int(round(rounded)))
    text = f"{rounded:.3f}".rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    return text


def _is_pvalue_col(name: str) -> bool:
    n = name.strip().lower()
    return n in {"p", "p-value", "p_value", "pvalue"} or "p-value" in n


def _is_numeric_text(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _shorten(text: str, width: int = 46) -> str:
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."



def _linspace(start: float, end: float, num: int) -> List[float]:
    if num <= 1:
        return [start]
    step = (end - start) / float(num - 1)
    return [start + i * step for i in range(num)]


def _auc_trapezoid(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    total = 0.0
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        total += dx * (y[i] + y[i - 1]) * 0.5
    return total


def _interp_sorted(x: Sequence[float], y: Sequence[float], x_new: Sequence[float]) -> List[float]:
    if not x or not y or len(x) != len(y):
        return [0.0 for _ in x_new]

    pairs = sorted(zip(x, y), key=lambda t: t[0])
    xs = [pairs[0][0]]
    ys = [pairs[0][1]]
    for px, py in pairs[1:]:
        if px == xs[-1]:
            ys[-1] = py
        else:
            xs.append(px)
            ys.append(py)

    out: List[float] = []
    j = 0
    n = len(xs)
    for xv in x_new:
        if xv <= xs[0]:
            out.append(ys[0])
            continue
        if xv >= xs[-1]:
            out.append(ys[-1])
            continue
        while j + 1 < n and xs[j + 1] < xv:
            j += 1
        x0, x1 = xs[j], xs[j + 1]
        y0, y1 = ys[j], ys[j + 1]
        if x1 == x0:
            out.append(y1)
        else:
            t = (xv - x0) / (x1 - x0)
            out.append(y0 + t * (y1 - y0))
    return out


@dataclass
class ReportPaths:
    reporting_dir: Path
    data_json: Path
    pdf: Path
    figures_dir: Path


@dataclass
class TableData:
    columns: List[str]
    rows: List[Dict[str, str]]


if FPDF is not None:

    class _StreamlinePDF(FPDF):  # type: ignore[misc]
        def __init__(self, footer_text: str):
            super().__init__(orientation="P", unit="mm", format="A4")
            self.footer_text = footer_text

        def footer(self):
            self.set_y(-10)
            self.set_font("Times", "I", 7)
            self.cell(0, 4, self.footer_text, border=0, ln=0, align="L")
            self.set_font("Times", "", 8)
            self.cell(0, 4, f"Page {self.page_no()}/{{nb}}", border=0, ln=0, align="R")

else:

    class _StreamlinePDF:  # pragma: no cover
        def __init__(self, *_args, **_kwargs):
            raise ImportError("fpdf2 is required for PDF rendering. Install `fpdf2`.")


class ReportPhaseJob:
    """
    STREAMLINE Testing Data Evaluation Report generator.

    This implementation covers binary/multiclass/regression outputs and
    follows the master reporting specification provided by the user.
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        experiment_name: Optional[str] = None,
        experiment_path: Optional[str] = None,
        reporting_dir: Optional[str] = None,
        report_mode: str = "standard",  # standard | replication
        outcome_label: Optional[str] = None,
        outcome_type: Optional[str] = None,
        instance_label: Optional[str] = None,
        make_pdf: bool = True,
        enable_plots: bool = True,
        reuse_existing_figures: bool = True,
    ):
        assert (output_path and experiment_name) or experiment_path, (
            "Provide (output_path, experiment_name) or experiment_path."
        )

        if experiment_path:
            self.exp_root = Path(experiment_path).resolve()
            self.output_path = str(self.exp_root.parent)
            self.experiment_name = self.exp_root.name
        else:
            self.output_path = str(output_path)
            self.experiment_name = str(experiment_name)
            self.exp_root = (Path(self.output_path) / self.experiment_name).resolve()

        if not self.exp_root.is_dir():
            raise FileNotFoundError(f"Experiment folder not found: {self.exp_root}")

        self.outcome_label = outcome_label
        self.outcome_type = outcome_type
        self.instance_label = instance_label
        self.report_mode = str(report_mode or "standard").strip().lower()
        if self.report_mode not in {"standard", "replication"}:
            raise ValueError("report_mode must be one of: standard, replication")
        self.make_pdf = make_pdf
        self.enable_plots = enable_plots
        self.reuse_existing_figures = reuse_existing_figures
        self.job_start_time: Optional[float] = None
        self.reporting_dir_override = Path(reporting_dir).resolve() if reporting_dir else None
        self.paths = self._init_paths()

        self._mpl_ready: Optional[bool] = None

    def _init_paths(self) -> ReportPaths:
        if self.reporting_dir_override:
            reporting_dir = self.reporting_dir_override
        else:
            if self.report_mode == "replication":
                reporting_dir = self.exp_root / "reporting_replication"
            else:
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

    def _read_pickle_if_exists(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            if path.is_file():
                blob = pickle.load(path.open("rb"))
                if isinstance(blob, dict):
                    return blob
        except Exception as exc:
            logger.warning("Failed reading pickle %s: %r", path, exc)
        return None

    def _latest_run_params(self, run_params: Dict[str, Any]) -> Dict[str, Any]:
        if not run_params:
            return {}
        try:
            key = sorted(run_params.keys())[-1]
            val = run_params.get(key)
            if isinstance(val, dict):
                return val
        except Exception:
            pass
        return {}

    def load_run_command_context(self) -> Dict[str, Any]:
        path = self.exp_root / RUN_COMMANDS_FILENAME
        store = load_run_commands(self.exp_root)
        phases_blob = store.get("phases", {}) if isinstance(store, dict) else {}
        phases = phases_blob if isinstance(phases_blob, dict) else {}

        records: Dict[str, Dict[str, Any]] = {}
        for phase, phase_store in phases.items():
            if not isinstance(phase_store, dict):
                continue
            latest = phase_store.get("latest", {})
            if not isinstance(latest, dict):
                continue
            args = latest.get("args") or {}
            safe_args = json.loads(json.dumps(args, default=str)) if isinstance(args, dict) else {}
            records[str(phase)] = {
                "phase": str(phase),
                "label": RUN_COMMAND_PHASE_LABELS.get(str(phase), str(phase)),
                "updated_at": latest.get("updated_at", ""),
                "command": latest.get("command", ""),
                "args": safe_args,
            }

        ordered_phases = [
            phase
            for phase in RUN_COMMAND_PHASE_ORDER
            if phase in records
        ] + sorted(
            phase
            for phase in records
            if phase not in set(RUN_COMMAND_PHASE_ORDER)
        )
        latest_phase = ""
        latest_updated_at = ""
        for phase in ordered_phases:
            updated = str(records[phase].get("updated_at") or "")
            if updated >= latest_updated_at:
                latest_phase = phase
                latest_updated_at = updated

        return {
            "present": path.is_file() and bool(records),
            "path": str(path),
            "recorded_phase_count": len(records),
            "recorded_phases": ordered_phases,
            "latest_phase": latest_phase,
            "latest_phase_label": RUN_COMMAND_PHASE_LABELS.get(latest_phase, latest_phase),
            "latest_updated_at": latest_updated_at,
            "records": records,
        }

    def phase_args_from_records(self, records: Dict[str, Any], phase: str) -> Dict[str, Any]:
        record = records.get(phase, {}) if isinstance(records, dict) else {}
        args = record.get("args", {}) if isinstance(record, dict) else {}
        return args if isinstance(args, dict) else {}

    def value_is_present(self, value: Any) -> bool:
        return value is not None and value != ""

    def first_present_value(self, *values: Any, default: Any = NO_VALUE) -> Any:
        for value in values:
            if self.value_is_present(value):
                return value
        if default is not NO_VALUE:
            return default
        return None

    def phase_summary_value(
        self,
        phase_ran: bool,
        *values: Any,
        default: Any = NO_VALUE,
    ) -> Any:
        if not phase_ran:
            return NOT_RUN
        value = self.first_present_value(*values)
        if self.value_is_present(value):
            return value
        if default is not NO_VALUE:
            return default
        return LEGACY_NOT_RECORDED

    def truthy_config_value(self, value: Any) -> Optional[bool]:
        if value in (NOT_RUN, LEGACY_NOT_RECORDED) or not self.value_is_present(value):
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return None

    def report_is_regression(self, metadata: Dict[str, Any], dataset_blocks: Sequence[Dict[str, Any]]) -> bool:
        outcome_type = str(metadata.get("Outcome Type") or "").strip().lower()
        if outcome_type in {"continuous", "regression", "numeric", "real", "float"}:
            return True
        return any(str(ds.get("task_type") or "").strip().lower() == "regression" for ds in dataset_blocks)

    def report_metric_for_outcome(self, value: Any, is_regression: bool, *, default: str) -> Any:
        if value in (NOT_RUN, LEGACY_NOT_RECORDED):
            return value
        if not self.value_is_present(value):
            return default
        if not is_regression:
            return value

        metric = str(value).strip()
        metric_key = metric.lower().replace(" ", "_").replace("-", "_")
        if metric_key in CLASSIFICATION_ONLY_METRIC_KEYS:
            return default
        return value

    def categorical_handling_summary(
        self,
        *,
        p1_ran: bool,
        p6_ran: bool,
        p1: Dict[str, Any],
        p6: Dict[str, Any],
        metadata_pickle: Dict[str, Any],
    ) -> Any:
        one_hot = self.phase_summary_value(
            p1_ran,
            p1.get("one_hot_encoding"),
            metadata_pickle.get("One Hot Encoding"),
            default=True,
        )
        bypass = self.phase_summary_value(
            p6_ran,
            p6.get("bypass_one_hot_for_native_models"),
            default=True,
        )
        native_models = self.phase_summary_value(
            p6_ran,
            p6.get("native_categorical_models"),
            default=NATIVE_CATEGORICAL_MODELS_DEFAULT,
        )

        one_hot_bool = self.truthy_config_value(one_hot)
        bypass_bool = self.truthy_config_value(bypass)
        native_text = self.report_value(native_models)

        if one_hot == NOT_RUN and bypass == NOT_RUN:
            return NOT_RUN
        if one_hot_bool is True and bypass_bool is True:
            return f"One-hot encoding enabled; native categorical models may bypass it ({native_text})."
        if one_hot_bool is True:
            return "One-hot encoding enabled for all modeling features."
        if one_hot_bool is False and bypass_bool is True:
            return f"One-hot encoding disabled; categorical features are passed to native-capable models ({native_text})."
        if one_hot_bool is False:
            return "One-hot encoding disabled; no native categorical bypass was recorded."
        if bypass_bool is True:
            return f"Native categorical bypass enabled for {native_text}."
        return LEGACY_NOT_RECORDED

    def feature_summary_page_title(self, *, continued: bool = False) -> str:
        title = "Feature Learning, Importance, and Selection"
        return f"{title} (continued)" if continued else title

    def performance_page_title(self) -> str:
        if self.report_mode == "replication":
            return "Replication Performance"
        return "Cross-Validation Performance"

    def evaluation_page_title(self, ds: Dict[str, Any]) -> str:
        is_regression = str(ds.get("task_type") or "").strip().lower() == "regression"
        if self.report_mode == "replication":
            return "Replication Regression Evaluation" if is_regression else "Replication ROC/PRC Evaluation"
        return "Regression Evaluation" if is_regression else "ROC/PRC Evaluation"

    def run_params_by_phase(self, run_params_all: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        if not isinstance(run_params_all, dict):
            return {}

        by_phase: Dict[str, Dict[str, Any]] = {}
        for timestamp in sorted(run_params_all):
            params = run_params_all.get(timestamp)
            if not isinstance(params, dict):
                continue

            phase = str(params.get("phase") or "")
            if not phase and any(
                key in params
                for key in (
                    "data_path",
                    "n_splits",
                    "partition_method",
                    "one_hot_encoding",
                    "categorical_features",
                    "quantitative_features",
                )
            ):
                phase = "p1_data_process"
            if not phase:
                continue
            by_phase[phase] = params
        return by_phase

    def merged_phase_args(
        self,
        command_args: Dict[str, Any],
        run_param_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(run_param_args or {})
        for key, value in (command_args or {}).items():
            if self.value_is_present(value):
                merged[key] = value
        return merged

    def summary_dataset_paths(self) -> Dict[str, List[Path]]:
        primary = self._list_primary_datasets()
        replication = self._list_replication_datasets()
        return {
            "primary": primary,
            "replication": replication,
        }

    def summary_cv_split_count(self, dataset_dirs: Sequence[Path]) -> Optional[int]:
        for ds_dir in dataset_dirs:
            cv_dir = ds_dir / "CVDatasets"
            if cv_dir.is_dir():
                count = len(list(cv_dir.glob("*_Train.csv")))
                if count:
                    return count
        return None

    def summary_feature_importance_models(self, dataset_dirs: Sequence[Path]) -> List[str]:
        models: Set[str] = set()
        for ds_dir in dataset_dirs:
            root = ds_dir / "feature_importance"
            if not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                if child.is_dir() and any(child.iterdir()):
                    models.add(child.name)
        return sorted(models)

    def summary_model_ids(self, dataset_dirs: Sequence[Path]) -> List[str]:
        ids: Set[str] = set()
        for ds_dir in dataset_dirs:
            for path in (ds_dir / "models" / "pickledModels").glob("*.pickle"):
                model_id = re.sub(r"_(?:CV_)?\d+$", "", path.stem)
                if model_id:
                    ids.add(model_id)
            if ids:
                continue
            table = self._read_csv_table(ds_dir / "model_evaluation" / "Summary_performance_mean.csv")
            if table and table.rows and table.columns:
                name_col = table.columns[0]
                for row in table.rows:
                    name = str(row.get(name_col, "")).strip()
                    if name:
                        ids.add(name)
        return sorted(ids)

    def summary_ensemble_ids(self, dataset_dirs: Sequence[Path]) -> List[str]:
        ids: Set[str] = set()
        for ds_dir in dataset_dirs:
            for path in (ds_dir / "ensemble_evaluation" / "metrics_by_cv").glob("*.json"):
                small = re.sub(r"_CV_\d+$", "", path.stem)
                ids.add(ENSEMBLE_SMALL_NAME_TO_ID.get(small, small))
            for path in (ds_dir / "ensemble_evaluation" / "pickled_ensembles").glob("*.pickle"):
                small = re.sub(r"_\d+$", "", path.stem)
                ids.add(ENSEMBLE_SMALL_NAME_TO_ID.get(small, small))
        return sorted(ids)

    def summary_optuna_trials(self, dataset_dirs: Sequence[Path]) -> Optional[str]:
        counts: List[int] = []
        for ds_dir in dataset_dirs:
            for path in (ds_dir / "models" / "optuna_trials").glob("*_optuna_trials*.csv"):
                try:
                    with path.open("r", newline="", encoding="utf-8-sig") as handle:
                        row_count = max(sum(1 for _ in handle) - 1, 0)
                    counts.append(row_count)
                except Exception as exc:
                    logger.warning("Could not count Optuna trials in %s: %r", path, exc)
        if not counts:
            return None
        total = sum(counts)
        return f"{total} completed across {len(counts)} model/CV runs"

    def summary_replication_labels(self, replication_dirs: Sequence[Path]) -> str:
        labels: List[str] = []
        for rep_dir in replication_dirs[:5]:
            parent = rep_dir.parents[1].name if len(rep_dir.parents) > 1 else ""
            labels.append(f"{rep_dir.name} from {parent}" if parent else rep_dir.name)
        if len(replication_dirs) > 5:
            labels.append(f"... ({len(replication_dirs)} total)")
        return ", ".join(labels)

    def report_value(self, value: Any, *, max_len: int = 120) -> str:
        if value is None:
            return "None"
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, (list, tuple, set)):
            items = [str(item) for item in value]
            if not items:
                return "None"
            if len(items) > 5:
                text = ", ".join(items[:5]) + f", ... ({len(items)} total)"
            else:
                text = ", ".join(items)
        elif isinstance(value, dict):
            text = json.dumps(value, sort_keys=True, default=str)
        else:
            text = str(value)

        text = " ".join(text.split())
        if text == "":
            return "None"
        return _shorten(text, max_len)

    def add_summary_line(
        self,
        lines: List[str],
        label: str,
        value: Any,
        *,
        max_len: int = 120,
    ) -> None:
        lines.append(f"{label}: {self.report_value(value, max_len=max_len)}")

    def build_dataset_summary_lines(self, dataset_blocks: Sequence[Dict[str, Any]]) -> List[str]:
        if not dataset_blocks:
            return ["No datasets discovered"]

        lines = [
            f"Dataset Count: {len(dataset_blocks)}",
        ]
        for ds in dataset_blocks[:6]:
            dataset_id = str(ds.get("dataset_id") or "")
            dataset_name = str(ds.get("dataset_name") or "")
            dataset_path = str(ds.get("dataset_path") or "")
            if "/replication/" in dataset_path:
                parent = Path(dataset_path).parents[1].name if len(Path(dataset_path).parents) > 1 else ""
                suffix = f" from {parent}" if parent else ""
                label = f"{dataset_id} = {dataset_name}{suffix}"
            else:
                label = f"{dataset_id} = {dataset_name}"
            lines.append(_shorten(label, 120))
        if len(dataset_blocks) > 6:
            lines.append(f"... {len(dataset_blocks) - 6} more datasets")
        return lines

    def build_run_command_summary(
        self,
        *,
        metadata: Dict[str, Any],
        metadata_pickle: Dict[str, Any],
        run_params_all: Dict[str, Any],
        run_params: Dict[str, Any],
        dataset_blocks: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        context = self.load_run_command_context()
        records = context.get("records", {})
        run_phase_args = self.run_params_by_phase(run_params_all)

        p1 = self.merged_phase_args(self.phase_args_from_records(records, "p1_data_process"), run_phase_args.get("p1_data_process", {}))
        p2 = self.merged_phase_args(self.phase_args_from_records(records, "p2_impute_scale"), run_phase_args.get("p2_impute_scale", {}))
        p3 = self.merged_phase_args(self.phase_args_from_records(records, "p3_feature_learning"), run_phase_args.get("p3_feature_learning", {}))
        p4 = self.merged_phase_args(self.phase_args_from_records(records, "p4_feature_importance"), run_phase_args.get("p4_feature_importance", {}))
        p5 = self.merged_phase_args(self.phase_args_from_records(records, "p5_feature_selection"), run_phase_args.get("p5_feature_selection", {}))
        p6 = self.merged_phase_args(self.phase_args_from_records(records, "p6_modeling"), run_phase_args.get("p6_modeling", {}))
        p7 = self.merged_phase_args(self.phase_args_from_records(records, "p7_ensembles"), run_phase_args.get("p7_ensembles", {}))
        p8 = self.merged_phase_args(self.phase_args_from_records(records, "p8_summary_statistics"), run_phase_args.get("p8_summary_statistics", {}))
        p10 = self.merged_phase_args(self.phase_args_from_records(records, "p10_replication"), run_phase_args.get("p10_replication", {}))

        summary_paths = self.summary_dataset_paths()
        primary_dirs = summary_paths["primary"]
        replication_dirs = summary_paths["replication"]
        recovered_cv_splits = self.summary_cv_split_count(primary_dirs)
        recovered_fi_models = self.summary_feature_importance_models(primary_dirs)
        recovered_model_ids = self.summary_model_ids(primary_dirs)
        recovered_ensemble_ids = self.summary_ensemble_ids(primary_dirs)
        recovered_optuna_trials = self.summary_optuna_trials(primary_dirs)
        recovered_replication_labels = self.summary_replication_labels(replication_dirs)
        is_regression = self.report_is_regression(metadata, dataset_blocks)
        default_metric = REGRESSION_DEFAULT_METRIC if is_regression else "balanced_accuracy"

        p1_ran = bool(p1 or primary_dirs or metadata_pickle)
        p2_ran = bool(p2 or any((ds / "impute_scale").is_dir() and any((ds / "impute_scale").iterdir()) for ds in primary_dirs))
        p3_ran = bool(p3 or any((ds / "feature_learning").is_dir() and any((ds / "feature_learning").iterdir()) for ds in primary_dirs))
        p4_ran = bool(p4 or recovered_fi_models)
        p5_ran = bool(p5 or any((ds / "feature_selection" / "InformativeFeatureSummary.csv").is_file() for ds in primary_dirs))
        p6_ran = bool(p6 or recovered_model_ids)
        p7_ran = bool(p7 or recovered_ensemble_ids)
        p8_ran = bool(p8 or any((ds / "runtime" / "runtime_Stats.txt").is_file() or (ds / "model_evaluation" / "Summary_performance_mean.csv").is_file() for ds in primary_dirs))
        p10_ran = bool(p10 or replication_dirs)

        overview: List[str] = []
        self.add_summary_line(overview, "Report Mode", self.report_mode)
        self.add_summary_line(overview, "Experiment Name", self.experiment_name)
        self.add_summary_line(overview, "Outcome Label", metadata.get("Outcome Label"))
        self.add_summary_line(overview, "Outcome Type", metadata.get("Outcome Type"))
        self.add_summary_line(overview, "Instance Label", metadata.get("Instance Label"))
        self.add_summary_line(overview, "STREAMLINE Version", _try_streamline_version())

        data_cv: List[str] = []
        self.add_summary_line(data_cv, "Data Path", self.phase_summary_value(p1_ran, p1.get("data_path"), metadata_pickle.get("Data Path")))
        self.add_summary_line(data_cv, "CV Splits", self.phase_summary_value(p1_ran, p1.get("n_splits"), p6.get("n_splits"), metadata_pickle.get("CV Partitions"), run_params.get("CV Partitions"), recovered_cv_splits))
        self.add_summary_line(data_cv, "Partition Method", self.phase_summary_value(p1_ran, p1.get("partition_method"), metadata_pickle.get("Partition Method"), run_params.get("Partition Method"), default="Stratified"))
        self.add_summary_line(data_cv, "Categorical Cutoff", self.phase_summary_value(p1_ran, p1.get("categorical_cutoff"), metadata_pickle.get("Categorical Cutoff"), default=10))
        self.add_summary_line(data_cv, "Ignored Features", self.phase_summary_value(p1_ran, p1.get("ignore_features"), metadata_pickle.get("Ignored Features"), default=[]))
        self.add_summary_line(data_cv, "Categorical Features", self.phase_summary_value(p1_ran, p1.get("categorical_features"), metadata_pickle.get("Specified Categorical Features"), default=[]), max_len=120)
        self.add_summary_line(data_cv, "Quantitative Features", self.phase_summary_value(p1_ran, p1.get("quantitative_features"), metadata_pickle.get("Specified Quantitative Features"), default=[]), max_len=120)

        processing: List[str] = []
        self.add_summary_line(processing, "Scale Data", self.phase_summary_value(p2_ran, p2.get("scale_data"), metadata_pickle.get("Use Data Scaling"), default=True))
        self.add_summary_line(processing, "Impute Data", self.phase_summary_value(p2_ran, p2.get("impute_data"), metadata_pickle.get("Use Data Imputation"), default=True))
        self.add_summary_line(processing, "Multivariate Imputation", self.phase_summary_value(p2_ran, p2.get("multi_impute"), metadata_pickle.get("Use Multivariate Imputation"), default=False))
        self.add_summary_line(processing, "Overwrite CV", self.phase_summary_value(p2_ran, p2.get("overwrite_cv"), metadata_pickle.get("Overwrite CV Datasets"), default=True))
        self.add_summary_line(processing, "SMOTE", self.phase_summary_value(p2_ran, p2.get("smote"), metadata_pickle.get("Use SMOTE"), default=False))
        self.add_summary_line(processing, "SMOTE Method", self.phase_summary_value(p2_ran, p2.get("smote_method"), metadata_pickle.get("P2 SMOTE Method"), default="auto"))
        self.add_summary_line(processing, "Missingness Cutoff", self.phase_summary_value(p1_ran, p1.get("featureeng_missingness"), metadata_pickle.get("Engineering Missingness Cutoff"), default=0.5))
        self.add_summary_line(processing, "Correlation Removal", self.phase_summary_value(p1_ran, p1.get("correlation_removal_threshold"), metadata_pickle.get("Correlation Removal Threshold"), default=1.0))

        feature_selection: List[str] = []
        self.add_summary_line(feature_selection, "Feature Learner", self.phase_summary_value(p3_ran, p3.get("learner_id"), metadata_pickle.get("P3 Learner Id"), default="pca"))
        self.add_summary_line(feature_selection, "Keep Original Features", self.phase_summary_value(p3_ran, p3.get("keep_original_features"), metadata_pickle.get("P3 Keep Original Features"), default=True))
        self.add_summary_line(feature_selection, "FI Models", self.phase_summary_value(p4_ran, p4.get("models"), metadata_pickle.get("P4 Models"), recovered_fi_models, default=["mutualinformation", "multisurf"]))
        self.add_summary_line(feature_selection, "FI Params", self.phase_summary_value(p4_ran, p4.get("models_params"), metadata_pickle.get("P4 Models Params"), default={}), max_len=130)
        self.add_summary_line(feature_selection, "FI Instance Subset", self.phase_summary_value(p4_ran, p4.get("instance_subset"), metadata_pickle.get("P4 Instance Subset"), default=2000))
        self.add_summary_line(feature_selection, "P5 Algorithms", self.phase_summary_value(p5_ran, p5.get("algorithms"), recovered_fi_models, default="auto"))
        self.add_summary_line(feature_selection, "Max Features To Keep", self.phase_summary_value(p5_ran, p5.get("max_features_to_keep"), metadata_pickle.get("Max Features to Keep"), default=2000))
        self.add_summary_line(feature_selection, "Top Features To Display", self.phase_summary_value(p5_ran or p8_ran, p5.get("top_features"), p8.get("top_features"), metadata_pickle.get("Top Model Features to Display"), default=20))

        modeling: List[str] = []
        self.add_summary_line(modeling, "Model Type", self.phase_summary_value(p6_ran, p6.get("model_type"), metadata.get("Outcome Type")))
        self.add_summary_line(modeling, "Models", self.phase_summary_value(p6_ran, p6.get("models"), recovered_model_ids, default="auto/default"))
        scoring_metric = self.phase_summary_value(p6_ran or p8_ran, p6.get("scoring_metric"), p8.get("scoring_metric"), metadata_pickle.get("Primary Metric"), default=default_metric)
        self.add_summary_line(modeling, "Scoring Metric", self.report_metric_for_outcome(scoring_metric, is_regression, default=default_metric))
        self.add_summary_line(modeling, "Metric Direction", self.phase_summary_value(p6_ran, p6.get("metric_direction"), default="maximize"))
        if self.value_is_present(p6.get("n_trials")):
            self.add_summary_line(modeling, "Optuna Trials Requested", p6.get("n_trials"))
        if self.value_is_present(p6.get("timeout")):
            self.add_summary_line(modeling, "Optuna Timeout Seconds", p6.get("timeout"))
        self.add_summary_line(modeling, "Optuna Trials Completed", self.phase_summary_value(p6_ran, recovered_optuna_trials, default="0 completed"))
        self.add_summary_line(modeling, "Training Subsample", self.phase_summary_value(p6_ran, p6.get("training_subsample"), default=0))
        self.add_summary_line(modeling, "Calibration", self.phase_summary_value(p6_ran, p6.get("calibrate"), default=False))
        self.add_summary_line(modeling, "Categorical Handling", self.categorical_handling_summary(p1_ran=p1_ran, p6_ran=p6_ran, p1=p1, p6=p6, metadata_pickle=metadata_pickle), max_len=150)
        self.add_summary_line(modeling, "Ensembles", self.phase_summary_value(p7_ran, p7.get("ensembles"), recovered_ensemble_ids, default="hard_voting,soft_voting,stack_lr"))
        self.add_summary_line(modeling, "Base Models", self.phase_summary_value(p7_ran, p7.get("base_models"), p6.get("models"), recovered_model_ids, default="auto/default"))

        replication: List[str] = []
        if self.report_mode == "replication":
            self.add_summary_line(replication, "Replication Data Path", self.phase_summary_value(p10_ran, p10.get("rep_data_path"), recovered_replication_labels), max_len=130)
            self.add_summary_line(replication, "Training Dataset For Rep", self.phase_summary_value(p10_ran, p10.get("dataset_for_rep"), [ds.name for ds in primary_dirs]), max_len=130)
            self.add_summary_line(replication, "Match Label", self.phase_summary_value(p10_ran, p10.get("match_label"), metadata_pickle.get("Match Label"), default="None"))
            self.add_summary_line(replication, "P10 Show Plots", self.phase_summary_value(p10_ran, p10.get("show_plots"), default=False))
            self.add_summary_line(replication, "Rep Report Focus", "Held-out/external replication folders only")

        reporting: List[str] = []
        self.add_summary_line(reporting, "Report Mode", self.report_mode)
        self.add_summary_line(reporting, "Make PDF", self.make_pdf)
        self.add_summary_line(reporting, "Enable Plots", self.enable_plots)
        self.add_summary_line(reporting, "Reuse Existing Figures", self.reuse_existing_figures)
        p8_metric_weight = self.phase_summary_value(p8_ran, p8.get("metric_weight"), default=default_metric)
        self.add_summary_line(reporting, "P8 Metric Weight", self.report_metric_for_outcome(p8_metric_weight, is_regression, default=default_metric))
        self.add_summary_line(reporting, "P8 Include Ensembles", self.phase_summary_value(p8_ran, p8.get("include_ensembles"), default=True))

        dataset_lines = self.build_dataset_summary_lines(dataset_blocks)

        sections = [
            {"title": "Run Overview", "lines": overview},
            {"title": "P1 Data Processing and CV", "lines": data_cv},
            {"title": "P1-P2 EDA, Scaling, Imputation, and SMOTE", "lines": processing},
            {"title": "P3-P5 Feature Learning, Importance, and Selection", "lines": feature_selection},
            {"title": "P6-P8 Modeling, Ensembles, and Metrics", "lines": modeling},
        ]
        if replication:
            sections.append({"title": "P10 Replication Settings", "lines": replication})
        sections.extend(
            [
                {"title": "P11 Reporting Settings", "lines": reporting},
                {"title": "Target Dataset(s)", "lines": dataset_lines},
            ]
        )

        return {
            **context,
            "sections": sections,
        }

    def _read_csv_table(self, path: Path) -> Optional[TableData]:
        if not path.is_file():
            return None
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    return None
                columns = [(h or "").strip() for h in header]
                if columns and columns[0] == "":
                    columns[0] = "Algorithm"
                rows: List[Dict[str, str]] = []
                for raw in reader:
                    if not raw:
                        continue
                    if len(raw) < len(columns):
                        raw = raw + [""] * (len(columns) - len(raw))
                    row = {columns[i]: (raw[i] if i < len(raw) else "").strip() for i in range(len(columns))}
                    rows.append(row)
                return TableData(columns=columns, rows=rows)
        except Exception as exc:
            logger.warning("Failed reading csv %s: %r", path, exc)
            return None

    def _read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def _report_figure_candidates(self, filename: str, ds_dir: Optional[Path] = None) -> List[Path]:
        """
        Candidate locations for figures generated by reporting itself.
        """
        candidates: List[Path] = [
            self.paths.figures_dir / filename,
            self.exp_root / "reporting" / "figures" / filename,
            self.exp_root / "reporting_replication" / "figures" / filename,
        ]
        if ds_dir is not None:
            # Legacy dataset-local reporting location kept for backwards compatibility.
            candidates.append(ds_dir / "reporting" / "figures" / filename)

        # Keep order stable but remove duplicates.
        out: List[Path] = []
        seen: Set[Path] = set()
        for p in candidates:
            rp = p.resolve() if p.exists() else p
            if rp in seen:
                continue
            seen.add(rp)
            out.append(p)
        return out

    def _reuse_generated_report_figure(self, filename: str, ds_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Reuse a previously generated reporting figure across standard/replication modes.
        If found in another report folder, copy into the current report folder for portability.
        """
        current = self.paths.figures_dir / filename
        if current.is_file():
            return current

        existing = _first_existing(self._report_figure_candidates(filename, ds_dir=ds_dir))
        if existing is None:
            return None

        try:
            if existing.resolve() != current.resolve():
                current.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(existing, current)
                return current
        except Exception as exc:
            logger.warning("Could not copy reused figure %s -> %s: %r", existing, current, exc)
        return existing

    def _list_primary_datasets(self) -> List[Path]:
        datasets: List[Path] = []
        ignore = {
            "jobs",
            "logs",
            "jobsCompleted",
            "dask_logs",
            "runtime",
            "DatasetComparisons",
            "reporting",
            "reporting_replication",
        }
        for p in sorted(self.exp_root.iterdir()):
            if not p.is_dir():
                continue
            if p.name in ignore:
                continue
            if (p / "exploratory").is_dir() and (p / "model_evaluation").is_dir():
                datasets.append(p)
        return datasets

    def _list_replication_datasets(self) -> List[Path]:
        datasets: List[Path] = []
        seen: Set[Path] = set()

        # Primary expected layout:
        # <experiment>/<train_dataset>/replication/<rep_dataset>/
        for train_ds in self._list_primary_datasets():
            rep_root = train_ds / "replication"
            if not rep_root.is_dir():
                continue
            for rep_ds in sorted(rep_root.iterdir()):
                if not rep_ds.is_dir():
                    continue
                if (rep_ds / "exploratory").is_dir() and (rep_ds / "model_evaluation").is_dir():
                    r = rep_ds.resolve()
                    if r not in seen:
                        datasets.append(rep_ds)
                        seen.add(r)

        # Fallback: search recursively for replication folders that match expected artifacts.
        if not datasets:
            for rep_root in sorted(self.exp_root.glob("**/replication")):
                if not rep_root.is_dir():
                    continue
                for rep_ds in sorted(rep_root.iterdir()):
                    if not rep_ds.is_dir():
                        continue
                    if (rep_ds / "exploratory").is_dir() and (rep_ds / "model_evaluation").is_dir():
                        r = rep_ds.resolve()
                        if r not in seen:
                            datasets.append(rep_ds)
                            seen.add(r)

        return datasets

    def _list_datasets(self) -> List[Path]:
        if self.report_mode == "replication":
            return self._list_replication_datasets()
        return self._list_primary_datasets()

    def _is_regression_from_values(self, values: Sequence[str]) -> bool:
        cleaned = [v for v in values if str(v).strip() != ""]
        if not cleaned:
            return False
        numeric = []
        for v in cleaned:
            fv = _safe_float(v)
            if fv is None:
                return False
            numeric.append(fv)
        unique = len(set(numeric))
        unique_fraction = unique / float(len(numeric))
        return unique > 20 or unique_fraction > 0.2

    def _find_target_column(self, header: Sequence[str], metadata: Dict[str, Any]) -> str:
        cols = [str(c).strip() for c in header if str(c).strip() != ""]
        if not cols:
            return header[0] if header else ""

        low_map = {c.lower(): c for c in cols}

        def _match_col(name: Any) -> Optional[str]:
            txt = str(name or "").strip()
            if txt == "":
                return None
            if txt in cols:
                return txt
            return low_map.get(txt.lower())

        instance_col = _match_col(self.instance_label) or _match_col(metadata.get("Instance Label"))
        outcome_col = _match_col(self.outcome_label) or _match_col(metadata.get("Outcome Label"))
        outcome_type = str(self.outcome_type or metadata.get("Outcome Type") or "").strip().lower()

        # Use explicit outcome label only when it does not collide with instance id.
        if outcome_col and (not instance_col or outcome_col.lower() != instance_col.lower()):
            return outcome_col

        is_regression_like = outcome_type in {"continuous", "regression", "numeric", "real", "float"}
        if is_regression_like:
            semantic_tokens = ("target", "outcome", "score", "response", "phenotype", "label", "y")
            id_like_names = {"class", "id", "instanceid", "instance_id", "sampleid", "sample_id"}

            for col in cols:
                cl = col.lower()
                if instance_col and cl == instance_col.lower():
                    continue
                if cl in id_like_names:
                    continue
                if any(tok in cl for tok in semantic_tokens):
                    return col

            for col in cols:
                cl = col.lower()
                if instance_col and cl == instance_col.lower():
                    continue
                if cl in id_like_names:
                    continue
                if cl.endswith("_id") or cl.endswith("id") or "instance" in cl:
                    continue
                return col

        class_col = _match_col("Class")
        if class_col:
            return class_col

        if outcome_col:
            return outcome_col

        if instance_col:
            for col in cols:
                if col.lower() != instance_col.lower():
                    return col

        return cols[0]

    def _detect_task_from_train(self, ds_dir: Path, metadata: Dict[str, Any]) -> str:
        cv_dir = ds_dir / "CVDatasets"
        train_files = sorted(cv_dir.glob("*_Train.csv"))
        if not train_files:
            return "Binary Classification"
        train = train_files[0]
        try:
            with train.open("r", newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    return "Binary Classification"
                target = self._find_target_column(header, metadata)
                try:
                    idx = list(header).index(target)
                except ValueError:
                    idx = 0
                values: List[str] = []
                for row in reader:
                    if idx < len(row):
                        values.append(row[idx].strip())
                    if len(values) >= 5000:
                        break
                unique_vals = sorted(set([v for v in values if v != ""]))
                unique = len(unique_vals)
                if self._is_regression_from_values(values):
                    return "Regression"
                if unique == 2:
                    return "Binary Classification"
                if unique > 2:
                    return "Multiclass Classification"
        except Exception as exc:
            logger.warning("Task detection fallback failed for %s: %r", ds_dir, exc)
        return "Binary Classification"

    def _detect_task_type(self, ds_dir: Path, metadata: Dict[str, Any]) -> str:
        # Rule priority:
        # 1) ClassCounts.csv if clearly binary.
        # 2) ClassCounts with high-cardinality numeric labels can indicate regression.
        # 3) Otherwise infer from *_Train.csv target values.
        cc = self._read_csv_table(ds_dir / "exploratory" / "ClassCounts.csv")
        if cc and cc.rows:
            label_col = cc.columns[0]
            labels = [row.get(label_col, "") for row in cc.rows if row.get(label_col, "").strip() != ""]
            unique = len(set(labels))
            if unique == 2:
                return "Binary Classification"
            if unique > 2:
                # Keep ClassCounts-driven behavior aligned with classification first.
                # Only treat as regression here when cardinality is clearly continuous-like.
                if len(set(labels)) > 20 and all(_safe_float(v) is not None for v in labels):
                    return "Regression"
                return "Multiclass Classification"
        return self._detect_task_from_train(ds_dir, metadata)

    def _metric_list_for_task(self, task_type: str) -> List[str]:
        if task_type == "Regression":
            return REGRESSION_METRICS[:]
        return CLASSIFICATION_METRICS[:]

    def _metric_default_distribution(self, task_type: str, columns: Sequence[str]) -> Optional[str]:
        if task_type == "Regression":
            if "Pearson Correlation" in columns:
                return "Pearson Correlation"
            if "Mean Absolute Error" in columns:
                return "Mean Absolute Error"
            return columns[0] if columns else None
        if "Balanced Accuracy" in columns:
            return "Balanced Accuracy"
        if "Accuracy" in columns:
            return "Accuracy"
        return columns[0] if columns else None

    def _find_algorithm_col(self, columns: Sequence[str]) -> str:
        preferred = {"algorithm", "ml algorithm", "model", "ensemble"}
        for c in columns:
            if c.strip().lower() in preferred:
                return c
        return columns[0] if columns else "Algorithm"

    def _extract_class_count_info(self, class_counts: Optional[TableData]) -> Tuple[List[str], List[float]]:
        if not class_counts or not class_counts.rows:
            return [], []
        label_col = class_counts.columns[0]
        count_col = class_counts.columns[1] if len(class_counts.columns) > 1 else class_counts.columns[0]
        labels: List[str] = []
        counts: List[float] = []
        for row in class_counts.rows:
            label = row.get(label_col, "")
            count = _safe_float(row.get(count_col, ""))
            if label.strip() == "" or count is None:
                continue
            labels.append(label)
            counts.append(count)
        return labels, counts

    @staticmethod
    def positive_label_text(values: Sequence[str]) -> str:
        cleaned = [str(v).strip() for v in values if str(v).strip() != ""]
        if not cleaned:
            return "1"
        if "1" in cleaned:
            return "1"
        if "True" in cleaned:
            return "True"
        if "true" in cleaned:
            return "true"
        try:
            return sorted(set(cleaned), key=lambda x: float(x))[-1]
        except Exception:
            return sorted(set(cleaned))[-1]

    def no_skill_from_outcome_folds(
        self,
        folds: Sequence[Sequence[str]],
        task_type: Optional[str] = None,
        class_count_labels: Optional[Sequence[str]] = None,
    ) -> Optional[float]:
        non_empty = [
            [str(v).strip() for v in fold if str(v).strip() != ""]
            for fold in folds
        ]
        non_empty = [fold for fold in non_empty if fold]
        if not non_empty:
            return None

        all_values = [value for fold in non_empty for value in fold]
        labels = sorted(set(all_values))
        if "multiclass" in str(task_type or "").lower() or len(labels) > 2:
            class_count = max(len(labels), len(class_count_labels or []), 1)
            return max(1e-6, min(1.0, 1.0 / float(class_count)))

        positive = self.positive_label_text(all_values)
        fold_rates = [
            sum(1 for value in fold if value == positive) / float(len(fold))
            for fold in non_empty
        ]
        if not fold_rates:
            return None
        return max(1e-6, min(1.0, sum(fold_rates) / float(len(fold_rates))))

    def cv_test_outcome_folds(self, ds_dir: Optional[Path], outcome_label: str) -> List[List[str]]:
        if ds_dir is None:
            return []
        cv_dir = ds_dir / "CVDatasets"
        if not cv_dir.is_dir():
            return []

        folds: List[List[str]] = []
        for path in sorted(cv_dir.glob(f"{ds_dir.name}_CV_*_Test.csv")):
            try:
                with path.open("r", newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames or []
                    low = {name.lower(): name for name in fieldnames}
                    target = outcome_label if outcome_label in fieldnames else low.get(outcome_label.lower())
                    if not target:
                        continue
                    values = [
                        str(row.get(target, "")).strip()
                        for row in reader
                        if str(row.get(target, "")).strip() != ""
                    ]
                    if values:
                        folds.append(values)
            except Exception as exc:
                logger.warning("Could not read CV outcomes for no-skill baseline from %s: %r", path, exc)
        return folds

    def _classification_no_skill(
        self,
        class_counts: Optional[TableData],
        ds_dir: Optional[Path] = None,
        outcome_label: str = "Class",
        task_type: Optional[str] = None,
    ) -> float:
        cv_baseline = self.no_skill_from_outcome_folds(
            self.cv_test_outcome_folds(ds_dir, outcome_label),
            task_type=task_type,
            class_count_labels=self._extract_class_count_info(class_counts)[0],
        )
        if cv_baseline is not None:
            return cv_baseline

        labels, counts = self._extract_class_count_info(class_counts)
        if not counts:
            return 0.5
        total = sum(counts)
        if total <= 0:
            return 0.5
        if len(labels) == 2:
            # Prefer class label "1" if available, else second class.
            if "1" in labels:
                idx = labels.index("1")
                return max(1e-6, min(1.0, counts[idx] / total))
            return max(1e-6, min(1.0, counts[1] / total))
        return max(1e-6, min(1.0, 1.0 / float(len(labels))))

    def _mpl_ok(self) -> bool:
        if not self.enable_plots:
            return False
        if self._mpl_ready is not None:
            return self._mpl_ready
        try:
            mpl_cfg = self.paths.reporting_dir / ".mplconfig"
            mpl_cfg.mkdir(parents=True, exist_ok=True)
            mpl_cache = self.paths.reporting_dir / ".cache"
            mpl_cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", str(mpl_cfg))
            os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache))
            import matplotlib  # type: ignore

            matplotlib.use("Agg")
            matplotlib.rcParams.update(
                {
                    "font.family": "sans-serif",
                    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
                    "axes.titlesize": 10,
                    "axes.titleweight": "normal",
                    "axes.labelsize": 9,
                    "xtick.labelsize": 8,
                    "ytick.labelsize": 8,
                    "legend.fontsize": 8,
                }
            )
            self._mpl_ready = True
        except Exception as exc:
            logger.warning("Matplotlib not available. Figure auto-generation disabled: %r", exc)
            self._mpl_ready = False
        return self._mpl_ready

    def _save_simple_placeholder(self, out: Path, title: str, body: str) -> Optional[str]:
        if self._mpl_ok():
            try:
                import matplotlib.pyplot as plt  # type: ignore

                out.parent.mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots(figsize=(7.5, 4.0))
                ax.axis("off")
                ax.text(0.5, 0.50, body, ha="center", va="center", fontsize=10)
                fig.tight_layout()
                fig.savefig(out, dpi=180)
                plt.close(fig)
                return str(out)
            except Exception as exc:
                logger.warning("Matplotlib placeholder figure failed for %s: %r", out, exc)
        return None

    def _plot_missingness_top25(self, table: Optional[TableData], out: Path, title: str) -> Optional[str]:
        if not table or not table.rows:
            return None
        low = {c.lower(): c for c in table.columns}
        feature_col = low.get("feature") or low.get("variable") or table.columns[0]
        value_col = low.get("count") or low.get("missing_count") or low.get("missingcount") or table.columns[-1]
        rows: List[Tuple[str, float]] = []
        for row in table.rows:
            feat = row.get(feature_col, "")
            val = _safe_float(row.get(value_col, ""))
            if feat and val is not None:
                rows.append((feat, val))
        if not rows:
            return None
        rows = sorted(rows, key=lambda x: x[1], reverse=True)[:25]
        labels = [x[0] for x in rows][::-1]
        vals = [x[1] for x in rows][::-1]

        if self._mpl_ok():
            try:
                import matplotlib.pyplot as plt  # type: ignore

                out.parent.mkdir(parents=True, exist_ok=True)
                fig_h = max(4.0, 0.18 * len(labels))
                fig, ax = plt.subplots(figsize=(8.0, fig_h))
                ax.barh(labels, vals, color="#4C78A8")
                ax.set_xlabel(value_col)
                ax.set_ylabel("Feature")
                fig.tight_layout()
                fig.savefig(out, dpi=180)
                plt.close(fig)
                return str(out)
            except Exception as exc:
                logger.warning("Missingness plot generation failed with matplotlib: %r", exc)
        return None

    def _plot_class_balance(self, table: Optional[TableData], out: Path, title: str) -> Optional[str]:
        if not table or not table.rows:
            return None
        labels, counts = self._extract_class_count_info(table)
        if not counts:
            return None

        if self._mpl_ok():
            try:
                import matplotlib.pyplot as plt  # type: ignore

                out.parent.mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots(figsize=(7.2, 4.4))
                ax.bar(labels, counts, color="#59A14F")
                ax.set_xlabel("Class")
                ax.set_ylabel("Count")
                for tick in ax.get_xticklabels():
                    tick.set_rotation(35)
                    tick.set_ha("right")
                fig.tight_layout()
                fig.savefig(out, dpi=180)
                plt.close(fig)
                return str(out)
            except Exception as exc:
                logger.warning("Class balance plot generation failed with matplotlib: %r", exc)
        return None

    def _plot_target_distribution(self, ds_dir: Path, metadata: Dict[str, Any], out: Path, title: str) -> Optional[str]:
        cv_dir = ds_dir / "CVDatasets"
        train_files = sorted(cv_dir.glob("*_Train.csv"))
        if not train_files:
            return None
        try:
            with train_files[0].open("r", newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    return None
                target = self._find_target_column(header, metadata)
                idx = list(header).index(target) if target in header else 0
                vals: List[float] = []
                for row in reader:
                    if idx < len(row):
                        fv = _safe_float(row[idx].strip())
                        if fv is not None:
                            vals.append(fv)
                if not vals:
                    return None
            if self._mpl_ok():
                import matplotlib.pyplot as plt  # type: ignore

                out.parent.mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots(figsize=(7.2, 4.4))
                bins = min(40, max(12, int(round(math.sqrt(len(vals))))))
                ax.hist(vals, bins=bins, histtype="bar", color="#E15759", alpha=0.9, edgecolor="#FFFFFF", linewidth=0.6)
                ax.set_xlabel("Target value")
                ax.set_ylabel("Frequency")
                fig.tight_layout()
                fig.savefig(out, dpi=180)
                plt.close(fig)
                return str(out)
        except Exception as exc:
            logger.warning("Target distribution plot generation failed with matplotlib path: %r", exc)
        return None

    def _figure_path_exploratory_correlation_matrix(self, ds_dir: Path) -> Optional[Path]:
        reused = self._reuse_generated_report_figure(f"{ds_dir.name}_corr_matrix.png", ds_dir=ds_dir) \
            if self.reuse_existing_figures else None
        return _first_existing(
            [
                ds_dir / "exploratory" / "FeatureCorrelationMatrix.png",
                ds_dir / "exploratory" / "FeatureCorrelation.png",
                ds_dir / "exploratory" / "CorrelationMatrix.png",
                ds_dir / "exploratory" / "CorrelationHeatmap.png",
                ds_dir / "exploratory" / "feature_correlation" / "CorrelationMatrix.png",
                ds_dir / "exploratory" / "feature_correlation" / "FeatureCorrelationMatrix.png",
                ds_dir / "exploratory" / "FeatureCorrelation" / "CorrelationMatrix.png",
                reused if reused is not None else (self.paths.figures_dir / f"{ds_dir.name}_corr_matrix.png"),
            ]
        )

    def _find_exploratory_correlation_csv(self, ds_dir: Path) -> Optional[Path]:
        return _first_existing(
            [
                ds_dir / "exploratory" / "FeatureCorrelations.csv",
                ds_dir / "exploratory" / "FeatureCorrelationMatrix.csv",
                ds_dir / "exploratory" / "CorrelationMatrix.csv",
                ds_dir / "exploratory" / "FeatureCorrelation.csv",
                ds_dir / "exploratory" / "feature_correlation" / "CorrelationMatrix.csv",
                ds_dir / "exploratory" / "FeatureCorrelation" / "CorrelationMatrix.csv",
                ds_dir / "exploratory" / "initial" / "FeatureCorrelations.csv",
            ]
        )

    def _read_correlation_matrix_csv(self, path: Path) -> Tuple[List[str], List[List[float]]]:
        labels: List[str] = []
        matrix: List[List[float]] = []
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as f:
                rows = list(csv.reader(f))
            if not rows:
                return labels, matrix

            header = rows[0]
            has_row_label_col = bool(header) and (header[0].strip() == "" or header[0].strip().lower() in {"feature", "variable", "var", "unnamed: 0"})
            col_labels = [h.strip() for h in (header[1:] if has_row_label_col else header)]
            if not col_labels:
                return labels, matrix

            for idx, row in enumerate(rows[1:]):
                if not row:
                    continue
                if has_row_label_col and len(row) >= 2:
                    row_label = row[0].strip() or f"R{idx+1}"
                    vals_raw = row[1:]
                else:
                    row_label = col_labels[idx] if idx < len(col_labels) else f"R{idx+1}"
                    vals_raw = row
                vals: List[float] = []
                for j, v in enumerate(vals_raw):
                    fv = _safe_float(v)
                    if fv is None or math.isnan(fv):
                        # Keep matrix dense for plotting.
                        if row_label in col_labels and j < len(col_labels) and col_labels[j] == row_label:
                            fv = 1.0
                        else:
                            fv = 0.0
                    vals.append(float(fv))
                if vals:
                    matrix.append(vals)
                    labels.append(row_label)

            if not matrix:
                return [], []

            # Make matrix rectangular and then square by clipping/padding.
            max_cols = max(len(r) for r in matrix)
            for r in matrix:
                if len(r) < max_cols:
                    r.extend([0.0] * (max_cols - len(r)))

            n = min(len(matrix), max_cols, len(col_labels))
            matrix = [row[:n] for row in matrix[:n]]
            labels = col_labels[:n] if len(col_labels) >= n else labels[:n]
            return labels, matrix
        except Exception as exc:
            logger.warning("Could not read correlation matrix CSV %s: %r", path, exc)
            return [], []

    def _plot_correlation_matrix_from_csv(self, csv_path: Path, out: Path, title: str) -> Optional[str]:
        labels, matrix = self._read_correlation_matrix_csv(csv_path)
        if not labels or not matrix:
            return None
        if self._mpl_ok():
            try:
                import matplotlib.pyplot as plt  # type: ignore

                n = len(labels)
                fig_size = min(11.0, max(5.8, 0.18 * n))
                out.parent.mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots(figsize=(fig_size, fig_size))
                img = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="equal", interpolation="nearest")
                if n <= 30:
                    ax.set_xticks(list(range(n)))
                    ax.set_yticks(list(range(n)))
                    ax.set_xticklabels(labels, fontsize=6, rotation=90)
                    ax.set_yticklabels(labels, fontsize=6)
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
                ax.tick_params(length=0)
                cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=7)
                fig.tight_layout()
                fig.savefig(out, dpi=180)
                plt.close(fig)
                return str(out)
            except Exception as exc:
                logger.warning("Correlation matrix plot generation failed with matplotlib for %s: %r", csv_path, exc)
        return None

    def _parse_mutual_info_scores(self, ds_dir: Path) -> List[Tuple[str, float]]:
        candidates = [
            ds_dir / "feature_importance" / "mutualinformation",
            ds_dir / "feature_importance" / "mutual_information",
            ds_dir / "feature_selection" / "mutualinformation",
            ds_dir / "feature_selection" / "mutual_information",
        ]
        score_map: Dict[str, List[float]] = {}
        for base in candidates:
            if not base.is_dir():
                continue
            for path in sorted(base.glob("mutualinformation_scores_cv_*.csv")):
                table = self._read_csv_table(path)
                if not table or not table.rows:
                    continue
                low = {c.lower(): c for c in table.columns}
                fcol = low.get("feature") or table.columns[0]
                scol = low.get("score") or table.columns[-1]
                for row in table.rows:
                    feat = row.get(fcol, "").strip()
                    score = _safe_float(row.get(scol, ""))
                    if feat and score is not None:
                        score_map.setdefault(feat, []).append(score)
        medians: List[Tuple[str, float]] = []
        for feat, vals in score_map.items():
            if vals:
                medians.append((feat, statistics.median(vals)))
        medians.sort(key=lambda t: t[1], reverse=True)
        return medians

    def _plot_mutual_info_top(self, ds_dir: Path, out: Path, top_n: int = 20) -> Optional[str]:
        data = self._parse_mutual_info_scores(ds_dir)
        if not data:
            return None
        data = data[:top_n]
        labels = [d[0] for d in data][::-1]
        vals = [d[1] for d in data][::-1]
        if self._mpl_ok():
            try:
                import matplotlib.pyplot as plt  # type: ignore

                out.parent.mkdir(parents=True, exist_ok=True)
                fig_side = max(5.8, min(7.2, 0.25 * len(labels) + 2.4))
                labels = [
                    f"{rank}. {feature}"
                    for rank, (feature, _score) in enumerate(data, start=1)
                ][::-1]
                fig, ax = plt.subplots(figsize=(fig_side, fig_side))
                ax.barh(labels, vals, color="#86A8CA", height=0.55)
                ax.set_xlabel("Median score across CV folds")
                ax.set_ylabel("Ranked feature order")
                ax.grid(axis="x", color="#E1E5EA", linewidth=0.7)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                fig.subplots_adjust(left=0.42, right=0.97, top=0.96, bottom=0.12)
                fig.savefig(out, dpi=180)
                plt.close(fig)
                return str(out)
            except Exception as exc:
                logger.warning("Mutual information plot generation failed with matplotlib: %r", exc)
        return None

    def _parse_feature_scores_method(
        self,
        ds_dir: Path,
        *,
        dir_aliases: Sequence[str],
        score_patterns: Sequence[str],
    ) -> List[Tuple[str, float]]:
        score_map: Dict[str, List[float]] = {}
        for root_name in ("feature_importance", "feature_selection"):
            root = ds_dir / root_name
            if not root.is_dir():
                continue
            for alias in dir_aliases:
                method_dir = root / alias
                if not method_dir.is_dir():
                    continue
                files: List[Path] = []
                for pattern in score_patterns:
                    files.extend(sorted(method_dir.glob(pattern)))
                if not files:
                    files.extend(sorted(method_dir.glob("*scores_cv_*.csv")))

                for path in files:
                    table = self._read_csv_table(path)
                    if not table or not table.rows:
                        continue
                    low = {c.lower(): c for c in table.columns}
                    fcol = low.get("feature") or table.columns[0]
                    scol = low.get("score") or low.get("importance") or table.columns[-1]
                    for row in table.rows:
                        feat = row.get(fcol, "").strip()
                        score = _safe_float(row.get(scol, ""))
                        if feat and score is not None:
                            score_map.setdefault(feat, []).append(score)

        medians: List[Tuple[str, float]] = []
        for feat, vals in score_map.items():
            if vals:
                medians.append((feat, statistics.median(vals)))
        medians.sort(key=lambda t: t[1], reverse=True)
        return medians

    def _plot_feature_scores_method_top(
        self,
        ds_dir: Path,
        out: Path,
        *,
        dir_aliases: Sequence[str],
        score_patterns: Sequence[str],
        top_n: int = 20,
    ) -> Optional[str]:
        data = self._parse_feature_scores_method(
            ds_dir,
            dir_aliases=dir_aliases,
            score_patterns=score_patterns,
        )
        if not data:
            return None
        data = data[:top_n]
        labels = [d[0] for d in data][::-1]
        vals = [d[1] for d in data][::-1]
        if self._mpl_ok():
            try:
                import matplotlib.pyplot as plt  # type: ignore

                out.parent.mkdir(parents=True, exist_ok=True)
                fig_side = max(5.8, min(7.2, 0.25 * len(labels) + 2.4))
                labels = [
                    f"{rank}. {feature}"
                    for rank, (feature, _score) in enumerate(data, start=1)
                ][::-1]
                fig, ax = plt.subplots(figsize=(fig_side, fig_side))
                ax.barh(labels, vals, color="#86A8CA", height=0.55)
                ax.set_xlabel("Median score across CV folds")
                ax.set_ylabel("Ranked feature order")
                ax.grid(axis="x", color="#E1E5EA", linewidth=0.7)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                fig.subplots_adjust(left=0.42, right=0.97, top=0.96, bottom=0.12)
                fig.savefig(out, dpi=180)
                plt.close(fig)
                return str(out)
            except Exception as exc:
                logger.warning("Feature score plot generation failed with matplotlib: %r", exc)
        return None

    def _resolve_feature_learning_panels(self, ds_dir: Path, dataset_name: str) -> List[Dict[str, str]]:
        panels: List[Dict[str, str]] = []
        for spec in FEATURE_LEARNING_METHODS:
            method_key = str(spec.get("key", "")).strip()
            if method_key == "":
                continue
            label = str(spec.get("label", method_key)).strip()
            dir_aliases = [str(x) for x in spec.get("dir_aliases", [])]
            score_patterns = [str(x) for x in spec.get("score_patterns", [])]
            out = self.paths.figures_dir / f"{dataset_name}_{method_key}_top20.png"

            fig_path: Optional[str] = None
            if self.reuse_existing_figures:
                reused = self._reuse_generated_report_figure(
                    f"{dataset_name}_{method_key}_top20.png",
                    ds_dir=ds_dir,
                )
                if reused is not None:
                    fig_path = str(reused)
            if fig_path is None and self.enable_plots:
                if method_key == "mutual_info":
                    fig_path = self._plot_mutual_info_top(ds_dir, out, top_n=20)
                else:
                    fig_path = self._plot_feature_scores_method_top(
                        ds_dir,
                        out,
                        dir_aliases=dir_aliases,
                        score_patterns=score_patterns,
                        top_n=20,
                    )
            if fig_path is None and out.is_file():
                fig_path = str(out)

            if fig_path:
                panels.append({"key": method_key, "title": f"Top Scores ({label})", "path": fig_path})

        return panels

    def format_feature_summary_value(self, values: Sequence[float]) -> str:
        clean_values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        if not clean_values:
            return ""
        mean_value = statistics.mean(clean_values)
        if len(clean_values) == 1:
            return _format_number(mean_value)
        sd_value = statistics.stdev(clean_values)
        if abs(sd_value) < 1e-12:
            return _format_number(mean_value)
        return f"{_format_number(mean_value)} +/- {_format_number(sd_value)}"

    def labels_match(self, value: Any, label: str) -> bool:
        value_text = str(value).strip()
        label_text = str(label).strip()
        if value_text == label_text:
            return True
        value_float = _safe_float(value_text)
        label_float = _safe_float(label_text)
        return value_float is not None and label_float is not None and abs(value_float - label_float) < 1e-12

    def feature_summary_columns(self, data_process_summary: Optional[TableData]) -> List[str]:
        if data_process_summary and len(data_process_summary.columns) > 1:
            return ["Step"] + list(data_process_summary.columns[1:])
        return [
            "Step",
            "Instances",
            "Total Features",
            "Categorical Features",
            "Quantitative Features",
            "Missing Values",
            "Missing Percent",
        ]

    def cv_train_files(self, ds_dir: Path, *, pre_selection: bool) -> List[Path]:
        dataset_name = ds_dir.name
        cv_dir = ds_dir / "CVDatasets"
        if not cv_dir.is_dir():
            return []
        if pre_selection:
            return sorted(cv_dir.glob(f"{dataset_name}_CVPre_*_Train.csv"))
        return sorted(path for path in cv_dir.glob(f"{dataset_name}_CV_*_Train.csv") if "_CVPre_" not in path.name)

    def summarize_cv_train_files(
        self,
        files: Sequence[Path],
        metadata: Dict[str, Any],
        columns: Sequence[str],
    ) -> Dict[str, str]:
        if not files:
            return {}

        outcome_label = str(metadata.get("Outcome Label") or metadata.get("outcome_label") or self.outcome_label or "Class")
        instance_label = str(metadata.get("Instance Label") or metadata.get("instance_label") or self.instance_label or "InstanceID")
        class_columns = [col for col in columns if col.lower().startswith("class ")]
        numeric_values: Dict[str, List[float]] = {col: [] for col in columns if col != "Step"}

        missing_tokens = {"", "?", "na", "n/a", "nan", "none", "null"}
        for path in files:
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    fieldnames = list(reader.fieldnames or [])
                    rows = list(reader)
            except OSError:
                continue
            if not fieldnames:
                continue

            outcome_col = outcome_label if outcome_label in fieldnames else None
            if outcome_col is None:
                lowered = {col.lower(): col for col in fieldnames}
                outcome_col = lowered.get(outcome_label.lower()) or lowered.get("class")
            instance_col = instance_label if instance_label in fieldnames else None
            if instance_col is None:
                lowered = {col.lower(): col for col in fieldnames}
                instance_col = lowered.get(instance_label.lower()) or lowered.get("instanceid")

            excluded = {col for col in (outcome_col, instance_col) if col}
            feature_cols = [col for col in fieldnames if col not in excluded]
            instances = float(len(rows))
            total_features = float(len(feature_cols))
            missing_values = 0.0
            for row in rows:
                for feature in feature_cols:
                    if str(row.get(feature, "")).strip().lower() in missing_tokens:
                        missing_values += 1.0
            total_cells = instances * total_features

            if "Instances" in numeric_values:
                numeric_values["Instances"].append(instances)
            if "Total Features" in numeric_values:
                numeric_values["Total Features"].append(total_features)
            if "Missing Values" in numeric_values:
                numeric_values["Missing Values"].append(missing_values)
            if "Missing Percent" in numeric_values:
                numeric_values["Missing Percent"].append(missing_values / total_cells if total_cells else 0.0)

            if outcome_col:
                for class_col in class_columns:
                    label = class_col.split("Class ", 1)[1].strip()
                    count = sum(1 for row in rows if self.labels_match(row.get(outcome_col, ""), label))
                    numeric_values.setdefault(class_col, []).append(float(count))

        return {col: self.format_feature_summary_value(vals) for col, vals in numeric_values.items() if vals}

    def last_data_process_row(self, data_process_summary: Optional[TableData], columns: Sequence[str]) -> Dict[str, str]:
        if not data_process_summary or not data_process_summary.rows:
            return {}
        source = data_process_summary.rows[-1]
        row: Dict[str, str] = {"Step": "Post Processing"}
        for col in columns[1:]:
            row[col] = source.get(col, "")
        return row

    def read_feature_manifests(self, ds_dir: Path) -> List[Dict[str, Any]]:
        feature_dir = ds_dir / "feature_learning"
        if not feature_dir.is_dir():
            return []
        manifests: List[Dict[str, Any]] = []
        for path in sorted(feature_dir.glob("feature_manifest_cv*.json")):
            blob = self._read_json(path)
            if blob:
                manifests.append(blob)
        return manifests

    def feature_input_summary_from_manifests(
        self,
        manifests: Sequence[Dict[str, Any]],
        columns: Sequence[str],
        cv_context: Dict[str, str],
    ) -> Dict[str, str]:
        if not manifests:
            return {}
        input_counts: List[float] = []
        instance_counts: List[float] = []
        for manifest in manifests:
            input_count = _safe_float(manifest.get("input_feature_count"))
            if input_count is not None:
                input_counts.append(input_count)
            train_shape = manifest.get("train_shape")
            if isinstance(train_shape, (list, tuple)) and train_shape:
                instance_count = _safe_float(train_shape[0])
                if instance_count is not None:
                    instance_counts.append(instance_count)

        summary: Dict[str, str] = {}
        if "Instances" in columns:
            summary["Instances"] = self.format_feature_summary_value(instance_counts)
        if "Total Features" in columns:
            summary["Total Features"] = self.format_feature_summary_value(input_counts)
        for col in ("Missing Values", "Missing Percent"):
            if col in columns:
                summary[col] = cv_context.get(col, "0")
        for col in columns:
            if col.lower().startswith("class "):
                summary[col] = cv_context.get(col, "")
        return {key: value for key, value in summary.items() if value != ""}

    def feature_rows_from_manifests(
        self,
        manifests: Sequence[Dict[str, Any]],
        columns: Sequence[str],
        base_row: Dict[str, str],
        cv_context: Dict[str, str],
    ) -> List[Dict[str, str]]:
        if not manifests:
            return []

        rows: List[Dict[str, str]] = []
        final_counts: List[float] = []
        instance_counts: List[float] = []
        keep_original_values: List[bool] = []
        for manifest in manifests:
            final_count = _safe_float(manifest.get("final_feature_count"))
            if final_count is not None:
                final_counts.append(final_count)
            train_shape = manifest.get("train_shape")
            if isinstance(train_shape, (list, tuple)) and train_shape:
                instance_count = _safe_float(train_shape[0])
                if instance_count is not None:
                    instance_counts.append(instance_count)
            params = manifest.get("params")
            if isinstance(params, dict):
                keep_original_values.append(bool(params.get("keep_original_features", True)))

        categorical_base = _safe_float(base_row.get("Categorical Features", ""))
        treat_as_all_engineered = bool(keep_original_values) and not any(keep_original_values)
        categorical_values: List[float] = []
        quantitative_values: List[float] = []
        if final_counts:
            if treat_as_all_engineered:
                categorical_values = [0.0 for _ in final_counts]
                quantitative_values = list(final_counts)
            elif categorical_base is not None:
                categorical_values = [categorical_base for _ in final_counts]
                quantitative_values = [max(0.0, count - categorical_base) for count in final_counts]

        for step in ("P3 Feature Learning", "P4 Feature Importance"):
            row: Dict[str, str] = {col: "" for col in columns}
            row["Step"] = step
            if "Instances" in row:
                row["Instances"] = self.format_feature_summary_value(instance_counts)
            if "Total Features" in row:
                row["Total Features"] = self.format_feature_summary_value(final_counts)
            if "Categorical Features" in row:
                row["Categorical Features"] = self.format_feature_summary_value(categorical_values)
            if "Quantitative Features" in row:
                row["Quantitative Features"] = self.format_feature_summary_value(quantitative_values)
            for col in ("Missing Values", "Missing Percent"):
                if col in row:
                    row[col] = "0"
            for col in columns:
                if col.lower().startswith("class "):
                    row[col] = cv_context.get(col, "")
            rows.append(row)
        return rows

    def feature_selection_counts(self, ds_dir: Path) -> List[float]:
        table = self._read_csv_table(ds_dir / "feature_selection" / "InformativeFeatureSummary.csv")
        if not table or not table.rows:
            return []
        informative_col = None
        for col in table.columns:
            low = col.lower()
            if "informative" in low and "uninformative" not in low:
                informative_col = col
                break
        if informative_col is None:
            return []
        counts: List[float] = []
        for row in table.rows:
            value = _safe_float(row.get(informative_col, ""))
            if value is not None:
                counts.append(value)
        return counts

    def feature_learning_selection_cv_summary(
        self,
        ds_dir: Path,
        metadata: Dict[str, Any],
        data_process_summary: Optional[TableData],
    ) -> Optional[TableData]:
        columns = self.feature_summary_columns(data_process_summary)
        rows: List[Dict[str, str]] = []

        pre_cv_summary = self.summarize_cv_train_files(self.cv_train_files(ds_dir, pre_selection=True), metadata, columns)
        final_cv_summary = self.summarize_cv_train_files(self.cv_train_files(ds_dir, pre_selection=False), metadata, columns)
        manifests = self.read_feature_manifests(ds_dir)
        cv_context = pre_cv_summary or final_cv_summary

        base_row = self.last_data_process_row(data_process_summary, columns)
        input_summary = self.feature_input_summary_from_manifests(manifests, columns, cv_context)
        if base_row or input_summary:
            row = {col: "" for col in columns}
            row.update(base_row)
            row["Step"] = "Post Processing"
            for col in columns[1:]:
                if col in input_summary:
                    row[col] = input_summary[col]
            rows.append(row)

        rows.extend(self.feature_rows_from_manifests(manifests, columns, base_row, cv_context))

        selection_counts = self.feature_selection_counts(ds_dir)
        if selection_counts or final_cv_summary:
            row = {col: "" for col in columns}
            row["Step"] = "P5 Feature Selection"
            for col in columns[1:]:
                if col in final_cv_summary:
                    row[col] = final_cv_summary[col]
            if "Total Features" in row and selection_counts:
                row["Total Features"] = self.format_feature_summary_value(selection_counts)
            rows.append(row)

        if not rows:
            return None
        return TableData(columns=columns, rows=rows)

    def _load_metrics_by_cv(self, metrics_dir: Path, metric_name: str) -> Dict[str, List[float]]:
        key = METRIC_JSON_KEYS.get(metric_name, metric_name)
        out: Dict[str, List[float]] = {}
        if not metrics_dir.is_dir():
            return out
        for path in sorted(metrics_dir.glob("*.json")):
            blob = self._read_json(path)
            if not blob:
                continue
            alg = path.name.split("_CV_")[0]
            metrics = blob.get("metrics")
            val = None
            if isinstance(metrics, dict):
                if key in metrics:
                    val = _safe_float(metrics.get(key))
                else:
                    # Compatibility fallbacks
                    for alt in [key.lower(), key.upper(), metric_name, metric_name.lower()]:
                        if alt in metrics:
                            val = _safe_float(metrics.get(alt))
                            break
            elif key in blob:
                val = _safe_float(blob.get(key))
            if val is not None:
                out.setdefault(alg, []).append(val)
        return out

    def _plot_metric_distribution(self, metrics_dir: Path, metric_name: str, out: Path, title: str) -> Optional[str]:
        data = self._load_metrics_by_cv(metrics_dir, metric_name)
        if not data:
            return None
        labels = sorted(data.keys())
        series = [data[k] for k in labels]
        if self._mpl_ok():
            try:
                import matplotlib.pyplot as plt  # type: ignore

                out.parent.mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots(figsize=(8.6, 4.8))
                ax.boxplot(series, tick_labels=labels, vert=True, patch_artist=True)
                ax.set_ylabel(metric_name)
                for tick in ax.get_xticklabels():
                    tick.set_rotation(35)
                    tick.set_ha("right")
                fig.tight_layout()
                fig.savefig(out, dpi=180)
                plt.close(fig)
                return str(out)
            except Exception as exc:
                logger.warning("Metric distribution plot generation failed with matplotlib: %r", exc)
        return None

    def _extract_cv_index(self, filename: str) -> Optional[int]:
        m = re.search(r"_CV_([0-9]+)", filename)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _load_metric_values_by_cv(self, metrics_dir: Path, metric_name: str) -> Dict[int, List[float]]:
        key = METRIC_JSON_KEYS.get(metric_name, metric_name)
        out: Dict[int, List[float]] = {}
        if not metrics_dir.is_dir():
            return out
        for path in sorted(metrics_dir.glob("*.json")):
            blob = self._read_json(path)
            if not blob:
                continue
            metrics = blob.get("metrics")
            val = None
            if isinstance(metrics, dict):
                if key in metrics:
                    val = _safe_float(metrics.get(key))
                else:
                    for alt in [key.lower(), key.upper(), metric_name, metric_name.lower()]:
                        if alt in metrics:
                            val = _safe_float(metrics.get(alt))
                            break
            elif key in blob:
                val = _safe_float(blob.get(key))
            if val is None:
                continue
            cv_idx = self._extract_cv_index(path.stem)
            if cv_idx is None:
                cv_idx = len(out)
            out.setdefault(cv_idx, []).append(val)
        return out

    def _dataset_metric_series_for_overview(self, ds_dir: Path, metric_name: str) -> List[float]:
        by_cv: Dict[int, List[float]] = {}
        for metrics_dir in [
            ds_dir / "model_evaluation" / "metrics_by_cv",
            ds_dir / "ensemble_evaluation" / "metrics_by_cv",
        ]:
            cv_map = self._load_metric_values_by_cv(metrics_dir, metric_name)
            for cv_idx, values in cv_map.items():
                by_cv.setdefault(cv_idx, []).extend(values)
        if not by_cv:
            return []

        higher = METRIC_DIRECTION_HIGHER_IS_BETTER.get(metric_name, True)
        series: List[float] = []
        for cv_idx in sorted(by_cv.keys()):
            values = [v for v in by_cv[cv_idx] if v is not None]
            if not values:
                continue
            series.append(max(values) if higher else min(values))
        return series

    def _plot_dataset_comparison_overview(
        self,
        dataset_blocks: Sequence[Dict[str, Any]],
        metric_name: str,
        out: Path,
    ) -> Optional[str]:
        labels: List[str] = []
        series: List[List[float]] = []
        for ds in dataset_blocks:
            ds_id = str(ds.get("dataset_id", ""))
            ds_path = Path(str(ds.get("dataset_path", "")))
            if not ds_id or not ds_path.is_dir():
                continue
            vals = self._dataset_metric_series_for_overview(ds_path, metric_name)
            if vals:
                labels.append(ds_id)
                series.append(vals)

        if not labels or not series:
            return None

        if self._mpl_ok():
            try:
                import matplotlib.pyplot as plt  # type: ignore

                out.parent.mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots(figsize=(8.6, 4.8))
                ax.boxplot(series, tick_labels=labels, vert=True, patch_artist=True)
                ax.set_xlabel("Dataset")
                ax.set_ylabel(metric_name)
                fig.tight_layout()
                fig.savefig(out, dpi=180)
                plt.close(fig)
                return str(out)
            except Exception as exc:
                logger.warning(
                    "Dataset comparison overview generation failed with matplotlib for %s: %r",
                    metric_name,
                    exc,
                )
        return None

    def _extract_curve_xy(self, blob: Dict[str, Any], curve_kind: str) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[str]]:
        if curve_kind == "roc":
            keys = ("fpr", "tpr")
        else:
            keys = ("recall", "precision")

        if keys[0] in blob and keys[1] in blob:
            try:
                x = [float(v) for v in blob[keys[0]]]
                y = [float(v) for v in blob[keys[1]]]
                return x, y, None
            except Exception:
                pass

        preferred = ["micro", "macro"]
        for key in preferred + list(blob.keys()):
            sub = blob.get(key)
            if isinstance(sub, dict) and keys[0] in sub and keys[1] in sub:
                try:
                    x = [float(v) for v in sub[keys[0]]]
                    y = [float(v) for v in sub[keys[1]]]
                    return x, y, str(key)
                except Exception:
                    continue
        return None, None, None

    def _curve_alg_from_filename(self, filename: str, curve_kind: str) -> str:
        # Parse algorithm name using the *last* _CV_<idx>_<kind>.json pattern.
        # This avoids collapsing names that may themselves contain "_CV_".
        pat = re.compile(rf"^(.*)_CV_[0-9]+_{re.escape(curve_kind)}\.json$", re.IGNORECASE)
        m = pat.match(filename)
        if m:
            return m.group(1)
        suffix = f"_{curve_kind}.json"
        if filename.lower().endswith(suffix):
            return filename[: -len(suffix)]
        return Path(filename).stem

    def _is_class_like_key(self, key: str) -> bool:
        k = str(key).strip()
        if k == "":
            return False
        if k.isdigit():
            return True
        try:
            float(k)
            return True
        except Exception:
            return False

    def _extract_curve_entries(
        self,
        blob: Dict[str, Any],
        curve_kind: str,
        default_alg: str,
    ) -> List[Tuple[str, List[float], List[float], Optional[str]]]:
        # 1) Direct or micro/macro-compatible payload.
        x, y, tag = self._extract_curve_xy(blob, curve_kind)
        if x and y and len(x) == len(y) and len(x) > 1:
            return [(default_alg, x, y, tag)]

        # 2) One-level nested payload.
        one_level: List[Tuple[str, List[float], List[float], Optional[str]]] = []
        for key, sub in blob.items():
            if not isinstance(sub, dict):
                continue
            sx, sy, stag = self._extract_curve_xy(sub, curve_kind)
            if sx and sy and len(sx) == len(sy) and len(sx) > 1:
                one_level.append((str(key), sx, sy, stag))

        if one_level:
            keys = [k for k, _, _, _ in one_level]
            if all(self._is_class_like_key(k) for k in keys):
                # Class-wise nested curves for one algorithm.
                # Aggregate to a single representative (macro-like) curve.
                x_grid = _linspace(0.0, 1.0, 250)
                interpolated: List[List[float]] = []
                for _, sx, sy, _ in one_level:
                    yi = _interp_sorted(sx, sy, x_grid)
                    yi = [max(0.0, min(1.0, v)) for v in yi]
                    interpolated.append(yi)
                if interpolated:
                    mean_y = [
                        sum(vals[i] for vals in interpolated) / float(len(interpolated))
                        for i in range(len(x_grid))
                    ]
                    return [(default_alg, x_grid, mean_y, "macro")]
                return []

            # Treat keys as algorithm names in combined files.
            entries: List[Tuple[str, List[float], List[float], Optional[str]]] = []
            for key, sx, sy, stag in one_level:
                alg = key.strip() or default_alg
                entries.append((alg, sx, sy, stag))
            return entries

        # 3) Two-level nested payload (algorithm -> class -> curve)
        entries: List[Tuple[str, List[float], List[float], Optional[str]]] = []
        for outer_key, outer_val in blob.items():
            if not isinstance(outer_val, dict):
                continue
            nested: List[Tuple[str, List[float], List[float], Optional[str]]] = []
            for inner_key, inner_val in outer_val.items():
                if not isinstance(inner_val, dict):
                    continue
                sx, sy, stag = self._extract_curve_xy(inner_val, curve_kind)
                if sx and sy and len(sx) == len(sy) and len(sx) > 1:
                    nested.append((str(inner_key), sx, sy, stag))
            if not nested:
                continue

            inner_keys = [k for k, _, _, _ in nested]
            alg_name = str(outer_key).strip() or default_alg
            if all(self._is_class_like_key(k) for k in inner_keys):
                x_grid = _linspace(0.0, 1.0, 250)
                interpolated: List[List[float]] = []
                for _, sx, sy, _ in nested:
                    yi = _interp_sorted(sx, sy, x_grid)
                    yi = [max(0.0, min(1.0, v)) for v in yi]
                    interpolated.append(yi)
                if interpolated:
                    mean_y = [
                        sum(vals[i] for vals in interpolated) / float(len(interpolated))
                        for i in range(len(x_grid))
                    ]
                    entries.append((alg_name, x_grid, mean_y, "macro"))
            else:
                for inner_name, sx, sy, stag in nested:
                    name = inner_name.strip() or alg_name
                    entries.append((name, sx, sy, stag))
        return entries

    def _load_curves_grouped(self, curves_dir: Path, curve_kind: str) -> Dict[str, List[Tuple[List[float], List[float], Optional[str]]]]:
        groups: Dict[str, List[Tuple[List[float], List[float], Optional[str]]]] = {}
        if not curves_dir.is_dir():
            return groups
        for path in sorted(curves_dir.glob(f"*_{curve_kind}.json")):
            blob = self._read_json(path)
            if not blob:
                continue
            default_alg = self._curve_alg_from_filename(path.name, curve_kind)
            entries = self._extract_curve_entries(blob, curve_kind, default_alg)
            for alg, x, y, tag in entries:
                if x and y and len(x) == len(y) and len(x) > 1:
                    groups.setdefault(alg, []).append((x, y, tag))
        return groups

    def _plot_curve_summary(
        self,
        curves_dir: Path,
        out: Path,
        *,
        curve_kind: str,
        title: str,
        no_skill: float = 0.5,
    ) -> Optional[str]:
        groups = self._load_curves_grouped(curves_dir, curve_kind)
        if not groups:
            return None
        x_grid = _linspace(0.0, 1.0, 250)
        plot_lines: List[Tuple[str, List[float], Optional[str], float]] = []
        for alg in sorted(groups.keys()):
            curves = groups[alg]
            interpolated: List[List[float]] = []
            tags: List[str] = []
            for x, y, tag in curves:
                yi = _interp_sorted(x, y, x_grid)
                yi = [max(0.0, min(1.0, v)) for v in yi]
                interpolated.append(yi)
                if tag:
                    tags.append(tag)
            if not interpolated:
                continue
            mean_y = [
                sum(vals[i] for vals in interpolated) / float(len(interpolated))
                for i in range(len(x_grid))
            ]
            auc_val = _auc_trapezoid(x_grid, mean_y)
            label_suffix = ""
            if tags:
                if "micro" in tags:
                    label_suffix = " (micro)"
                elif "macro" in tags:
                    label_suffix = " (macro)"
            label = f"{alg}{label_suffix} (AUC={_format_number(auc_val)})"
            plot_lines.append((label, mean_y, label_suffix or None, auc_val))

        if not plot_lines:
            return None

        if self._mpl_ok():
            try:
                import matplotlib.pyplot as plt  # type: ignore

                out.parent.mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots(figsize=(7.6, 5.4))
                for label, mean_y, _tag, _auc in plot_lines:
                    ax.plot(x_grid, mean_y, linewidth=1.6, label=label)

                if curve_kind == "roc":
                    ax.plot(
                        [0.0, 1.0],
                        [0.0, 1.0],
                        linestyle="--",
                        color="black",
                        linewidth=1.0,
                        label="No Skill (AUROC=0.500)",
                    )
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                else:
                    y_base = max(0.0, min(1.0, no_skill))
                    ax.plot(
                        [0.0, 1.0],
                        [y_base, y_base],
                        linestyle="--",
                        color="black",
                        linewidth=1.0,
                        label=f"No Skill (AUPRC={_format_number(y_base)})",
                    )
                    ax.set_xlabel("Recall")
                    ax.set_ylabel("Precision")
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.02)
                ax.legend(loc="lower right", fontsize=8)
                fig.tight_layout()
                fig.savefig(out, dpi=180)
                plt.close(fig)
                return str(out)
            except Exception as exc:
                logger.warning("Curve summary generation failed with matplotlib (%s): %r", curve_kind, exc)
        return None

    def _plot_regression_residual_fallbacks(self, ds_dir: Path, dataset_name: str) -> Dict[str, Optional[str]]:
        out: Dict[str, Optional[str]] = {
            "actual_vs_predicted": None,
            "residual_distribution": None,
            "test_residual": None,
        }
        test = self._read_csv_table(ds_dir / "model_evaluation" / "residual_test.csv")
        if not test:
            return out

        def residuals_from(table: Optional[TableData]) -> List[float]:
            if not table:
                return []
            low = {c.lower(): c for c in table.columns}
            rcol = low.get("residual") or (table.columns[1] if len(table.columns) > 1 else table.columns[0])
            vals: List[float] = []
            for row in table.rows:
                rv = _safe_float(row.get(rcol, ""))
                if rv is not None:
                    vals.append(rv)
            return vals

        test_vals = residuals_from(test)

        def actual_pred_from(table: Optional[TableData]) -> Tuple[List[float], List[float], List[str]]:
            if not table:
                return [], [], []
            low = {c.lower(): c for c in table.columns}
            pred_col = low.get("predicted") or low.get("prediction") or low.get("pred") or low.get("y_pred")
            actual_col = low.get("actual") or low.get("outcome") or low.get("observed") or low.get("y_test")
            alg_col = low.get("algorithm") or low.get("model")
            if not pred_col or not actual_col:
                return [], [], []
            preds: List[float] = []
            actuals: List[float] = []
            algs: List[str] = []
            for row in table.rows:
                pv = _safe_float(row.get(pred_col, ""))
                av = _safe_float(row.get(actual_col, ""))
                if pv is None or av is None:
                    continue
                preds.append(pv)
                actuals.append(av)
                algs.append(row.get(alg_col, "").strip() if alg_col else "")
            return preds, actuals, algs

        preds, actuals, pred_algs = actual_pred_from(test)
        if self._mpl_ok():
            try:
                import matplotlib.pyplot as plt  # type: ignore

                if test_vals:
                    dist_out = self.paths.figures_dir / f"{dataset_name}_residual_distribution_fallback.png"
                    fig, ax = plt.subplots(figsize=(7.2, 4.2))
                    ax.hist(test_vals, bins=35, alpha=0.85, color="#E15759")
                    ax.set_xlabel("Residual")
                    ax.set_ylabel("Frequency")
                    fig.tight_layout()
                    fig.savefig(dist_out, dpi=180)
                    plt.close(fig)
                    out["residual_distribution"] = str(dist_out)

                if test_vals:
                    test_out = self.paths.figures_dir / f"{dataset_name}_test_residual_fallback.png"
                    fig, ax = plt.subplots(figsize=(7.2, 4.2))
                    vals_sorted = sorted(test_vals)
                    n = len(vals_sorted)
                    if n > 1:
                        theo = [statistics.NormalDist().inv_cdf((i + 0.5) / n) for i in range(n)]
                        ax.scatter(theo, vals_sorted, s=8, alpha=0.6, color="#E15759")
                    else:
                        ax.scatter([0.0], vals_sorted, s=8, alpha=0.6, color="#E15759")
                    ax.set_xlabel("Theoretical Quantiles")
                    ax.set_ylabel("Ordered Residual")
                    fig.tight_layout()
                    fig.savefig(test_out, dpi=180)
                    plt.close(fig)
                    out["test_residual"] = str(test_out)

                if preds and actuals:
                    avp_out = self.paths.figures_dir / f"{dataset_name}_actual_vs_predicted_fallback.png"
                    fig, ax = plt.subplots(figsize=(7.2, 4.2))
                    if pred_algs and any(a for a in pred_algs):
                        for alg in sorted(set([a for a in pred_algs if a])):
                            idxs = [i for i, a in enumerate(pred_algs) if a == alg]
                            xvals = [preds[i] for i in idxs]
                            yvals = [actuals[i] for i in idxs]
                            ax.scatter(xvals, yvals, s=10, alpha=0.45, label=alg)
                        ax.legend(loc="upper right", fontsize=7)
                    else:
                        ax.scatter(preds, actuals, s=10, alpha=0.45)
                    ax.set_xlabel("Predicted Outcome")
                    ax.set_ylabel("Actual Outcome")
                    fig.tight_layout()
                    fig.savefig(avp_out, dpi=180)
                    plt.close(fig)
                    out["actual_vs_predicted"] = str(avp_out)
            except Exception as exc:
                logger.warning("Regression fallback plot generation failed with matplotlib for %s: %r", ds_dir, exc)

        if out["actual_vs_predicted"] is None:
            out["actual_vs_predicted"] = self._save_simple_placeholder(
                self.paths.figures_dir / f"{dataset_name}_actual_vs_predicted_fallback.png",
                "Actual vs Predicted",
                "Predictions not found in exported CSVs.",
            )
        return out

    def _format_rows_for_display(self, table: Optional[TableData]) -> Optional[TableData]:
        if not table:
            return None
        rows: List[Dict[str, str]] = []
        for row in table.rows:
            clean: Dict[str, str] = {}
            for col in table.columns:
                val = row.get(col, "")
                fv = _safe_float(val)
                if fv is not None:
                    clean[col] = _format_number(fv, is_pvalue=_is_pvalue_col(col))
                else:
                    clean[col] = val
            rows.append(clean)
        return TableData(columns=table.columns[:], rows=rows)

    def _univariate_top10(self, table: Optional[TableData]) -> Optional[TableData]:
        if not table or not table.rows:
            return None
        low = {c.lower(): c for c in table.columns}
        pcol = low.get("p-value") or low.get("p_value") or "p-value"
        stat_col = low.get("test-statistic") or low.get("test_statistic")

        rows = table.rows[:]
        rows.sort(key=lambda r: (_safe_float(r.get(pcol, "")) if _safe_float(r.get(pcol, "")) is not None else 999.0))
        top = rows[:10]

        formatted: List[Dict[str, str]] = []
        for row in top:
            out = {}
            for col in table.columns:
                val = row.get(col, "")
                if col == pcol:
                    out[col] = _format_number(val, is_pvalue=True)
                elif stat_col and col == stat_col:
                    out[col] = _format_number(val)
                else:
                    fv = _safe_float(val)
                    out[col] = _format_number(fv) if fv is not None else val
            formatted.append(out)
        return TableData(columns=table.columns[:], rows=formatted)

    def _combine_perf_tables(
        self,
        task_type: str,
        models_mean: Optional[TableData],
        models_std: Optional[TableData],
        models_median: Optional[TableData],
        ens_mean: Optional[TableData],
        ens_std: Optional[TableData],
        ens_median: Optional[TableData],
    ) -> Dict[str, Any]:
        metrics = self._metric_list_for_task(task_type)

        def as_map(table: Optional[TableData], alg_label_hint: str) -> Tuple[str, Dict[str, Dict[str, str]]]:
            if not table:
                return alg_label_hint, {}
            alg_col = self._find_algorithm_col(table.columns)
            mapping: Dict[str, Dict[str, str]] = {}
            for row in table.rows:
                alg = row.get(alg_col, "").strip()
                if alg:
                    mapping[alg] = row
            return alg_col, mapping

        _, m_map = as_map(models_mean, "Algorithm")
        _, s_map = as_map(models_std, "Algorithm")
        _, md_map = as_map(models_median, "Algorithm")

        _, em_map = as_map(ens_mean, "Ensemble")
        _, es_map = as_map(ens_std, "Ensemble")
        _, ed_map = as_map(ens_median, "Ensemble")

        ordered_algs = list(m_map.keys())
        ordered_ens = list(em_map.keys())

        available_metrics: List[str] = []
        for metric in metrics:
            present = False
            for row in list(m_map.values()) + list(em_map.values()):
                if metric in row:
                    present = True
                    break
            if present:
                available_metrics.append(metric)

        # Build mean/std combined rows
        mean_rows: List[List[str]] = []
        raw_means: Dict[str, Dict[str, float]] = {}
        for alg in ordered_algs:
            row = [alg]
            raw_means[alg] = {}
            for metric in available_metrics:
                mval = _safe_float(m_map.get(alg, {}).get(metric, ""))
                sval = _safe_float(s_map.get(alg, {}).get(metric, ""))
                if mval is None:
                    row.append("")
                    continue
                raw_means[alg][metric] = mval
                if sval is None:
                    row.append(_format_number(mval))
                else:
                    row.append(f"{_format_number(mval)} +/- {_format_number(sval)}")
            mean_rows.append(row)

        for ens in ordered_ens:
            label = ens if ens.endswith("- Ensemble") else f"{ens} - Ensemble"
            row = [label]
            raw_means[label] = {}
            for metric in available_metrics:
                mval = _safe_float(em_map.get(ens, {}).get(metric, ""))
                sval = _safe_float(es_map.get(ens, {}).get(metric, ""))
                if mval is None:
                    row.append("")
                    continue
                raw_means[label][metric] = mval
                if sval is None:
                    row.append(_format_number(mval))
                else:
                    row.append(f"{_format_number(mval)} +/- {_format_number(sval)}")
            mean_rows.append(row)

        mean_columns = ["Algorithm"] + available_metrics

        # Highlight best mean per metric with tie handling at 3 decimals.
        mean_highlight_cells: Set[Tuple[int, int]] = set()
        for c_idx, metric in enumerate(available_metrics, start=1):
            scored: List[Tuple[int, float]] = []
            for r_idx, row in enumerate(mean_rows, start=1):
                raw_name = row[0]
                val = raw_means.get(raw_name, {}).get(metric)
                if val is not None:
                    scored.append((r_idx, round(val, 3)))
            if not scored:
                continue
            higher = METRIC_DIRECTION_HIGHER_IS_BETTER.get(metric, True)
            best_val = max(v for _, v in scored) if higher else min(v for _, v in scored)
            for r_idx, v in scored:
                if v == best_val:
                    mean_highlight_cells.add((r_idx, c_idx))

        # Combined median rows.
        median_rows: List[List[str]] = []
        median_raw: Dict[str, Dict[str, float]] = {}
        for alg in ordered_algs:
            row = [alg]
            median_raw[alg] = {}
            for metric in available_metrics:
                val = _safe_float(md_map.get(alg, {}).get(metric, ""))
                if val is None:
                    row.append("")
                else:
                    row.append(_format_number(val))
                    median_raw[alg][metric] = val
            median_rows.append(row)
        for ens in ordered_ens:
            label = ens if ens.endswith("- Ensemble") else f"{ens} - Ensemble"
            row = [label]
            median_raw[label] = {}
            for metric in available_metrics:
                val = _safe_float(ed_map.get(ens, {}).get(metric, ""))
                if val is None:
                    row.append("")
                else:
                    row.append(_format_number(val))
                    median_raw[label][metric] = val
            median_rows.append(row)

        median_columns = ["Algorithm"] + available_metrics
        median_highlight_cells: Set[Tuple[int, int]] = set()
        for c_idx, metric in enumerate(available_metrics, start=1):
            scored: List[Tuple[int, float]] = []
            for r_idx, row in enumerate(median_rows, start=1):
                raw_name = row[0]
                val = median_raw.get(raw_name, {}).get(metric)
                if val is not None:
                    scored.append((r_idx, round(val, 3)))
            if not scored:
                continue
            higher = METRIC_DIRECTION_HIGHER_IS_BETTER.get(metric, True)
            best_val = max(v for _, v in scored) if higher else min(v for _, v in scored)
            for r_idx, v in scored:
                if v == best_val:
                    median_highlight_cells.add((r_idx, c_idx))

        return {
            "mean_columns": mean_columns,
            "mean_rows": mean_rows,
            "mean_highlight_cells": [(r, c) for r, c in sorted(mean_highlight_cells)],
            "mean_bold_cells": [(r, c) for r, c in sorted(mean_highlight_cells)],
            "median_columns": median_columns,
            "median_rows": median_rows,
            "median_highlight_cells": [(r, c) for r, c in sorted(median_highlight_cells)],
            "median_bold_cells": [(r, c) for r, c in sorted(median_highlight_cells)],
        }

    def _resolve_dataset_images(
        self,
        ds_dir: Path,
        dataset_name: str,
        task_type: str,
        metadata: Dict[str, Any],
        class_counts: Optional[TableData],
        missingness_table: Optional[TableData],
        perf_metric_default: Optional[str],
    ) -> Dict[str, Any]:
        figs: Dict[str, Any] = {}

        # Missingness top 25
        figs["missingness_top25"] = None
        reused_missing = self._reuse_generated_report_figure(
            f"{dataset_name}_missingness_top25.png",
            ds_dir=ds_dir,
        ) if self.reuse_existing_figures else None
        existing_missing = _first_existing(
            [
                ds_dir / "exploratory" / "DataMissingness.png",
                ds_dir / "exploratory" / "Missingness.png",
                ds_dir / "exploratory" / "MissingnessTop25.png",
                reused_missing if reused_missing is not None else (self.paths.figures_dir / f"{dataset_name}_missingness_top25.png"),
            ]
        )
        if self.reuse_existing_figures and existing_missing:
            figs["missingness_top25"] = str(existing_missing)
        elif self.enable_plots:
            out = self.paths.figures_dir / f"{dataset_name}_missingness_top25.png"
            figs["missingness_top25"] = self._plot_missingness_top25(
                missingness_table, out, f"{dataset_name}: Missingness Top 25 Features"
            )

        # Correlation matrix (prefer existing exploratory PNG; generate from CSV if missing)
        figs["correlation_matrix"] = None
        corr_png = self._figure_path_exploratory_correlation_matrix(ds_dir)
        if self.reuse_existing_figures and corr_png is not None:
            figs["correlation_matrix"] = str(corr_png)
        else:
            corr_csv = self._find_exploratory_correlation_csv(ds_dir)
            if corr_csv is not None:
                figs["correlation_matrix"] = self._plot_correlation_matrix_from_csv(
                    corr_csv,
                    self.paths.figures_dir / f"{dataset_name}_corr_matrix.png",
                    f"{dataset_name}: Feature Correlation Matrix",
                )
            if figs["correlation_matrix"] is None and corr_png is not None:
                figs["correlation_matrix"] = str(corr_png)

        # Class balance or target distribution
        if task_type == "Regression":
            out_path = self.paths.figures_dir / f"{dataset_name}_target_distribution.png"
            # Prefer a freshly generated histogram for regression target distribution.
            if self.enable_plots:
                figs["target_distribution"] = self._plot_target_distribution(
                    ds_dir,
                    metadata,
                    out_path,
                    f"{dataset_name}: Target Distribution Histogram",
                )
            else:
                figs["target_distribution"] = None
            if figs["target_distribution"] is None:
                reused_target = self._reuse_generated_report_figure(
                    f"{dataset_name}_target_distribution.png",
                    ds_dir=ds_dir,
                ) if self.reuse_existing_figures else None
                existing = _first_existing(
                    [
                        reused_target if reused_target is not None else out_path,
                        out_path,
                        ds_dir / "exploratory" / "TargetDistribution.png",
                        ds_dir / "exploratory" / "target_distribution.png",
                    ]
                )
                figs["target_distribution"] = str(existing) if existing else None
            figs["class_balance"] = None
        else:
            reused_class = self._reuse_generated_report_figure(
                f"{dataset_name}_class_balance.png",
                ds_dir=ds_dir,
            ) if self.reuse_existing_figures else None
            existing = _first_existing(
                [
                    ds_dir / "exploratory" / "ClassCountsBarPlot.png",
                    ds_dir / "exploratory" / "ClassCountsBarplot.png",
                    ds_dir / "exploratory" / "ClassCounts.png",
                    reused_class if reused_class is not None else (self.paths.figures_dir / f"{dataset_name}_class_balance.png"),
                ]
            )
            if self.reuse_existing_figures and existing:
                figs["class_balance"] = str(existing)
            else:
                figs["class_balance"] = self._plot_class_balance(
                    class_counts,
                    self.paths.figures_dir / f"{dataset_name}_class_balance.png",
                    f"{dataset_name}: Class Balance",
                )
            figs["target_distribution"] = None

        # Feature learning is omitted from replication-mode reports.
        if self.report_mode == "replication":
            figs["feature_learning_panels"] = []
            figs["mutual_info"] = None
        else:
            # Feature learning FI methods: deterministic report outputs per method.
            fi_panels = self._resolve_feature_learning_panels(ds_dir, dataset_name)
            figs["feature_learning_panels"] = fi_panels
            figs["mutual_info"] = None
            for panel in fi_panels:
                if panel.get("key") == "mutual_info":
                    figs["mutual_info"] = panel.get("path")
                    break

        # Performance distribution
        figs["performance_distribution"] = None
        if perf_metric_default:
            perf_fig_name = f"{dataset_name}_distribution_{perf_metric_default.replace(' ', '_')}.png"
            reused_perf = self._reuse_generated_report_figure(perf_fig_name, ds_dir=ds_dir) \
                if self.reuse_existing_figures else None
            existing_box = _first_existing(
                [
                    ds_dir / "model_evaluation" / "metricBoxplots" / f"Compare_{perf_metric_default}.png",
                    reused_perf if reused_perf is not None else (self.paths.figures_dir / perf_fig_name),
                ]
            )
            if existing_box and self.reuse_existing_figures:
                figs["performance_distribution"] = str(existing_box)
            else:
                figs["performance_distribution"] = self._plot_metric_distribution(
                    ds_dir / "model_evaluation" / "metrics_by_cv",
                    perf_metric_default,
                    self.paths.figures_dir / perf_fig_name,
                    f"{dataset_name}: {perf_metric_default} Distribution",
                )

        # Classification curve pages
        if task_type != "Regression":
            # Model curves: reuse summary if present, else generate.
            mroc = _first_existing([ds_dir / "model_evaluation" / "Summary_ROC.png"])
            mprc = _first_existing([ds_dir / "model_evaluation" / "Summary_PRC.png"])
            outcome_label = str(
                self.outcome_label
                or metadata.get("Outcome Label")
                or metadata.get("outcome_label")
                or "Class"
            )
            no_skill = self._classification_no_skill(
                class_counts,
                ds_dir=ds_dir,
                outcome_label=outcome_label,
                task_type=task_type,
            )

            reused_models_roc = self._reuse_generated_report_figure(
                f"{dataset_name}_models_roc_summary.png",
                ds_dir=ds_dir,
            ) if self.reuse_existing_figures else None
            if self.reuse_existing_figures and (mroc or reused_models_roc):
                figs["models_roc"] = str(mroc if mroc is not None else reused_models_roc)
            else:
                figs["models_roc"] = self._plot_curve_summary(
                    ds_dir / "model_evaluation" / "curves_by_cv",
                    self.paths.figures_dir / f"{dataset_name}_models_roc_summary.png",
                    curve_kind="roc",
                    title=f"{dataset_name}: Summary ROC (Models)",
                    no_skill=no_skill,
                )
                if figs["models_roc"] is None and mroc:
                    figs["models_roc"] = str(mroc)

            reused_models_prc = self._reuse_generated_report_figure(
                f"{dataset_name}_models_prc_summary.png",
                ds_dir=ds_dir,
            ) if self.reuse_existing_figures else None
            if self.reuse_existing_figures and (mprc or reused_models_prc):
                figs["models_prc"] = str(mprc if mprc is not None else reused_models_prc)
            else:
                figs["models_prc"] = self._plot_curve_summary(
                    ds_dir / "model_evaluation" / "curves_by_cv",
                    self.paths.figures_dir / f"{dataset_name}_models_prc_summary.png",
                    curve_kind="prc",
                    title=f"{dataset_name}: Summary PRC (Models)",
                    no_skill=no_skill,
                )
                if figs["models_prc"] is None and mprc:
                    figs["models_prc"] = str(mprc)

            # Ensemble curves: prefer generating even if png exists.
            figs["ensembles_roc"] = self._plot_curve_summary(
                ds_dir / "ensemble_evaluation" / "curves_by_cv",
                self.paths.figures_dir / f"{dataset_name}_ensembles_roc_summary.png",
                curve_kind="roc",
                title=f"{dataset_name}: Summary ROC (Ensembles)",
                no_skill=no_skill,
            )
            figs["ensembles_prc"] = self._plot_curve_summary(
                ds_dir / "ensemble_evaluation" / "curves_by_cv",
                self.paths.figures_dir / f"{dataset_name}_ensembles_prc_summary.png",
                curve_kind="prc",
                title=f"{dataset_name}: Summary PRC (Ensembles)",
                no_skill=no_skill,
            )
            if figs["ensembles_roc"] is None:
                reused_ens_roc = self._reuse_generated_report_figure(
                    f"{dataset_name}_ensembles_roc_summary.png",
                    ds_dir=ds_dir,
                ) if self.reuse_existing_figures else None
                fallback = _first_existing([
                    ds_dir / "ensemble_evaluation" / "Summary_ROC_ensembles.png",
                    reused_ens_roc if reused_ens_roc is not None else (self.paths.figures_dir / f"{dataset_name}_ensembles_roc_summary.png"),
                ])
                figs["ensembles_roc"] = str(fallback) if fallback else None
            if figs["ensembles_prc"] is None:
                reused_ens_prc = self._reuse_generated_report_figure(
                    f"{dataset_name}_ensembles_prc_summary.png",
                    ds_dir=ds_dir,
                ) if self.reuse_existing_figures else None
                fallback = _first_existing([
                    ds_dir / "ensemble_evaluation" / "Summary_PRC_ensembles.png",
                    reused_ens_prc if reused_ens_prc is not None else (self.paths.figures_dir / f"{dataset_name}_ensembles_prc_summary.png"),
                ])
                figs["ensembles_prc"] = str(fallback) if fallback else None
        else:
            # Regression eval plots
            figs["reg_actual_vs_pred"] = None
            figs["reg_residual_dist"] = None
            figs["reg_test_resid"] = None
            eval_dir = ds_dir / "model_evaluation" / "evalPlots"
            figs["reg_actual_vs_pred"] = str(eval_dir / "actual_vs_predict_all_algorithms.png") if (eval_dir / "actual_vs_predict_all_algorithms.png").is_file() else None
            figs["reg_residual_dist"] = str(eval_dir / "residual_distrib_all_algorithms.png") if (eval_dir / "residual_distrib_all_algorithms.png").is_file() else None
            figs["reg_test_resid"] = str(eval_dir / "probability_test_residual_all_algorithms.png") if (eval_dir / "probability_test_residual_all_algorithms.png").is_file() else None
            if not all([figs["reg_actual_vs_pred"], figs["reg_residual_dist"], figs["reg_test_resid"]]):
                fb = self._plot_regression_residual_fallbacks(ds_dir, dataset_name)
                if figs["reg_actual_vs_pred"] is None:
                    figs["reg_actual_vs_pred"] = fb.get("actual_vs_predicted")
                if figs["reg_residual_dist"] is None:
                    figs["reg_residual_dist"] = fb.get("residual_distribution")
                if figs["reg_test_resid"] is None:
                    figs["reg_test_resid"] = fb.get("test_residual")

        # Composite feature score plot: reuse only, never auto-generate.
        composite = _first_existing([ds_dir / "model_evaluation" / "feature_importance" / "Compare_FI_Norm_Weight.png"])
        figs["composite_feature_scores"] = str(composite) if composite else None

        return figs

    def _format_runtime_table(self, runtimes: Optional[TableData]) -> Optional[TableData]:
        if not runtimes or not runtimes.rows:
            return None
        cols = runtimes.columns[:]
        phase_col = None
        for c in cols:
            if c.strip().lower() == "phase":
                phase_col = c
                break
        if phase_col is None:
            return self._format_rows_for_display(runtimes)
        rows = runtimes.rows[:]
        rows.sort(key=lambda r: (_safe_float(r.get(phase_col, "")) if _safe_float(r.get(phase_col, "")) is not None else 999.0))

        formatted: List[Dict[str, str]] = []
        for row in rows:
            out: Dict[str, str] = {}
            for c in cols:
                if c == phase_col:
                    phase_raw = row.get(c, "")
                    phase_key = str(int(_safe_float(phase_raw))) if _safe_float(phase_raw) is not None else str(phase_raw)
                    label = PHASE_LABELS.get(phase_key)
                    out[c] = f"{phase_key} - {label}" if label else phase_key
                elif _is_numeric_text(row.get(c, "")):
                    out[c] = _format_number(row.get(c, ""))
                else:
                    out[c] = row.get(c, "")
            formatted.append(out)
        return TableData(columns=cols, rows=formatted)

    def _collect_dataset_block(
        self,
        ds_dir: Path,
        dataset_id: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        dataset_name = ds_dir.name
        task_type = self._detect_task_type(ds_dir, metadata)

        explore = ds_dir / "exploratory"
        univariate = self._read_csv_table(explore / "univariate_analyses" / "Univariate_Significance.csv")
        univariate_top10 = self._univariate_top10(univariate)
        class_counts = self._read_csv_table(explore / "ClassCounts.csv")
        missingness_table = self._format_rows_for_display(self._read_csv_table(explore / "DataMissingness.csv"))
        informative_summary = self._format_rows_for_display(self._read_csv_table(ds_dir / "feature_selection" / "InformativeFeatureSummary.csv"))
        data_process_summary = self._format_rows_for_display(self._read_csv_table(explore / "DataProcessSummary.csv"))
        if data_process_summary and data_process_summary.columns and data_process_summary.columns[0] == "Algorithm":
            old_col = data_process_summary.columns[0]
            data_process_summary.columns[0] = "Step"
            for row in data_process_summary.rows:
                row["Step"] = row.pop(old_col, "")
        feature_cv_summary = self.feature_learning_selection_cv_summary(ds_dir, metadata, data_process_summary)
        runtimes = self._format_runtime_table(self._read_csv_table(ds_dir / "runtimes.csv"))

        model_eval = ds_dir / "model_evaluation"
        summary_mean = self._read_csv_table(model_eval / "Summary_performance_mean.csv")
        summary_std = self._read_csv_table(model_eval / "Summary_performance_std.csv")
        summary_median = self._read_csv_table(model_eval / "Summary_performance_median.csv")

        ens_eval = ds_dir / "ensemble_evaluation"
        ens_mean = self._read_csv_table(ens_eval / "Ensembles_performance_mean.csv")
        ens_std = self._read_csv_table(ens_eval / "Ensembles_performance_std.csv")
        ens_median = self._read_csv_table(ens_eval / "Ensembles_performance_median.csv")

        perf_combined = self._combine_perf_tables(
            task_type=task_type,
            models_mean=summary_mean,
            models_std=summary_std,
            models_median=summary_median,
            ens_mean=ens_mean,
            ens_std=ens_std,
            ens_median=ens_median,
        )

        perf_metric_default = self._metric_default_distribution(task_type, perf_combined.get("mean_columns", [])[1:])
        figures = self._resolve_dataset_images(
            ds_dir=ds_dir,
            dataset_name=dataset_name,
            task_type=task_type,
            metadata=metadata,
            class_counts=class_counts,
            missingness_table=missingness_table,
            perf_metric_default=perf_metric_default,
        )

        tables = {
            "univariate_top10": {
                "columns": univariate_top10.columns if univariate_top10 else [],
                "rows": [[row.get(c, "") for c in univariate_top10.columns] for row in (univariate_top10.rows if univariate_top10 else [])],
            },
            "informative_feature_summary": {
                "columns": informative_summary.columns if informative_summary else [],
                "rows": [[row.get(c, "") for c in informative_summary.columns] for row in (informative_summary.rows if informative_summary else [])],
            },
            "feature_cv_summary": {
                "columns": feature_cv_summary.columns if feature_cv_summary else [],
                "rows": [[row.get(c, "") for c in feature_cv_summary.columns] for row in (feature_cv_summary.rows if feature_cv_summary else [])],
            },
            "data_process_summary": {
                "columns": data_process_summary.columns if data_process_summary else [],
                "rows": [[row.get(c, "") for c in data_process_summary.columns] for row in (data_process_summary.rows if data_process_summary else [])],
            },
            "runtime": {
                "columns": runtimes.columns if runtimes else [],
                "rows": [[row.get(c, "") for c in runtimes.columns] for row in (runtimes.rows if runtimes else [])],
            },
        }

        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "dataset_path": str(ds_dir.resolve()),
            "task_type": task_type,
            "figures": figures,
            "tables": tables,
            "performance": perf_combined,
            "performance_distribution_metric": perf_metric_default,
        }

    def _resolve_dataset_comparison_images(
        self,
        task_type: str,
        dataset_blocks: Sequence[Dict[str, Any]],
    ) -> Dict[str, Optional[str]]:
        dc_dir = self.exp_root / "DatasetComparisons" / "dataCompBoxplots"
        figs: Dict[str, Optional[str]] = {}
        if task_type == "Regression":
            metrics = [
                "Pearson Correlation",
                "Explained Variance",
                "Mean Absolute Error",
                "Mean Squared Error",
                "Median Absolute Error",
                "Max Error",
            ]
        else:
            metrics = [
                "Balanced Accuracy",
                "ROC AUC",
                "PRC AUC",
                "F1 Score",
            ]
        for metric in metrics:
            key = f"overview_{metric}"
            existing = _first_existing(
                [
                    dc_dir / f"DataCompareAllModels_{metric}.png",
                    self.paths.figures_dir / f"DataCompareAllModels_{metric}.png",
                ]
            )
            if existing is not None and self.reuse_existing_figures:
                figs[key] = str(existing)
                continue
            generated = self._plot_dataset_comparison_overview(
                dataset_blocks,
                metric,
                self.paths.figures_dir / f"DataCompareAllModels_{metric}.png",
            )
            figs[key] = generated if generated is not None else (str(existing) if existing is not None else None)
        return figs

    def _format_comparison_table(self, table: Optional[TableData]) -> Dict[str, Any]:
        if not table or not table.rows:
            return {"columns": [], "rows": [], "bold_cells": []}
        rows_out: List[List[str]] = []
        bold_cells: Set[Tuple[int, int]] = set()
        p_idx = None
        sig_idx = None
        for i, c in enumerate(table.columns):
            cl = c.strip().lower()
            if p_idx is None and (cl == "p-value" or cl == "p_value" or cl == "pvalue"):
                p_idx = i
            if sig_idx is None and "sig" in cl:
                sig_idx = i

        for r_i, row in enumerate(table.rows, start=1):
            out_row: List[str] = []
            p_val = None
            for c_i, col in enumerate(table.columns):
                raw = row.get(col, "")
                fv = _safe_float(raw)
                if fv is not None:
                    text = _format_number(fv, is_pvalue=_is_pvalue_col(col))
                else:
                    text = raw
                out_row.append(text)
                if c_i == p_idx:
                    p_val = _safe_float(raw)
            if p_val is not None and p_val < 0.05:
                if p_idx is not None:
                    bold_cells.add((r_i, p_idx))
                if sig_idx is not None:
                    bold_cells.add((r_i, sig_idx))
            rows_out.append(out_row)

        return {
            "columns": table.columns[:],
            "rows": rows_out,
            "bold_cells": [(r, c) for r, c in sorted(bold_cells)],
        }

    def _collect_dataset_comparisons(
        self,
        task_type: str,
        dataset_blocks: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        dc = self.exp_root / "DatasetComparisons"
        if not dc.is_dir():
            return {"present": False}
        kw = self._read_csv_table(dc / "BestCompare_KruskalWallis.csv")
        mw = self._read_csv_table(dc / "BestCompare_MannWhitney.csv")
        wx = self._read_csv_table(dc / "BestCompare_WilcoxonRank.csv")

        out = {
            "present": True,
            "figures": self._resolve_dataset_comparison_images(task_type, dataset_blocks),
            "tables": {
                "kw": self._format_comparison_table(kw),
                "mw": self._format_comparison_table(mw),
                "wx": self._format_comparison_table(wx),
            },
        }

        # Optional KW p-value visualization
        kw_table = out["tables"]["kw"]
        if kw_table["columns"] and kw_table["rows"] and self.enable_plots:
            cols = kw_table["columns"]
            rows = kw_table["rows"]
            p_idx = None
            metric_idx = 0
            for i, c in enumerate(cols):
                if _is_pvalue_col(c):
                    p_idx = i
                    break
            if p_idx is not None:
                vals: List[float] = []
                labels: List[str] = []
                for row in rows:
                    pv = _safe_float(row[p_idx])
                    if pv is None:
                        continue
                    labels.append(row[metric_idx] if metric_idx < len(row) else f"M{len(labels)+1}")
                    vals.append(pv)
                if labels and vals:
                    pfig = self.paths.figures_dir / "datasetcompare_kw_pvalues.png"
                    generated = False
                    if self._mpl_ok():
                        try:
                            import matplotlib.pyplot as plt  # type: ignore

                            fig, ax = plt.subplots(figsize=(8.2, 4.5))
                            ax.bar(labels, vals, color="#4E79A7")
                            ax.axhline(0.05, linestyle="--", color="red", linewidth=1.0)
                            ax.set_ylabel("P-Value")
                            for tick in ax.get_xticklabels():
                                tick.set_rotation(35)
                                tick.set_ha("right")
                            fig.tight_layout()
                            fig.savefig(pfig, dpi=180)
                            plt.close(fig)
                            generated = True
                        except Exception as exc:
                            logger.warning("Could not generate KW p-value visualization with matplotlib: %r", exc)

                    if generated:
                        out["figures"]["kw_pvalues"] = str(pfig)
        return out

    def _image_dimensions_px(self, path: Path) -> Optional[Tuple[int, int]]:
        try:
            with path.open("rb") as f:
                head = f.read(32)
            if len(head) >= 24 and head.startswith(b"\x89PNG\r\n\x1a\n"):
                w = int.from_bytes(head[16:20], "big")
                h = int.from_bytes(head[20:24], "big")
                if w > 0 and h > 0:
                    return (w, h)
        except Exception:
            pass

        try:
            with path.open("rb") as f:
                head = f.read(10)
            if len(head) >= 10 and (head[:6] in {b"GIF87a", b"GIF89a"}):
                w = int.from_bytes(head[6:8], "little")
                h = int.from_bytes(head[8:10], "little")
                if w > 0 and h > 0:
                    return (w, h)
        except Exception:
            pass

        try:
            with path.open("rb") as f:
                if f.read(2) != b"\xff\xd8":
                    return None
                sof_markers = {
                    b"\xc0",
                    b"\xc1",
                    b"\xc2",
                    b"\xc3",
                    b"\xc5",
                    b"\xc6",
                    b"\xc7",
                    b"\xc9",
                    b"\xca",
                    b"\xcb",
                    b"\xcd",
                    b"\xce",
                    b"\xcf",
                }
                while True:
                    b = f.read(1)
                    if not b:
                        break
                    if b != b"\xff":
                        continue
                    marker = f.read(1)
                    while marker == b"\xff":
                        marker = f.read(1)
                    if not marker:
                        break
                    if marker in {b"\xd8", b"\xd9"}:
                        continue
                    seg_len_raw = f.read(2)
                    if len(seg_len_raw) != 2:
                        break
                    seg_len = int.from_bytes(seg_len_raw, "big")
                    if seg_len < 2:
                        break
                    if marker in sof_markers:
                        data = f.read(5)
                        if len(data) != 5:
                            break
                        h = int.from_bytes(data[1:3], "big")
                        w = int.from_bytes(data[3:5], "big")
                        if w > 0 and h > 0:
                            return (w, h)
                        break
                    f.seek(seg_len - 2, 1)
        except Exception:
            pass

        return None

    def _fit_image_in_box(
        self,
        img_w_px: int,
        img_h_px: int,
        box_x: float,
        box_y: float,
        box_w: float,
        box_h: float,
    ) -> Tuple[float, float, float, float]:
        ratio = float(img_w_px) / float(img_h_px)
        box_ratio = box_w / box_h
        if ratio >= box_ratio:
            draw_w = box_w
            draw_h = box_w / ratio
        else:
            draw_h = box_h
            draw_w = box_h * ratio
        draw_x = box_x + (box_w - draw_w) * 0.5
        draw_y = box_y + (box_h - draw_h) * 0.5
        return draw_x, draw_y, draw_w, draw_h

    def _render_table(
        self,
        pdf: _StreamlinePDF,
        *,
        x: float,
        y: float,
        width: float,
        columns: List[str],
        rows: List[List[str]],
        font_size: float = 6.0,
        row_h: float = 3.8,
        bold_cells: Optional[Set[Tuple[int, int]]] = None,
        shade_cells: Optional[Set[Tuple[int, int]]] = None,
        max_first_col_width: float = 42.0,
    ) -> float:
        if not columns:
            return y
        n = len(columns)
        shaded = shade_cells if shade_cells is not None else (bold_cells or set())

        if n == 1:
            col_widths = [width]
        else:
            if n <= 3:
                first_frac = 0.38
            elif n <= 6:
                first_frac = 0.31
            else:
                first_frac = 0.22
            first_w = min(max_first_col_width, width * first_frac)
            rest_w = max(1.0, width - first_w)

            weights: List[float] = []
            sample_rows = rows[: min(len(rows), 40)]
            for c_i in range(1, n):
                max_len = len(str(columns[c_i]))
                for row in sample_rows:
                    if c_i < len(row):
                        max_len = max(max_len, len(str(row[c_i])))
                weights.append(float(max(6, min(max_len, 28))))
            w_sum = sum(weights) if weights else 1.0
            col_widths = [first_w] + [rest_w * (w / w_sum) for w in weights]

        def _cell_numeric(v: Any) -> bool:
            txt = str(v).strip()
            if txt == "":
                return False
            if "+/-" in txt:
                txt = txt.split("+/-", 1)[0].strip()
            return _safe_float(txt) is not None

        numeric_cols: Set[int] = set()
        for c_i in range(1, n):
            non_empty = 0
            numeric = 0
            for row in rows:
                if c_i >= len(row):
                    continue
                text = str(row[c_i]).strip()
                if text == "":
                    continue
                non_empty += 1
                if _cell_numeric(text):
                    numeric += 1
            if non_empty > 0 and (numeric / float(non_empty)) >= 0.80:
                numeric_cols.add(c_i)

        def draw_row(r_i: int, row: Sequence[Any]):
            pdf.set_x(x)
            for c_i in range(n):
                txt = row[c_i] if c_i < len(row) else ""
                max_chars = max(8, int(col_widths[c_i] * 2.2))
                txt = _shorten(str(txt), width=max_chars)
                style = "B" if r_i == 0 else ""
                align = "R" if (r_i > 0 and c_i in numeric_cols) else "L"
                fill = r_i > 0 and (r_i, c_i) in shaded
                if fill:
                    pdf.set_fill_color(230, 230, 230)
                pdf.set_font("Times", style, font_size)
                pdf.cell(col_widths[c_i], row_h, txt, border=1, ln=0, align=align, fill=fill)
            pdf.ln(row_h)

        # Paginate long tables and repeat header on each new page.
        page_bottom = pdf.h - pdf.b_margin
        pdf.set_xy(x, y)
        draw_row(0, columns)
        for idx, row in enumerate(rows, start=1):
            if pdf.get_y() + row_h > page_bottom:
                pdf.add_page()
                pdf.set_xy(x, 20)
                draw_row(0, columns)
            draw_row(idx, row)
        return pdf.get_y()

    def _render_box(self, pdf: _StreamlinePDF, *, x: float, y: float, w: float, title: str, lines: List[str]) -> float:
        pdf.set_xy(x, y)
        pdf.set_font("Times", "B", 9)
        pdf.cell(w, 5, title, border=1, ln=1, align="L")
        pdf.set_x(x)
        pdf.set_font("Times", "", 7)
        body = "\n".join(lines) if lines else "Not available"
        pdf.multi_cell(w, 3.7, body, border=1, align="L")
        return pdf.get_y()

    def _draw_image_panel(
        self,
        pdf: _StreamlinePDF,
        *,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        img_path: Optional[str],
    ) -> float:
        pdf.set_xy(x, y)
        pdf.set_font("Times", "B", 9)
        pdf.cell(w, 5, title, border=1, ln=1, align="L")
        body_y = pdf.get_y()
        pdf.set_xy(x, body_y)
        pdf.cell(w, h, "", border=0, ln=0)

        if img_path:
            p = Path(img_path)
            if not p.is_absolute():
                p = (self.paths.reporting_dir / p).resolve()
            if p.is_file():
                try:
                    inner_x = x + 1.2
                    inner_y = body_y + 1.2
                    inner_w = w - 2.4
                    inner_h = h - 2.4

                    # Keep original figure aspect ratio and center inside panel.
                    dims = self._image_dimensions_px(p)
                    if dims is not None:
                        draw_x, draw_y, draw_w, draw_h = self._fit_image_in_box(
                            dims[0], dims[1], inner_x, inner_y, inner_w, inner_h
                        )
                        pdf.image(str(p), x=draw_x, y=draw_y, w=draw_w, h=draw_h)
                    else:
                        # fpdf2 can preserve aspect if available.
                        try:
                            pdf.image(
                                str(p),
                                x=inner_x,
                                y=inner_y,
                                w=inner_w,
                                h=inner_h,
                                keep_aspect_ratio=True,
                            )
                        except TypeError:
                            # Last-resort fallback preserves aspect by fixing width only.
                            pdf.image(str(p), x=inner_x, y=inner_y, w=inner_w)
                    return body_y + h
                except Exception as exc:
                    logger.warning("Could not render image %s: %r", p, exc)
        pdf.set_xy(x + 1.5, body_y + 2.0)
        pdf.set_font("Times", "", 7)
        pdf.multi_cell(w - 3.0, 3.6, "Figure not available", border=0, align="L")
        return body_y + h

    def _render_global_summary(self, pdf: _StreamlinePDF, report_data: Dict[str, Any]):
        pdf.add_page()
        pdf.set_font("Times", "B", 12)
        pdf.cell(
            190,
            8,
            f"{report_data.get('title', 'STREAMLINE Testing Data Evaluation Report')}: {report_data.get('generated_at')}",
            border=1,
            ln=1,
            align="L",
        )
        pdf.ln(1)

        left_x, left_w = 10.0, 92.0
        right_x, right_w = 108.0, 92.0
        y0 = pdf.get_y()
        column_y = [y0, y0]
        column_x = [left_x, right_x]
        column_w = [left_w, right_w]

        summary = report_data.get("run_command_summary", {}) or {}
        sections = summary.get("sections") or []
        if not sections:
            sections = [
                {
                    "title": "Run Overview",
                    "lines": [
                        f"Experiment Name: {report_data.get('experiment_name', '')}",
                        f"Experiment Root: {report_data.get('experiment_root', '')}",
                        f"Report Mode: {report_data.get('report_mode', '')}",
                    ],
                }
            ]

        for section in sections:
            if not isinstance(section, dict):
                continue
            title = str(section.get("title") or "Summary")
            lines = [
                str(line)
                for line in (section.get("lines") or [])
                if str(line).strip() != ""
            ]
            column = 0 if column_y[0] <= column_y[1] else 1
            if column_y[column] > 260:
                pdf.add_page()
                column_y = [20.0, 20.0]
                column = 0
            column_y[column] = (
                self._render_box(
                    pdf,
                    x=column_x[column],
                    y=column_y[column],
                    w=column_w[column],
                    title=title,
                    lines=lines,
                )
                + 1
            )

    def _render_dataset_header(self, pdf: _StreamlinePDF, ds: Dict[str, Any], section_title: str):
        pdf.add_page()
        pdf.set_font("Times", "B", 11)
        pdf.cell(190, 6, section_title, border=1, ln=1, align="L")
        pdf.set_font("Times", "B", 10)
        pdf.set_fill_color(235, 238, 242)
        pdf.cell(190, 6.5, f"{ds.get('dataset_id')} | Dataset: {ds.get('dataset_name')}", border=1, ln=1, align="L", fill=True)
        pdf.set_font("Times", "", 7.5)
        pdf.cell(190, 5, f"Dataset Path: {ds.get('dataset_path')}", border=1, ln=1, align="L")

    def _render_dataset_eda_page(self, pdf: _StreamlinePDF, ds: Dict[str, Any]):
        self._render_dataset_header(pdf, ds, "EDA and Feature Engineering")
        y_start = 34.0

        uv = ds.get("tables", {}).get("univariate_top10", {})
        uv_cols = uv.get("columns", [])
        uv_rows = uv.get("rows", [])
        pdf.set_xy(10, y_start)
        pdf.set_font("Times", "B", 9)
        pdf.cell(190, 5, "Univariate Analysis (Top 10)", border=1, ln=1, align="L")
        y_after = self._render_table(
            pdf,
            x=10,
            y=pdf.get_y(),
            width=190,
            columns=uv_cols,
            rows=uv_rows,
            font_size=6.0,
            row_h=3.6,
            max_first_col_width=58.0,
        )

        figs = ds.get("figures", {})
        left_bottom = self._draw_image_panel(
            pdf,
            x=10,
            y=y_after + 1,
            w=94,
            h=74,
            title="Missingness Overview (Top 25 Features)",
            img_path=figs.get("missingness_top25"),
        )
        if ds.get("task_type") == "Regression":
            right_title = "Target Distribution (Histogram)"
            right_img = figs.get("target_distribution")
        else:
            right_title = "Class Balance (Observed)"
            right_img = figs.get("class_balance")
        right_bottom = self._draw_image_panel(
            pdf,
            x=106,
            y=y_after + 1,
            w=94,
            h=74,
            title=right_title,
            img_path=right_img,
        )

        y_next = max(left_bottom, right_bottom) + 1

        corr_bottom = self._draw_image_panel(
            pdf,
            x=10,
            y=y_next,
            w=190,
            h=72,
            title="Feature Correlation Matrix (Pearson Coefficients)",
            img_path=figs.get("correlation_matrix"),
        )
        y_next = corr_bottom + 1

        dps = ds.get("tables", {}).get("data_process_summary", {})
        dps_cols = dps.get("columns", [])
        dps_rows = dps.get("rows", [])
        dps_changed_cells: Set[Tuple[int, int]] = set()
        for r_idx in range(1, len(dps_rows)):
            current = dps_rows[r_idx]
            previous = dps_rows[r_idx - 1]
            for c_idx in range(1, min(len(dps_cols), len(current))):
                current_value = str(current[c_idx]).strip()
                previous_value = str(previous[c_idx]).strip() if c_idx < len(previous) else ""
                current_float = _safe_float(current_value)
                previous_float = _safe_float(previous_value)
                if current_float is not None and previous_float is not None:
                    if abs(current_float - previous_float) > 1e-12:
                        dps_changed_cells.add((r_idx + 1, c_idx))
                elif current_value != previous_value:
                    dps_changed_cells.add((r_idx + 1, c_idx))

        ce_lines = [
            "C1 - Remove instances with no outcome and features to ignore",
            "E1 - Feature engineering: add missingness features",
            "C2 - Remove features with invariance or high missingness",
            "C3 - Remove instances with high missingness",
            "E2 - Feature engineering: add or bypass categorical one-hot encoding as configured",
            "C4 - Remove highly correlated features",
            "Gray cells mark values that changed from the previous step.",
        ]

        # Keep the DataProcessSummary + legend together and avoid overlap.
        est_table_h = 5.0 + 3.4 * float(max(2, len(dps_rows) + 1))
        est_legend_h = 5.0 + 3.7 * float(max(2, len(ce_lines)))
        if y_next + est_table_h + est_legend_h > 270:
            self._render_dataset_header(pdf, ds, "EDA and Feature Engineering (continued)")
            y_next = 34.0

        pdf.set_xy(10, y_next)
        pdf.set_font("Times", "B", 9)
        pdf.cell(190, 5, "Data Process and Feature Engineering Summary", border=1, ln=1, align="L")
        if dps_cols:
            y_next = self._render_table(
                pdf,
                x=10,
                y=pdf.get_y(),
                width=190,
                columns=dps_cols,
                rows=dps_rows,
                font_size=5.4,
                row_h=3.4,
                shade_cells=dps_changed_cells,
                max_first_col_width=30.0,
            )
        else:
            pdf.set_x(10)
            pdf.set_font("Times", "", 7)
            pdf.multi_cell(190, 3.6, "DataProcessSummary.csv not found.", border=1, align="L")
            y_next = pdf.get_y()

        legend_title = "DataProcessSummary Step Key (C = cleaning, E = feature engineering)"
        self._render_box(
            pdf,
            x=10,
            y=y_next + 2,
            w=190,
            title=legend_title,
            lines=ce_lines,
        )

    def _render_feature_learning_page(self, pdf: _StreamlinePDF, ds: Dict[str, Any]):
        self._render_dataset_header(pdf, ds, "Feature Learning and Feature Selection")
        figs = ds.get("figures", {})
        panels = [p for p in (figs.get("feature_learning_panels") or []) if isinstance(p, dict) and p.get("path")]

        if len(panels) >= 2:
            left_bottom = self._draw_image_panel(
                pdf,
                x=10,
                y=34,
                w=90,
                h=88,
                title=str(panels[0].get("title") or "Top Scores (Method 1)"),
                img_path=str(panels[0].get("path") or ""),
            )
            right_bottom = self._draw_image_panel(
                pdf,
                x=110,
                y=34,
                w=90,
                h=88,
                title=str(panels[1].get("title") or "Top Scores (Method 2)"),
                img_path=str(panels[1].get("path") or ""),
            )
            y_next = max(left_bottom, right_bottom) + 3
        elif len(panels) == 1:
            y_next = self._draw_image_panel(
                pdf,
                x=52,
                y=34,
                w=106,
                h=96,
                title=str(panels[0].get("title") or "Top Scores"),
                img_path=str(panels[0].get("path") or ""),
            ) + 3
        else:
            y_next = self._draw_image_panel(
                pdf,
                x=52,
                y=34,
                w=106,
                h=96,
                title="Top Feature Importance Scores",
                img_path=figs.get("mutual_info"),
            ) + 3

        cv_summary = ds.get("tables", {}).get("feature_cv_summary", {})
        cv_cols = cv_summary.get("columns", [])
        cv_rows = cv_summary.get("rows", [])
        if cv_cols:
            pdf.set_xy(10, y_next)
            pdf.set_font("Times", "B", 9)
            pdf.cell(190, 5, "Feature Learning / Selection CV Summary", border=1, ln=1, align="L")
            y_next = self._render_table(
                pdf,
                x=10,
                y=pdf.get_y(),
                width=190,
                columns=cv_cols,
                rows=cv_rows,
                font_size=5.6,
                row_h=3.7,
                max_first_col_width=34.0,
            ) + 3
        else:
            y_next = max(y_next, 133)

        t = ds.get("tables", {}).get("informative_feature_summary", {})
        cols = t.get("columns", [])
        rows = t.get("rows", [])
        pdf.set_xy(10, y_next)
        pdf.set_font("Times", "B", 9)
        pdf.cell(190, 5, "Informative Feature Summary", border=1, ln=1, align="L")
        self._render_table(
            pdf,
            x=10,
            y=pdf.get_y(),
            width=190,
            columns=cols,
            rows=rows,
            font_size=6.5,
            row_h=4.0,
            max_first_col_width=62.0,
        )

        # Render additional FI method panels on continuation pages when present.
        if len(panels) > 2:
            remaining = panels[2:]
            slots = [(10, 34), (110, 34), (10, 132), (110, 132)]
            for i in range(0, len(remaining), 4):
                self._render_dataset_header(pdf, ds, "Feature Learning and Feature Selection (continued)")
                chunk = remaining[i : i + 4]
                for panel, (x, y) in zip(chunk, slots):
                    self._draw_image_panel(
                        pdf,
                        x=x,
                        y=y,
                        w=90,
                        h=88,
                        title=str(panel.get("title") or "Top Scores"),
                        img_path=str(panel.get("path") or ""),
                    )

    def _render_performance_page(self, pdf: _StreamlinePDF, ds: Dict[str, Any]):
        self._render_dataset_header(pdf, ds, self.performance_page_title())
        perf = ds.get("performance", {})
        figs = ds.get("figures", {})
        mean_cols = perf.get("mean_columns", [])
        mean_rows = perf.get("mean_rows", [])
        mean_highlight = set(
            (int(r), int(c))
            for r, c in perf.get("mean_highlight_cells", perf.get("mean_bold_cells", []))
        )
        med_cols = perf.get("median_columns", [])
        med_rows = perf.get("median_rows", [])
        med_highlight = set(
            (int(r), int(c))
            for r, c in perf.get("median_highlight_cells", perf.get("median_bold_cells", []))
        )

        y = 34.0
        pdf.set_xy(10, y)
        pdf.set_font("Times", "B", 9)
        pdf.cell(190, 5, "Model and Ensemble Performance (Mean +/- SD; gray = best/tied metric)", border=1, ln=1, align="L")
        y = self._render_table(
            pdf,
            x=10,
            y=pdf.get_y(),
            width=190,
            columns=mean_cols,
            rows=mean_rows,
            font_size=5.5,
            row_h=3.4,
            shade_cells=mean_highlight,
            max_first_col_width=46.0,
        )

        y += 2
        pdf.set_xy(10, y)
        pdf.set_font("Times", "B", 9)
        pdf.cell(190, 5, "Model and Ensemble Performance (Median; gray = best/tied metric)", border=1, ln=1, align="L")
        y = self._render_table(
            pdf,
            x=10,
            y=pdf.get_y(),
            width=190,
            columns=med_cols,
            rows=med_rows,
            font_size=5.5,
            row_h=3.4,
            shade_cells=med_highlight,
            max_first_col_width=46.0,
        )

        y = max(y + 2, 178)
        metric = ds.get("performance_distribution_metric") or ""
        distribution_title = f"{metric} Distribution by Algorithm" if metric else "Performance Distribution by Algorithm"
        composite_path = figs.get("composite_feature_scores")
        distribution_path = figs.get("performance_distribution")

        if composite_path and distribution_path:
            self._draw_image_panel(
                pdf,
                x=10,
                y=y,
                w=94,
                h=78,
                title="Permutation Feature Importance (Composite)",
                img_path=composite_path,
            )
            self._draw_image_panel(
                pdf,
                x=106,
                y=y,
                w=94,
                h=78,
                title=distribution_title,
                img_path=distribution_path,
            )
        elif composite_path:
            self._draw_image_panel(
                pdf,
                x=10,
                y=y,
                w=190,
                h=78,
                title="Permutation Feature Importance (Composite)",
                img_path=composite_path,
            )
        else:
            self._draw_image_panel(
                pdf,
                x=10,
                y=y,
                w=190,
                h=78,
                title=distribution_title,
                img_path=distribution_path,
            )

    def _render_evaluation_page(self, pdf: _StreamlinePDF, ds: Dict[str, Any]):
        if ds.get("task_type") == "Regression":
            self._render_dataset_header(pdf, ds, self.evaluation_page_title(ds))
            figs = ds.get("figures", {})
            self._draw_image_panel(
                pdf,
                x=10,
                y=34,
                w=94,
                h=104,
                title="Actual vs Predicted (Test Set)",
                img_path=figs.get("reg_actual_vs_pred"),
            )
            self._draw_image_panel(
                pdf,
                x=106,
                y=34,
                w=94,
                h=104,
                title="Test Residual Q-Q Plot",
                img_path=figs.get("reg_test_resid"),
            )
            self._draw_image_panel(
                pdf,
                x=10,
                y=142,
                w=190,
                h=104,
                title="Residual Distribution (Test Set)",
                img_path=figs.get("reg_residual_dist"),
            )
            return

        self._render_dataset_header(pdf, ds, self.evaluation_page_title(ds))
        figs = ds.get("figures", {})
        self._draw_image_panel(
            pdf,
            x=10,
            y=34,
            w=94,
            h=104,
            title="ROC Summary (Base Models)",
            img_path=figs.get("models_roc"),
        )
        self._draw_image_panel(
            pdf,
            x=106,
            y=34,
            w=94,
            h=104,
            title="PRC Summary (Base Models)",
            img_path=figs.get("models_prc"),
        )
        self._draw_image_panel(
            pdf,
            x=10,
            y=142,
            w=94,
            h=104,
            title="ROC Summary (Ensembles)",
            img_path=figs.get("ensembles_roc"),
        )
        self._draw_image_panel(
            pdf,
            x=106,
            y=142,
            w=94,
            h=104,
            title="PRC Summary (Ensembles)",
            img_path=figs.get("ensembles_prc"),
        )

    def _render_runtime_page(self, pdf: _StreamlinePDF, ds: Dict[str, Any]):
        self._render_dataset_header(pdf, ds, "Runtime Summary")
        rt = ds.get("tables", {}).get("runtime", {})
        cols = rt.get("columns", [])
        rows = rt.get("rows", [])
        self._render_table(
            pdf,
            x=10,
            y=34,
            width=190,
            columns=cols,
            rows=rows,
            font_size=6.5,
            row_h=4.0,
            max_first_col_width=72.0,
        )

    def _render_dataset_comparison_pages(self, pdf: _StreamlinePDF, block: Dict[str, Any], task_type: str):
        if not block.get("present"):
            return

        pdf.add_page()
        pdf.set_font("Times", "B", 11)
        pdf.cell(190, 6, "Dataset Comparisons", border=1, ln=1, align="L")
        pdf.set_font("Times", "", 9)
        pdf.cell(190, 5, "Comparison Overview (All Datasets)", border=1, ln=1, align="L")

        figs = block.get("figures", {})
        if task_type == "Regression":
            keys = [
                "overview_Pearson Correlation",
                "overview_Explained Variance",
                "overview_Mean Absolute Error",
                "overview_Mean Squared Error",
            ]
        else:
            keys = [
                "overview_Balanced Accuracy",
                "overview_ROC AUC",
                "overview_PRC AUC",
                "overview_F1 Score",
            ]

        # Compact overview layout so DatasetComparisons consumes fewer pages.
        panels = [
            (10, 34, 94, 82, keys[0], "Overview 1"),
            (106, 34, 94, 82, keys[1], "Overview 2"),
            (10, 120, 94, 82, keys[2], "Overview 3"),
            (106, 120, 94, 82, keys[3], "Overview 4"),
        ]
        for x, y, w, h, key, fallback_title in panels:
            if key.startswith("overview_"):
                metric_label = key.replace("overview_", "")
                title = f"Across Datasets: {metric_label}"
            else:
                title = fallback_title
            self._draw_image_panel(pdf, x=x, y=y, w=w, h=h, title=title, img_path=figs.get(key))

        if figs.get("kw_pvalues"):
            pdf.set_xy(10, 206)
            pdf.set_font("Times", "B", 9)
            pdf.cell(190, 5, "Dataset Comparisons: Kruskal-Wallis P-Values", border=1, ln=1, align="L")
            self._draw_image_panel(
                pdf,
                x=10,
                y=211,
                w=190,
                h=48,
                title="Kruskal-Wallis P-Value Overview",
                img_path=figs.get("kw_pvalues"),
            )

        table_specs = [
            ("Best Comparisons - Kruskal-Wallis", "kw"),
            ("Best Comparisons - Mann-Whitney U", "mw"),
            ("Best Comparisons - Wilcoxon Rank-Sum", "wx"),
        ]
        # Render all stats tables across as few pages as possible.
        pdf.add_page()
        y = 24.0
        for title, key in table_specs:
            t = block.get("tables", {}).get(key, {})
            cols = t.get("columns", [])
            rows = t.get("rows", [])
            if not cols:
                continue
            bold = set((int(r), int(c)) for r, c in t.get("bold_cells", []))
            est_h = 5.0 + 3.2 * float(max(2, len(rows) + 1))
            if y + est_h > 274:
                pdf.add_page()
                y = 24.0
            pdf.set_xy(10, y)
            pdf.set_font("Times", "B", 11)
            pdf.cell(190, 6, title, border=1, ln=1, align="L")
            y = self._render_table(
                pdf,
                x=10,
                y=pdf.get_y(),
                width=190,
                columns=cols,
                rows=rows,
                font_size=5.0,
                row_h=3.2,
                bold_cells=bold,
                max_first_col_width=28.0,
            )
            y += 3.0

    def _render_pdf(self, report_data: Dict[str, Any]):
        if FPDF is None:
            raise ImportError("fpdf2 is required for PDF rendering. Install `fpdf2`.")
        footer_text = (
            f"Generated with STREAMLINE ({report_data.get('streamline_version', 'unknown')}): "
            "(https://github.com/UrbsLab/STREAMLINE)"
        )
        pdf = _StreamlinePDF(footer_text=footer_text)
        pdf.alias_nb_pages()
        pdf.set_auto_page_break(auto=True, margin=12)
        pdf.set_margins(10, 8, 10)

        self._render_global_summary(pdf, report_data)

        datasets = report_data.get("datasets", [])
        is_replication_report = str(report_data.get("report_mode", "standard")).strip().lower() == "replication"
        for ds in datasets:
            self._render_dataset_eda_page(pdf, ds)
            if not is_replication_report:
                self._render_feature_learning_page(pdf, ds)
            self._render_performance_page(pdf, ds)
            self._render_evaluation_page(pdf, ds)
            self._render_runtime_page(pdf, ds)

        dc = report_data.get("dataset_comparisons", {})
        task_type = datasets[0].get("task_type", "Multiclass Classification") if datasets else "Multiclass Classification"
        self._render_dataset_comparison_pages(pdf, dc, task_type)

        pdf.output(str(self.paths.pdf))

    def save_runtime(self):
        rt_dir = self.exp_root / "runtime"
        try:
            rt_dir.mkdir(exist_ok=True)
            elapsed = time.time() - (self.job_start_time or time.time())
            (rt_dir / "runtime_report.txt").write_text(str(elapsed))
        except PermissionError:
            logger.warning("Could not write runtime_report.txt under %s (permission denied).", rt_dir)

    def run(self):
        self.job_start_time = time.time()

        datasets = self._list_datasets()
        if not datasets:
            if self.report_mode == "replication":
                raise RuntimeError(
                    "No replication dataset folders found. Expected "
                    "<dataset>/replication/<rep_dataset>/ with exploratory/ and model_evaluation/."
                )
            raise RuntimeError(
                "No dataset folders found. Expected subdirectories containing exploratory/ and model_evaluation/."
            )

        metadata_pickle = self._read_pickle_if_exists(self.exp_root / "metadata.pickle") or {}
        alg_info = self._read_pickle_if_exists(self.exp_root / "algInfo.pickle") or {}
        run_params_all = self._read_pickle_if_exists(self.exp_root / "run_params.pickle") or {}
        run_params = self._latest_run_params(run_params_all)

        # Optional metadata fallback to constructor values.
        metadata = {
            "Experiment Root": str(self.exp_root),
            "Output Path": str(self.exp_root.parent),
            "Experiment Name": self.experiment_name,
            "Outcome Label": self.outcome_label or metadata_pickle.get("Outcome Label", ""),
            "Outcome Type": self.outcome_type or metadata_pickle.get("Outcome Type", ""),
            "Instance Label": self.instance_label or metadata_pickle.get("Instance Label", ""),
        }

        dataset_blocks: List[Dict[str, Any]] = []
        for idx, ds_dir in enumerate(datasets, start=1):
            dataset_id = f"D{idx}"
            dataset_blocks.append(self._collect_dataset_block(ds_dir, dataset_id, metadata_pickle))

        primary_task = dataset_blocks[0].get("task_type", "Multiclass Classification")
        if self.report_mode == "replication":
            # Replication mode focuses on available replication outputs only.
            dc_block = {"present": False}
        else:
            dc_block = (
                self._collect_dataset_comparisons(primary_task, dataset_blocks)
                if len(dataset_blocks) > 1
                else {"present": False}
            )

        if self.report_mode == "replication":
            report_title = "STREAMLINE Replication Data Evaluation Report"
        else:
            report_title = "STREAMLINE Testing Data Evaluation Report"

        run_command_summary = self.build_run_command_summary(
            metadata=metadata,
            metadata_pickle=metadata_pickle,
            run_params_all=run_params_all,
            run_params=run_params,
            dataset_blocks=dataset_blocks,
        )

        report_data: Dict[str, Any] = {
            "title": report_title,
            "generated_at": _now_iso_local(),
            "generated_at_epoch": int(time.time()),
            "streamline_version": _try_streamline_version(),
            "experiment_name": self.experiment_name,
            "experiment_root": str(self.exp_root),
            "report_mode": self.report_mode,
            "metadata": metadata,
            "metadata_pickle": metadata_pickle,
            "alg_info": alg_info,
            "run_params": run_params,
            "run_command_summary": run_command_summary,
            "datasets": dataset_blocks,
            "dataset_comparisons": dc_block,
        }

        self.paths.data_json.write_text(json.dumps(report_data, indent=2))

        if self.make_pdf:
            self._render_pdf(report_data)

        jobs_completed = self.exp_root / "jobsCompleted"
        try:
            jobs_completed.mkdir(exist_ok=True)
            (jobs_completed / "job_reporting.txt").write_text("complete")
        except PermissionError:
            logger.warning("Could not write job completion marker under %s (permission denied).", jobs_completed)
        self.save_runtime()
        logger.info("Reporting phase complete: %s", self.paths.pdf)


ReportPhaseJobPdfFlow = ReportPhaseJob


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate STREAMLINE Testing Data Evaluation PDF report.")
    parser.add_argument("--experiment-path", help="Path to experiment output directory.")
    parser.add_argument("--output-path", help="Parent output path containing experiment directory.")
    parser.add_argument("--experiment-name", help="Experiment folder name.")
    parser.add_argument(
        "--reporting-dir",
        default=None,
        help="Optional output directory for report artifacts (report_data.json, report.pdf, figures).",
    )
    parser.add_argument(
        "--report-mode",
        default="standard",
        choices=["standard", "replication"],
        help="Report scope: standard datasets or replication datasets under replication/ folders.",
    )
    parser.add_argument("--outcome-label", default=None)
    parser.add_argument("--outcome-type", default=None)
    parser.add_argument("--instance-label", default=None)
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF generation and only build report_data.json.")
    parser.add_argument("--no-plots", action="store_true", help="Disable on-the-fly figure generation.")
    parser.add_argument(
        "--no-reuse-figures",
        action="store_true",
        help="Do not reuse existing PNGs from report/figure locations before generation.",
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    job = ReportPhaseJob(
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        experiment_path=args.experiment_path,
        reporting_dir=args.reporting_dir,
        report_mode=args.report_mode,
        outcome_label=args.outcome_label,
        outcome_type=args.outcome_type,
        instance_label=args.instance_label,
        make_pdf=not args.no_pdf,
        enable_plots=not args.no_plots,
        reuse_existing_figures=not args.no_reuse_figures,
    )
    job.run()


if __name__ == "__main__":
    main()
