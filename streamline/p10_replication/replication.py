from __future__ import annotations

import json
import logging
import os
import pickle
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import brier_score_loss

from streamline.p1_data_process.data_process import DataProcess
from streamline.p6_modeling.utils.submodels import (
    BinaryClassificationModel,
    MulticlassClassificationModel,
    RegressionModel,
)
from streamline.p6_modeling.utils.support import multiclass_brier_score
from streamline.p7_ensembles.ensembles import (
    _calc_basic_metrics,
    _calc_curves_scores_from_proba,
)
from streamline.p8_summary_statistics.statistics import StatisticsPhaseJob


logger = logging.getLogger(__name__)


def _normalize_outcome_type(outcome_type: str) -> str:
    """Normalize multiple aliases to STREAMLINE internal outcome type labels."""
    value = (outcome_type or "").strip().lower()
    if value in {"binary", "bin", "classification_binary"}:
        return "Binary"
    if value in {"multiclass", "multi", "classification_multiclass"}:
        return "Multiclass"
    if value in {"continuous", "regression", "numeric"}:
        return "Continuous"
    return outcome_type


def _read_table(file_path: str) -> pd.DataFrame:
    """Read CSV/TSV/TXT input consistently with phase-1 behavior."""
    ext = Path(file_path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(file_path, na_values="NA", sep=",")
    if ext == ".tsv":
        return pd.read_csv(file_path, na_values="NA", sep="\t")
    if ext == ".txt":
        return pd.read_csv(file_path, na_values="NA", delim_whitespace=True)
    raise ValueError(f"Unsupported replication file extension: {ext}")


def _safe_pickle_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("rb") as f:
        return pickle.load(f)


def _jsonify(value: Any) -> Any:
    """Convert numpy/pandas scalar containers to JSON-safe python values."""
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, pd.Series):
        return [_jsonify(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return _jsonify(value.to_dict(orient="records"))
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    return value


class ReplicationJob:
    """
    Phase 10 replication job.

    For one replication dataset, this job:
    1. Replays the p1 data-processing decisions learned on the train dataset.
    2. Replays p2 imputation/scaling per CV split.
    3. Applies p6 trained base models and writes p8-compatible metrics/curves artifacts.
    4. Applies p7 trained ensembles (classification only) and writes ensemble artifacts.
    5. Runs p8 statistics on replication outputs.

    The produced tree is rooted at:
      <train_dataset_dir>/replication/<replication_dataset_name>/
    """

    def __init__(
        self,
        dataset_filename: str,
        dataset_for_rep: str,
        full_path: str,
        outcome_label: str,
        outcome_type: str,
        instance_label: Optional[str],
        match_label: Optional[str],
        ignore_features: Optional[List[str]] = None,
        cv_partitions: int = 3,
        exclude_plots: Optional[List[str]] = None,
        categorical_cutoff: int = 10,
        sig_cutoff: float = 0.05,
        featureeng_missingness: float = 0.5,
        cleaning_missingness: float = 0.5,
        scale_data: bool = True,
        impute_data: bool = True,
        multi_impute: bool = True,
        show_plots: bool = False,
        scoring_metric: str = "balanced_accuracy",
        random_state: Optional[int] = None,
    ):
        self.dataset_filename = dataset_filename
        self.dataset_for_rep = dataset_for_rep
        self.train_root = Path(full_path)
        self.experiment_root = self.train_root.parent

        self.outcome_label = outcome_label
        self.outcome_type = _normalize_outcome_type(outcome_type)
        self.instance_label = instance_label
        self.match_label = match_label

        self.ignore_features = ignore_features or []
        self.cv_partitions = int(cv_partitions)
        self.exclude_plots = exclude_plots or []
        self.categorical_cutoff = int(categorical_cutoff)
        self.sig_cutoff = float(sig_cutoff)
        self.featureeng_missingness = float(featureeng_missingness)
        self.cleaning_missingness = float(cleaning_missingness)
        self.scale_data = bool(scale_data)
        self.impute_data = bool(impute_data)
        self.multi_impute = bool(multi_impute)
        self.show_plots = bool(show_plots)
        self.scoring_metric = scoring_metric
        self.random_state = random_state

        self.train_name = self.train_root.name
        self.apply_name = Path(self.dataset_filename).stem
        self.rep_root = self.train_root / "replication" / self.apply_name

        self.exploratory_dir = self.rep_root / "exploratory"
        self.cv_dir = self.rep_root / "CVDatasets"
        self.model_eval_dir = self.rep_root / "model_evaluation"
        self.model_metrics_dir = self.model_eval_dir / "metrics_by_cv"
        self.model_curves_dir = self.model_eval_dir / "curves_by_cv"
        self.model_pickled_metrics_dir = self.model_eval_dir / "pickled_metrics"
        self.ensemble_root = self.rep_root / "ensemble_evaluation"
        self.ensemble_metrics_dir = self.ensemble_root / "metrics_by_cv"
        self.ensemble_curves_dir = self.ensemble_root / "curves_by_cv"
        self.ensemble_pickled_dir = self.ensemble_root / "pickled_ensembles"

        self.algorithms, self.abbrev = self._load_algorithms()

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._prepare_dirs()
        self._auto_correct_labels_from_training_cv()

        rep_raw = _read_table(self.dataset_filename)
        rep_raw.columns = rep_raw.columns.str.strip()

        raw_train_columns = self._load_training_raw_columns()
        missing_cols = [c for c in raw_train_columns if c not in rep_raw.columns]
        if missing_cols:
            raise Exception(
                "Replication dataset is missing one or more training columns: "
                + ", ".join(missing_cols)
            )
        rep_raw = rep_raw[raw_train_columns].copy()

        if self.instance_label and self.instance_label not in rep_raw.columns:
            logger.warning("Instance label '%s' not in replication dataset; ignoring.", self.instance_label)
            self.instance_label = None
        if self.match_label and self.match_label not in rep_raw.columns:
            logger.warning("Match label '%s' not in replication dataset; ignoring.", self.match_label)
            self.match_label = None

        if self.outcome_label not in rep_raw.columns:
            raise Exception(f"Outcome label '{self.outcome_label}' is missing in replication dataset")

        processed, cat_features, quant_features, transition_df = self._replay_data_processing(rep_raw)

        self._write_processed_dataset(processed, cat_features, quant_features, transition_df)
        self._write_eda_artifacts(processed, cat_features, quant_features)

        fold_map = self._resolve_fold_map()
        if not fold_map:
            raise Exception("No CV train folds found in training dataset; cannot run replication")

        self._evaluate_models(processed, cat_features, quant_features, fold_map)

        if self.outcome_type in {"Binary", "Multiclass"}:
            self._evaluate_ensembles(processed, fold_map)

        self._run_statistics(cv_partitions=len(fold_map))

        logger.info("Replication complete for %s", self.apply_name)
        jobs_completed = self.experiment_root / "jobsCompleted"
        jobs_completed.mkdir(parents=True, exist_ok=True)
        with (jobs_completed / f"job_apply_{self.apply_name}.txt").open("w") as f:
            f.write("complete")

    # ------------------------------------------------------------------
    # Setup and discovery
    # ------------------------------------------------------------------

    def _prepare_dirs(self) -> None:
        self.exploratory_dir.mkdir(parents=True, exist_ok=True)
        self.cv_dir.mkdir(parents=True, exist_ok=True)
        self.model_metrics_dir.mkdir(parents=True, exist_ok=True)
        self.model_curves_dir.mkdir(parents=True, exist_ok=True)
        self.model_pickled_metrics_dir.mkdir(parents=True, exist_ok=True)


    def _auto_correct_labels_from_training_cv(self) -> None:
        """
        Guard against stale metadata labels by inferring labels from training CV files.

        - Outcome label is expected to be the first column in CV train files.
        - Instance label must be highly unique; otherwise treat it as a feature.
        """
        cv_train_files = sorted((self.train_root / "CVDatasets").glob(f"{self.train_name}_CV_*_Train.csv"))
        if not cv_train_files:
            return

        train_df = pd.read_csv(cv_train_files[0], nrows=500, na_values="NA", sep=",")
        if train_df.empty:
            return

        first_col = str(train_df.columns[0])
        if first_col and self.outcome_label != first_col:
            logger.warning(
                "Outcome label '%s' does not match training CV schema; using '%s' for replication.",
                self.outcome_label,
                first_col,
            )
            self.outcome_label = first_col

        if self.instance_label:
            if self.instance_label == self.outcome_label:
                logger.warning(
                    "Instance label '%s' equals outcome label; ignoring instance label for replication.",
                    self.instance_label,
                )
                self.instance_label = None
            elif self.instance_label in train_df.columns:
                uniq = train_df[self.instance_label].nunique(dropna=True)
                ratio = float(uniq) / float(max(1, len(train_df)))
                if ratio < 0.95:
                    logger.warning(
                        "Instance label '%s' is not near-unique in training CV data; treating it as a regular feature.",
                        self.instance_label,
                    )
                    self.instance_label = None
            else:
                logger.warning(
                    "Instance label '%s' not found in training CV schema; ignoring.",
                    self.instance_label,
                )
                self.instance_label = None

        if self.match_label and self.match_label not in train_df.columns:
            logger.warning("Match label '%s' not found in training CV schema; ignoring.", self.match_label)
            self.match_label = None

        if self.outcome_type == "Continuous":
            valid_regression_metrics = {
                "explained_variance",
                "max_error",
                "mean_absolute_error",
                "mean_squared_error",
                "median_absolute_error",
                "pearson_correlation",
            }
            metric_name = str(self.scoring_metric).strip().lower() if self.scoring_metric is not None else ""
            if metric_name not in valid_regression_metrics:
                self.scoring_metric = "explained_variance"

    def _load_algorithms(self) -> Tuple[List[str], Dict[str, str]]:
        """
        Discover active base algorithms and their short names.

        Priority:
          1) experiment-level algInfo.pickle
          2) model pickle filenames under train dataset
        """
        alg_info_path = self.experiment_root / "algInfo.pickle"
        algorithms: List[str] = []
        abbrev: Dict[str, str] = {}

        if alg_info_path.exists():
            with alg_info_path.open("rb") as f:
                alg_info = pickle.load(f)
            for algorithm, payload in alg_info.items():
                if not isinstance(payload, (list, tuple)) or len(payload) == 0:
                    continue
                use_flag = bool(payload[0])
                short_name = payload[1] if len(payload) > 1 and payload[1] else algorithm
                if use_flag:
                    algorithms.append(algorithm)
                    abbrev[algorithm] = short_name

        if algorithms:
            return algorithms, abbrev

        model_dir = self.train_root / "models" / "pickledModels"
        if model_dir.exists():
            shorts = sorted(
                {
                    m.group(1)
                    for fn in model_dir.glob("*.pickle")
                    for m in [re.match(r"(.+?)_\d+\.pickle$", fn.name)]
                    if m
                }
            )
            algorithms = shorts
            abbrev = {s: s for s in shorts}

        return algorithms, abbrev

    def _load_training_raw_columns(self) -> List[str]:
        """Recover training raw-column order, including labels."""
        train_file = Path(self.dataset_for_rep)
        if train_file.exists():
            train_df = _read_table(str(train_file))
            train_df.columns = train_df.columns.str.strip()
            return list(train_df.columns)

        # Fallback to p1 artifact if original raw file path is not available.
        original_names = self.train_root / "exploratory" / "initial" / "OriginalFeatureNames.csv"
        if original_names.exists():
            row = pd.read_csv(original_names, header=None).iloc[0].dropna().tolist()
            cols = [str(x) for x in row]
            for lbl in (self.outcome_label, self.instance_label, self.match_label):
                if lbl and lbl not in cols:
                    cols.append(lbl)
            return cols

        raise Exception(
            "Could not determine training raw columns. "
            "Provide dataset_for_rep as the original training dataset file path."
        )

    def _resolve_fold_map(self) -> List[Tuple[int, int]]:
        """
        Map contiguous replication fold ids to source training fold ids with strict parity.

        Returns list of tuples: (replication_cv_id, source_training_cv_id)
        """
        cv_files = sorted((self.train_root / "CVDatasets").glob(f"{self.train_name}_CV_*_Train.csv"))
        source_ids = []
        for path in cv_files:
            match = re.search(r"_CV_(\d+)_Train\.csv$", path.name)
            if match:
                source_ids.append(int(match.group(1)))

        source_ids = sorted(set(source_ids))
        if not source_ids:
            return []

        # Strictly align expected folds to metadata-defined CV partitions when available.
        if self.cv_partitions > 0:
            expected_source_ids = list(range(self.cv_partitions))
            missing_cv_files = [cv for cv in expected_source_ids if cv not in source_ids]
            if missing_cv_files:
                raise Exception(
                    "Strict fold parity failed: missing training CV files for folds "
                    + ", ".join(str(cv) for cv in missing_cv_files)
                )
            source_ids = expected_source_ids

        # Strictly require all expected folds for every active base algorithm.
        model_dir = self.train_root / "models" / "pickledModels"
        if not model_dir.exists():
            raise Exception("Strict fold parity failed: models/pickledModels directory is missing")

        if self.algorithms:
            for algorithm in self.algorithms:
                small = self.abbrev.get(algorithm, algorithm)
                ids = {
                    int(m.group(1))
                    for pattern in (f"{small}_*.pickle", f"{algorithm}_*.pickle")
                    for p in model_dir.glob(pattern)
                    for m in [re.match(r".+?_(\d+)\.pickle$", p.name)]
                    if m
                }
                missing_model_folds = [cv for cv in source_ids if cv not in ids]
                if missing_model_folds:
                    raise Exception(
                        "Strict fold parity failed: missing trained model pickles for "
                        f"algorithm={algorithm}, folds={missing_model_folds}"
                    )

        return [(idx, source_cv) for idx, source_cv in enumerate(source_ids)]

    # ------------------------------------------------------------------
    # p1 replay on replication data
    # ------------------------------------------------------------------

    def _replay_data_processing(
        self,
        raw_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame]:
        data = raw_df.copy()

        initial_cat = _safe_pickle_load(
            self.train_root / "exploratory" / "initial" / "initial_categorical_features.pickle",
            [],
        )
        initial_quant = _safe_pickle_load(
            self.train_root / "exploratory" / "initial" / "initial_quantitative_features.pickle",
            [],
        )

        categorical_features = [f for f in initial_cat if f in data.columns and f != self.outcome_label]
        quantitative_features = [f for f in initial_quant if f in data.columns and f != self.outcome_label]

        class_count = self._load_training_class_count(default=data[self.outcome_label].nunique(dropna=True))
        transition_columns = self._build_transition_columns(class_count)
        transition_df = pd.DataFrame(columns=transition_columns)

        transition_df.loc["Original"] = self._counts_summary_row(
            data, categorical_features, quantitative_features, class_count
        )

        # C1 - label alignment, remove ignored and missing outcome rows
        data = self._apply_ordinal_encoding(data)
        self._apply_binary_consistency(data)
        data = self._drop_ignored_and_missing_outcome(data)
        categorical_features, quantitative_features = self._sync_feature_lists(
            data, categorical_features, quantitative_features
        )
        transition_df.loc["C1"] = self._counts_summary_row(
            data, categorical_features, quantitative_features, class_count
        )

        # E1 - engineered missingness indicators from training
        data, categorical_features = self._apply_engineered_missingness(data, categorical_features)
        transition_df.loc["E1"] = self._counts_summary_row(
            data, categorical_features, quantitative_features, class_count
        )

        # C2 - invariant + training-removed features
        data, categorical_features, quantitative_features = self._drop_invariant_features(
            data, categorical_features, quantitative_features
        )
        data, categorical_features, quantitative_features = self._drop_training_removed_features(
            data, categorical_features, quantitative_features
        )
        transition_df.loc["C2"] = self._counts_summary_row(
            data, categorical_features, quantitative_features, class_count
        )

        # C3 - remove high-missingness rows
        data = self._drop_high_missing_rows(data)
        transition_df.loc["C3"] = self._counts_summary_row(
            data, categorical_features, quantitative_features, class_count
        )

        # E2 - one hot encode multi-level categoricals
        data, categorical_features = self._apply_one_hot_encoding(data, categorical_features)
        categorical_features, quantitative_features = self._sync_feature_lists(
            data, categorical_features, quantitative_features
        )
        transition_df.loc["E2"] = self._counts_summary_row(
            data, categorical_features, quantitative_features, class_count
        )

        # C4 - remove correlated features from training + final feature alignment
        data, categorical_features, quantitative_features = self._drop_training_correlated_features(
            data, categorical_features, quantitative_features
        )
        data, categorical_features, quantitative_features = self._align_to_training_processed_features(
            data, categorical_features, quantitative_features
        )
        transition_df.loc["C4"] = self._counts_summary_row(
            data, categorical_features, quantitative_features, class_count
        )

        return data, categorical_features, quantitative_features, transition_df

    def _load_training_class_count(self, default: int) -> int:
        class_counts_path = self.train_root / "exploratory" / "ClassCounts.csv"
        if not class_counts_path.exists():
            return max(2, int(default)) if self.outcome_type == "Multiclass" else int(default)
        try:
            class_counts = pd.read_csv(class_counts_path)
            if class_counts.shape[0] > 0:
                return int(class_counts.shape[0])
        except Exception:
            pass
        return max(2, int(default)) if self.outcome_type == "Multiclass" else int(default)

    def _build_transition_columns(self, class_count: int) -> List[str]:
        base = [
            "Instances",
            "Total Features",
            "Categorical Features",
            "Quantitative Features",
            "Missing Values",
            "Missing Percent",
        ]
        if self.outcome_type == "Binary":
            return base + ["Class 0", "Class 1"]
        if self.outcome_type == "Multiclass":
            n_classes = max(2, int(class_count))
            return base + [f"Class {i}" for i in range(n_classes)]
        return base

    def _counts_summary_row(
        self,
        data: pd.DataFrame,
        categorical_features: Sequence[str],
        quantitative_features: Sequence[str],
        class_count: int,
    ) -> List[float]:
        feature_count = data.shape[1] - 1
        if self.instance_label and self.instance_label in data.columns:
            feature_count -= 1
        if self.match_label and self.match_label in data.columns:
            feature_count -= 1

        total_missing = int(data.isnull().sum().sum())
        denom = max(1, data.shape[0] * max(1, feature_count))
        missing_percent = total_missing / float(denom)

        row: List[float] = [
            int(data.shape[0]),
            int(max(0, feature_count)),
            int(len(categorical_features)),
            int(len(quantitative_features)),
            int(total_missing),
            float(round(missing_percent, 5)),
        ]

        if self.outcome_type == "Binary":
            vc = data[self.outcome_label].value_counts(dropna=False)
            row.extend([int(vc.get(0, 0)), int(vc.get(1, 0))])
        elif self.outcome_type == "Multiclass":
            counts = data[self.outcome_label].value_counts(dropna=False).sort_index().tolist()
            n_classes = max(2, int(class_count))
            row.extend([int(counts[i]) if i < len(counts) else 0 for i in range(n_classes)])

        return row

    def _sync_feature_lists(
        self,
        data: pd.DataFrame,
        categorical_features: Sequence[str],
        quantitative_features: Sequence[str],
    ) -> Tuple[List[str], List[str]]:
        labels = {self.outcome_label, self.instance_label, self.match_label}
        labels = {x for x in labels if x}

        cat = [f for f in categorical_features if f in data.columns and f not in labels]
        quant = [f for f in quantitative_features if f in data.columns and f not in labels and f not in cat]

        # Any remaining numeric feature not already listed should be treated as quantitative.
        for col in data.columns:
            if col in labels or col in cat or col in quant:
                continue
            if is_numeric_dtype(data[col]):
                quant.append(col)
            else:
                cat.append(col)

        return cat, quant

    def _apply_ordinal_encoding(self, data: pd.DataFrame) -> pd.DataFrame:
        ord_map_path = self.train_root / "exploratory" / "ordinal_encoding.pickle"
        if not ord_map_path.exists():
            return data

        ord_map = _safe_pickle_load(ord_map_path, pd.DataFrame())
        if not isinstance(ord_map, pd.DataFrame) or ord_map.empty:
            return data

        for feat in ord_map.index:
            if feat not in data.columns:
                continue

            categories = ord_map.loc[feat, "Category"]
            encodings = ord_map.loc[feat, "Encoding"]
            if not isinstance(categories, (list, tuple)) or not isinstance(encodings, (list, tuple)):
                continue

            values = data[feat].dropna()
            if values.empty:
                continue

            # Skip if already encoded numerically with the same coding range.
            if is_numeric_dtype(values):
                try:
                    enc_set = {int(x) for x in encodings if x is not None}
                    val_set = {int(x) for x in values.astype(float).tolist()}
                    if val_set.issubset(enc_set):
                        continue
                except Exception:
                    pass

            mapping = {cat: enc for cat, enc in zip(categories, encodings)}
            before_non_na = int(data[feat].notna().sum())
            data[feat] = data[feat].map(mapping)
            after_non_na = int(data[feat].notna().sum())

            if after_non_na < before_non_na:
                logger.warning(
                    "Replication feature '%s' contained unseen labels; mapped to NaN for %d rows",
                    feat,
                    before_non_na - after_non_na,
                )

        return data

    def _apply_binary_consistency(self, data: pd.DataFrame) -> None:
        binary_map_path = self.train_root / "exploratory" / "binary_categorical_dict.pickle"
        binary_map = _safe_pickle_load(binary_map_path, {})
        if not isinstance(binary_map, dict):
            return

        for feat, train_values in binary_map.items():
            if feat not in data.columns:
                continue
            if not isinstance(train_values, (list, tuple, set)):
                continue
            train_set = set(train_values)
            observed = set(data[feat].dropna().unique().tolist())
            new_values = observed - train_set
            if new_values:
                logger.warning(
                    "Replication binary feature '%s' has unseen values; replacing with NaN: %s",
                    feat,
                    sorted(new_values),
                )
                data.loc[data[feat].isin(list(new_values)), feat] = np.nan

    def _drop_ignored_and_missing_outcome(self, data: pd.DataFrame) -> pd.DataFrame:
        cleaned = data.dropna(axis=0, how="any", subset=[self.outcome_label]).reset_index(drop=True)
        if self.ignore_features:
            cleaned = cleaned.drop(columns=[f for f in self.ignore_features if f in cleaned.columns], errors="ignore")

        # Match p1 behavior: cast classification outcome to int when possible.
        if self.outcome_type in {"Binary", "Multiclass"}:
            try:
                cleaned[self.outcome_label] = cleaned[self.outcome_label].astype("int64")
            except Exception:
                pass

        return cleaned

    def _apply_engineered_missingness(
        self,
        data: pd.DataFrame,
        categorical_features: Sequence[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        train_engineered = _safe_pickle_load(
            self.train_root / "exploratory" / "engineered_features.pickle",
            [],
        )
        if not isinstance(train_engineered, (list, tuple)):
            return data, list(categorical_features)

        cat = list(categorical_features)
        for source_feat in train_engineered:
            if source_feat in data.columns:
                miss_feat = f"Miss_{source_feat}"
                data[miss_feat] = data[source_feat].isnull().astype(int)
                if miss_feat not in cat:
                    cat.append(miss_feat)

        return data, cat

    def _drop_invariant_features(
        self,
        data: pd.DataFrame,
        categorical_features: Sequence[str],
        quantitative_features: Sequence[str],
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        invariant = [c for c in data.columns if data[c].nunique(dropna=True) <= 1 and c != self.outcome_label]
        if invariant:
            data = data.drop(columns=invariant, errors="ignore")
        cat = [c for c in categorical_features if c not in invariant]
        quant = [c for c in quantitative_features if c not in invariant]
        return data, cat, quant

    def _drop_training_removed_features(
        self,
        data: pd.DataFrame,
        categorical_features: Sequence[str],
        quantitative_features: Sequence[str],
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        removed = _safe_pickle_load(self.train_root / "exploratory" / "removed_features.pickle", [])
        if not isinstance(removed, (list, tuple)):
            removed = []

        removed_set = set(removed)
        data = data.drop(columns=[c for c in removed if c in data.columns], errors="ignore")
        cat = [c for c in categorical_features if c not in removed_set]
        quant = [c for c in quantitative_features if c not in removed_set]
        return data, cat, quant

    def _drop_high_missing_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        feature_count = data.shape[1] - 1
        if self.instance_label and self.instance_label in data.columns:
            feature_count -= 1
        if self.match_label and self.match_label in data.columns:
            feature_count -= 1

        threshold = int(self.cleaning_missingness * max(1, feature_count))
        if threshold <= 0:
            return data
        return data[data.isnull().sum(axis=1) < threshold].reset_index(drop=True)

    def _apply_one_hot_encoding(
        self,
        data: pd.DataFrame,
        categorical_features: Sequence[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        non_binary = [
            c
            for c in categorical_features
            if c in data.columns and data[c].nunique(dropna=True) > 2
        ]

        cat = list(categorical_features)
        if not non_binary:
            return data, cat

        one_hot_df = pd.get_dummies(data[non_binary], columns=non_binary)
        data = data.drop(columns=non_binary, errors="ignore")
        data = pd.concat([data, one_hot_df], axis=1)

        cat = [c for c in cat if c not in non_binary]
        cat.extend(list(one_hot_df.columns))
        return data, cat

    def _drop_training_correlated_features(
        self,
        data: pd.DataFrame,
        categorical_features: Sequence[str],
        quantitative_features: Sequence[str],
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        correlated = _safe_pickle_load(
            self.train_root / "exploratory" / "correlated_features.pickle",
            [],
        )
        if not isinstance(correlated, (list, tuple)):
            correlated = []

        correlated_set = set(correlated)
        data = data.drop(columns=[c for c in correlated if c in data.columns], errors="ignore")
        cat = [c for c in categorical_features if c not in correlated_set]
        quant = [c for c in quantitative_features if c not in correlated_set]
        return data, cat, quant

    def _align_to_training_processed_features(
        self,
        data: pd.DataFrame,
        categorical_features: Sequence[str],
        quantitative_features: Sequence[str],
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        post_processed = _safe_pickle_load(
            self.train_root / "exploratory" / "post_processed_features.pickle",
            [],
        )

        if not isinstance(post_processed, (list, tuple)) or len(post_processed) == 0:
            post_processed = list(data.columns)

        # Ensure labels are present in final schema.
        for lbl in (self.outcome_label, self.instance_label, self.match_label):
            if lbl and lbl not in post_processed:
                post_processed = [lbl] + list(post_processed)

        for feat in post_processed:
            if feat not in data.columns:
                data[feat] = 0

        data = data[[c for c in post_processed if c in data.columns]].copy()

        cat, quant = self._sync_feature_lists(data, categorical_features, quantitative_features)
        return data, cat, quant

    # ------------------------------------------------------------------
    # Artifact writing for exploratory outputs
    # ------------------------------------------------------------------

    def _write_processed_dataset(
        self,
        processed: pd.DataFrame,
        categorical_features: Sequence[str],
        quantitative_features: Sequence[str],
        transition_df: pd.DataFrame,
    ) -> None:
        self.exploratory_dir.mkdir(parents=True, exist_ok=True)
        (self.exploratory_dir / "initial").mkdir(parents=True, exist_ok=True)

        transition_df.to_csv(self.exploratory_dir / "DataProcessSummary.csv", index=True)

        with (self.exploratory_dir / "categorical_features.pickle").open("wb") as f:
            pickle.dump(list(categorical_features), f)

        with (self.exploratory_dir / "post_processed_features.pickle").open("wb") as f:
            pickle.dump(list(processed.columns), f)

        # p1 format: one-row CSV with feature names only (no labels).
        feature_headers = [c for c in processed.columns if c not in self._label_columns(processed)]
        pd.DataFrame([feature_headers]).to_csv(
            self.exploratory_dir / "ProcessedFeatureNames.csv",
            index=False,
            header=False,
        )

        processed.to_csv(self.rep_root / f"{self.apply_name}_Processed.csv", index=False)

        # Preserve initial feature-type artifacts for compatibility.
        initial_cat = _safe_pickle_load(
            self.train_root / "exploratory" / "initial" / "initial_categorical_features.pickle",
            [],
        )
        initial_quant = _safe_pickle_load(
            self.train_root / "exploratory" / "initial" / "initial_quantitative_features.pickle",
            [],
        )
        with (self.exploratory_dir / "initial" / "initial_categorical_features.pickle").open("wb") as f:
            pickle.dump([c for c in initial_cat if c in processed.columns], f)
        with (self.exploratory_dir / "initial" / "initial_quantitative_features.pickle").open("wb") as f:
            pickle.dump([c for c in initial_quant if c in processed.columns], f)

    def _write_eda_artifacts(
        self,
        processed: pd.DataFrame,
        categorical_features: Sequence[str],
        quantitative_features: Sequence[str],
    ) -> None:
        """Generate p1-style exploratory CSV outputs used downstream by reporting."""
        experiment_path = str(self.train_root / "replication")

        eda = DataProcess(
            data=processed.copy(),
            experiment_path=experiment_path,
            outcome_label=self.outcome_label,
            match_label=self.match_label,
            instance_label=self.instance_label,
            categorical_cutoff=self.categorical_cutoff,
            sig_cutoff=self.sig_cutoff,
            random_state=self.random_state,
            show_plots=self.show_plots,
            dataset_name=self.apply_name,
            enable_plots=False,
        )
        eda.outcome_type = self.outcome_type
        eda.categorical_features = list(categorical_features)
        eda.quantitative_features = list(quantitative_features)
        eda.make_log_folders()

        eda.describe_data()
        total_missing = eda.missingness_counts()
        eda.counts_summary(total_missing=total_missing, save=True, replicate=True)

        if "feature_correlations" not in self.exclude_plots:
            try:
                eda.feature_correlation(x_data=eda.feature_only_data())
            except Exception as exc:
                logger.warning("Failed to compute feature correlation for replication set: %s", exc)

        try:
            eda.univariate_analysis(top_features=20)
        except Exception as exc:
            logger.warning("Failed to compute univariate analysis for replication set: %s", exc)

    # ------------------------------------------------------------------
    # p2/p6 replay and p7 replication evaluation
    # ------------------------------------------------------------------

    def _evaluate_models(
        self,
        processed: pd.DataFrame,
        categorical_features: Sequence[str],
        quantitative_features: Sequence[str],
        fold_map: Sequence[Tuple[int, int]],
    ) -> None:
        if not self.algorithms:
            raise Exception("No trained algorithms found to evaluate on replication dataset")

        for rep_cv_idx, source_cv_idx in fold_map:
            train_cv_path = self.train_root / "CVDatasets" / f"{self.train_name}_CV_{source_cv_idx}_Train.csv"
            if not train_cv_path.exists():
                raise Exception(
                    "Strict fold parity failed: missing training CV fold file "
                    f"{train_cv_path}"
                )

            train_cv_df = pd.read_csv(train_cv_path, na_values="NA", sep=",")
            rep_cv_df = processed.copy()

            feature_columns = [
                c
                for c in train_cv_df.columns
                if c not in self._label_columns(train_cv_df)
            ]

            if self.impute_data:
                rep_cv_df = self._apply_imputation(
                    rep_cv_df,
                    feature_columns,
                    source_cv_idx,
                    train_cv_df,
                    categorical_features,
                    quantitative_features,
                )

            if self.scale_data:
                rep_cv_df = self._apply_scaling(rep_cv_df, feature_columns, source_cv_idx, train_cv_df)

            # Align exactly to training fold columns.
            missing_cols = [col for col in train_cv_df.columns if col not in rep_cv_df.columns]
            if missing_cols:
                filler = pd.DataFrame(0, index=rep_cv_df.index, columns=missing_cols)
                rep_cv_df = pd.concat([rep_cv_df, filler], axis=1)
            rep_cv_df = rep_cv_df[list(train_cv_df.columns)].copy()

            # Persist CV artifacts in replication dataset namespace.
            rep_test_path = self.cv_dir / f"{self.apply_name}_CV_{rep_cv_idx}_Test.csv"
            rep_train_path = self.cv_dir / f"{self.apply_name}_CV_{rep_cv_idx}_Train.csv"
            rep_cv_df.to_csv(rep_test_path, index=False)
            shutil.copy2(train_cv_path, rep_train_path)

            eval_df = rep_cv_df.copy()
            if self.instance_label and self.instance_label in eval_df.columns:
                eval_df = eval_df.drop(columns=[self.instance_label])

            if self.outcome_label not in eval_df.columns:
                raise Exception(
                    "Strict fold parity failed: outcome label "
                    f"'{self.outcome_label}' missing in replication fold {rep_cv_idx}"
                )

            x_test = eval_df.drop(columns=[self.outcome_label]).values
            y_test = eval_df[self.outcome_label].values

            for algorithm in self.algorithms:
                small = self.abbrev.get(algorithm, algorithm)
                model_path_candidates = [
                    self.train_root / "models" / "pickledModels" / f"{small}_{source_cv_idx}.pickle",
                    self.train_root / "models" / "pickledModels" / f"{algorithm}_{source_cv_idx}.pickle",
                ]
                model_path = next((p for p in model_path_candidates if p.exists()), None)
                if model_path is None:
                    raise Exception(
                        "Strict fold parity failed: missing trained model pickle for "
                        f"algorithm={algorithm}, source_cv={source_cv_idx}"
                    )

                with model_path.open("rb") as f:
                    model = pickle.load(f)

                fi_list = self._load_training_feature_importance(
                    small_name=small,
                    source_cv_idx=source_cv_idx,
                    expected_len=x_test.shape[1],
                )

                if self.outcome_type == "Binary":
                    evaluator = BinaryClassificationModel(None, algorithm, scoring_metric=self.scoring_metric)
                    evaluator.model = model
                    metrics_dict, curves_dict = evaluator.model_evaluation(x_test, y_test)
                    self._write_base_outputs(rep_cv_idx, small, metrics_dict, curves_dict, fi_list)
                elif self.outcome_type == "Multiclass":
                    evaluator = MulticlassClassificationModel(None, algorithm, scoring_metric=self.scoring_metric)
                    evaluator.model = model
                    metrics_dict, curves_dict = evaluator.model_evaluation(x_test, y_test)
                    self._write_base_outputs(rep_cv_idx, small, metrics_dict, curves_dict, fi_list)
                else:
                    evaluator = RegressionModel(None, algorithm, scoring_metric=self.scoring_metric)
                    evaluator.model = model
                    metrics_dict = evaluator.model_evaluation(x_test, y_test)
                    y_pred = evaluator.predict(x_test)
                    residual_test = y_test - y_pred
                    self._write_base_outputs(
                        rep_cv_idx,
                        small,
                        metrics_dict,
                        None,
                        fi_list,
                        residual_test=residual_test,
                        y_pred=y_pred,
                        y_true=y_test,
                    )

    def _label_columns(self, df: pd.DataFrame) -> List[str]:
        labels = [self.outcome_label]
        if self.instance_label and self.instance_label in df.columns:
            labels.append(self.instance_label)
        if self.match_label and self.match_label in df.columns:
            labels.append(self.match_label)
        return labels

    def _apply_imputation(
        self,
        data: pd.DataFrame,
        feature_columns: Sequence[str],
        source_cv_idx: int,
        train_cv_df: pd.DataFrame,
        categorical_features: Sequence[str],
        quantitative_features: Sequence[str],
    ) -> pd.DataFrame:
        active_features = [c for c in feature_columns if c in data.columns and c in train_cv_df.columns]
        if not active_features:
            return data

        # 1) Categorical mode imputer from p2
        cat_path = self.train_root / "impute_scale" / f"categorical_imputer_cv{source_cv_idx}.pickle"
        cat_imputer = _safe_pickle_load(cat_path, {})
        if isinstance(cat_imputer, dict):
            for c, fill_val in cat_imputer.items():
                if c in active_features:
                    data[c] = data[c].fillna(fill_val)

        # 2) Numeric imputer from p2
        ord_path = self.train_root / "impute_scale" / f"ordinal_imputer_cv{source_cv_idx}.pickle"
        ord_imputer = _safe_pickle_load(ord_path, None)

        transformed = False
        if ord_imputer is not None and hasattr(ord_imputer, "transform"):
            try:
                x = data[active_features].copy()
                expected_features = getattr(ord_imputer, "feature_names_in_", None)
                if expected_features is not None:
                    expected_features = list(expected_features)
                    missing_expected = [c for c in expected_features if c not in x.columns]
                    if missing_expected:
                        raise ValueError(
                            f"Missing {len(missing_expected)} imputer-fit features (fallback imputation will be used)"
                        )
                    x = x[expected_features]

                xt = ord_imputer.transform(x)
                if isinstance(xt, pd.DataFrame):
                    for col in xt.columns:
                        if col in data.columns:
                            data[col] = xt[col].values
                else:
                    xt_df = pd.DataFrame(xt, columns=list(x.columns), index=data.index)
                    for col in xt_df.columns:
                        if col in data.columns:
                            data[col] = xt_df[col].values
                transformed = True
            except Exception as exc:
                logger.warning(
                    "Failed to apply ordinal imputer transform on replication CV %s: %s",
                    source_cv_idx,
                    exc,
                )

        if not transformed and isinstance(ord_imputer, dict):
            # Legacy median-dict style: {feature: value}
            if "id" not in ord_imputer:
                for c, fill_val in ord_imputer.items():
                    if c in active_features:
                        data[c] = data[c].fillna(fill_val)
                transformed = True

        # Fallback for remaining NaNs in active feature columns.
        if data[active_features].isnull().sum().sum() > 0:
            train_num = train_cv_df[active_features].copy()
            for col in active_features:
                if data[col].isnull().sum() == 0:
                    continue
                if col in categorical_features:
                    mode = train_num[col].mode(dropna=True)
                    if not mode.empty:
                        data[col] = data[col].fillna(mode.iloc[0])
                elif col in quantitative_features or is_numeric_dtype(train_num[col]):
                    data[col] = data[col].fillna(train_num[col].median())
                else:
                    mode = train_num[col].mode(dropna=True)
                    if not mode.empty:
                        data[col] = data[col].fillna(mode.iloc[0])

        return data

    def _apply_scaling(
        self,
        data: pd.DataFrame,
        feature_columns: Sequence[str],
        source_cv_idx: int,
        train_cv_df: pd.DataFrame,
    ) -> pd.DataFrame:
        active_features = [c for c in feature_columns if c in data.columns and c in train_cv_df.columns]
        if not active_features:
            return data

        scale_path = self.train_root / "impute_scale" / f"scaler_cv{source_cv_idx}.pickle"
        scaler = _safe_pickle_load(scale_path, None)

        if scaler is not None and hasattr(scaler, "transform"):
            try:
                x = data[active_features].copy()
                expected_features = getattr(scaler, "feature_names_in_", None)
                if expected_features is not None:
                    expected_features = list(expected_features)
                    missing_expected = [c for c in expected_features if c not in x.columns]
                    if missing_expected:
                        raise ValueError(
                            f"Missing {len(missing_expected)} scaler-fit features (fallback scaling will be used)"
                        )
                    x = x[expected_features]

                xt = scaler.transform(x)
                if isinstance(xt, pd.DataFrame):
                    for col in xt.columns:
                        if col in data.columns:
                            data[col] = xt[col].values
                else:
                    xt_df = pd.DataFrame(xt, columns=list(x.columns), index=data.index)
                    for col in xt_df.columns:
                        if col in data.columns:
                            data[col] = xt_df[col].values
                return data
            except Exception as exc:
                logger.warning(
                    "Failed to apply scaler transform on replication CV %s: %s",
                    source_cv_idx,
                    exc,
                )

        # Fallback: if scaler object was not serializable with fit-state, use train-fold z-score.
        train_x = train_cv_df[active_features].copy()
        for col in active_features:
            if not is_numeric_dtype(train_x[col]):
                continue
            mean = pd.to_numeric(train_x[col], errors="coerce").mean()
            std = pd.to_numeric(train_x[col], errors="coerce").std(ddof=0)
            if std is None or np.isnan(std) or std == 0:
                continue
            data[col] = (pd.to_numeric(data[col], errors="coerce") - mean) / std

        return data

    def _load_training_feature_importance(

        self,
        small_name: str,
        source_cv_idx: int,
        expected_len: int,
    ) -> List[float]:
        metrics_path = self.train_root / "model_evaluation" / "metrics_by_cv" / f"{small_name}_CV_{source_cv_idx}.json"
        if not metrics_path.exists():
            return [0.0] * int(expected_len)

        try:
            with metrics_path.open("r") as f:
                payload = json.load(f)
            fi = payload.get("feature_importance", [])
            fi = [float(x) for x in fi]
        except Exception:
            fi = [0.0] * int(expected_len)

        if len(fi) < expected_len:
            fi = fi + [0.0] * (expected_len - len(fi))
        elif len(fi) > expected_len:
            fi = fi[:expected_len]

        return fi

    def _write_base_outputs(
        self,
        rep_cv_idx: int,
        small_name: str,
        metrics_dict: Dict[str, Any],
        curves_dict: Optional[Dict[str, Any]],
        fi_list: Sequence[float],
        residual_test: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None,
    ) -> None:
        payload = {
            "metrics": _jsonify(metrics_dict),
            "feature_importance": _jsonify(list(fi_list)),
        }
        with (self.model_metrics_dir / f"{small_name}_CV_{rep_cv_idx}.json").open("w") as f:
            json.dump(payload, f, indent=2)

        if curves_dict:
            roc = curves_dict.get("roc", {})
            prc = curves_dict.get("prc", {})
            with (self.model_curves_dir / f"{small_name}_CV_{rep_cv_idx}_roc.json").open("w") as f:
                json.dump(_jsonify(roc), f, indent=2)
            with (self.model_curves_dir / f"{small_name}_CV_{rep_cv_idx}_prc.json").open("w") as f:
                json.dump(_jsonify(prc), f, indent=2)

        if self.outcome_type == "Continuous" and residual_test is not None and y_pred is not None and y_true is not None:
            # Keep p6 payload shape for compatibility with p8 residual plotting.
            residual_payload = [
                np.array([], dtype=float),           # train residual (not available in replication)
                np.asarray(residual_test, dtype=float),
                np.array([], dtype=float),           # train predictions (not available)
                np.asarray(y_pred, dtype=float),
                np.array([], dtype=float),           # y_train (not available)
                np.asarray(y_true, dtype=float),
            ]
            with (self.model_pickled_metrics_dir / f"{small_name}_CV_{rep_cv_idx}_residuals.pickle").open("wb") as f:
                pickle.dump(residual_payload, f)

    def _evaluate_ensembles(
        self,
        processed: pd.DataFrame,
        fold_map: Sequence[Tuple[int, int]],
    ) -> None:
        src_pickled = self.train_root / "ensemble_evaluation" / "pickled_ensembles"
        if not src_pickled.exists():
            return

        source_fold_ids = [source_cv for _, source_cv in fold_map]
        ensemble_fold_map: Dict[str, set] = {}
        for ens_pickle in sorted(src_pickled.glob("*.pickle")):
            match = re.match(r"(.+?)_(\d+)\.pickle$", ens_pickle.name)
            if not match:
                continue
            ens_id = match.group(1)
            fold_id = int(match.group(2))
            ensemble_fold_map.setdefault(ens_id, set()).add(fold_id)

        # If ensemble pickles are present, enforce strict fold parity for each ensemble id.
        for ens_id, folds in ensemble_fold_map.items():
            missing_folds = [cv for cv in source_fold_ids if cv not in folds]
            if missing_folds:
                raise Exception(
                    "Strict fold parity failed: missing ensemble pickles for "
                    f"ensemble={ens_id}, folds={missing_folds}"
                )

        self.ensemble_metrics_dir.mkdir(parents=True, exist_ok=True)
        self.ensemble_curves_dir.mkdir(parents=True, exist_ok=True)
        self.ensemble_pickled_dir.mkdir(parents=True, exist_ok=True)

        for rep_cv_idx, source_cv_idx in fold_map:
            rep_test_path = self.cv_dir / f"{self.apply_name}_CV_{rep_cv_idx}_Test.csv"
            if not rep_test_path.exists():
                raise Exception(
                    "Strict fold parity failed: missing replication CV test file "
                    f"{rep_test_path}"
                )

            test_df = pd.read_csv(rep_test_path, na_values="NA", sep=",")
            eval_df = test_df.copy()
            if self.instance_label and self.instance_label in eval_df.columns:
                eval_df = eval_df.drop(columns=[self.instance_label])

            if self.outcome_label not in eval_df.columns:
                raise Exception(
                    "Strict fold parity failed: outcome label "
                    f"'{self.outcome_label}' missing in replication fold {rep_cv_idx}"
                )

            x_test = eval_df.drop(columns=[self.outcome_label]).values
            y_test = eval_df[self.outcome_label].values

            for ens_id in sorted(ensemble_fold_map.keys()):
                ens_pickle = src_pickled / f"{ens_id}_{source_cv_idx}.pickle"
                if not ens_pickle.exists():
                    raise Exception(
                        "Strict fold parity failed: missing ensemble pickle "
                        f"{ens_pickle}"
                    )

                with ens_pickle.open("rb") as f:
                    model = pickle.load(f)

                # Keep a copy of ensemble pickle for traceability in replication outputs.
                dst_pickle = self.ensemble_pickled_dir / f"{ens_id}_{rep_cv_idx}.pickle"
                with dst_pickle.open("wb") as f:
                    pickle.dump(model, f)

                y_pred = model.predict(x_test)
                metrics = _calc_basic_metrics(y_test, y_pred)

                roc_curve_dict = None
                prc_curve_dict = None
                roc_auc_val = None
                prc_auc_val = None
                aps_val = None
                brier_val = None

                proba = None
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(x_test)
                    except Exception as exc:
                        logger.warning("predict_proba failed for ensemble %s: %s", ens_id, exc)
                        proba = None
                elif hasattr(model, "decision_function"):
                    try:
                        score = np.asarray(model.decision_function(x_test))
                        if score.ndim == 1:
                            score = (score - score.min()) / (score.max() - score.min() + 1e-12)
                            proba = np.column_stack([1.0 - score, score])
                        else:
                            s_min = score.min(axis=0, keepdims=True)
                            s_max = score.max(axis=0, keepdims=True)
                            proba = (score - s_min) / (s_max - s_min + 1e-12)
                    except Exception as exc:
                        logger.warning("decision_function failed for ensemble %s: %s", ens_id, exc)
                        proba = None

                if proba is not None:
                    classes = getattr(model, "classes_", None)
                    roc_curve_dict, prc_curve_dict, roc_auc_val, prc_auc_val, aps_val = _calc_curves_scores_from_proba(
                        y_test,
                        proba,
                        classes=classes,
                    )

                    try:
                        proba_arr = np.asarray(proba)
                        unique_classes = np.unique(y_test)
                        if proba_arr.ndim == 1 and len(unique_classes) == 2:
                            brier_val = brier_score_loss(y_test, proba_arr)
                        elif proba_arr.ndim == 2 and len(unique_classes) == 2 and proba_arr.shape[1] >= 2:
                            brier_val = brier_score_loss(y_test, proba_arr[:, 1])
                        elif proba_arr.ndim == 2 and len(unique_classes) > 2:
                            brier_val = multiclass_brier_score(y_test, proba_arr)
                    except Exception:
                        brier_val = None

                if roc_auc_val is not None:
                    metrics["ROC AUC"] = float(roc_auc_val)
                if prc_auc_val is not None:
                    metrics["PRC AUC"] = float(prc_auc_val)
                if aps_val is not None:
                    metrics["PRC APS"] = float(aps_val)
                if brier_val is not None:
                    metrics["Brier Score"] = float(brier_val)

                with (self.ensemble_metrics_dir / f"{ens_id}_CV_{rep_cv_idx}.json").open("w") as f:
                    json.dump(_jsonify(metrics), f, indent=2)

                if roc_curve_dict:
                    with (self.ensemble_curves_dir / f"{ens_id}_CV_{rep_cv_idx}_roc.json").open("w") as f:
                        json.dump(_jsonify(roc_curve_dict), f, indent=2)
                if prc_curve_dict:
                    with (self.ensemble_curves_dir / f"{ens_id}_CV_{rep_cv_idx}_prc.json").open("w") as f:
                        json.dump(_jsonify(prc_curve_dict), f, indent=2)

    # ------------------------------------------------------------------
    # p8 on replication outputs
    # ------------------------------------------------------------------

    def _run_statistics(self, cv_partitions: int) -> None:
        if self.outcome_type == "Continuous":
            metric_weight = "explained_variance"
        else:
            metric_weight = "balanced_accuracy"

        # StatisticsPhaseJob writes completion flags under <dataset_parent>/jobsCompleted.
        (self.rep_root.parent / "jobsCompleted").mkdir(parents=True, exist_ok=True)

        scoring_metric = self.scoring_metric
        if self.outcome_type == "Continuous":
            scoring_metric = "explained_variance"

        stats = StatisticsPhaseJob(
            full_path=str(self.rep_root),
            outcome_label=self.outcome_label,
            outcome_type=self.outcome_type,
            instance_label=self.instance_label,
            scoring_metric=scoring_metric,
            cv_partitions=cv_partitions,
            top_features=40,
            sig_cutoff=self.sig_cutoff,
            metric_weight=metric_weight,
            scale_data=self.scale_data,
            exclude_plots=self.exclude_plots,
            show_plots=self.show_plots,
            include_ensembles=self.outcome_type in {"Binary", "Multiclass"},
        )
        stats.run()
