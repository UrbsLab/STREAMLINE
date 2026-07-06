# streamline/p4_feature_importance/importance.py
from __future__ import annotations
import os, time, json, pickle, random, logging
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from streamline.p4_feature_importance.utils.fi_loader import (
    load_importance,
    normalize_importance_key,
    resolve_importance_id,
)

REBATE_MODEL_IDS = {"multisurf", "multisurfstar", "multiswrfdb", "multiswrfdbstar"}
OUTCOME_TYPE_TO_REBATE_LABEL = {
    "binary": "binary",
    "multiclass": "multiclass",
    "continuous": "continuous",
    "regression": "continuous",
}

class FeatureImportance:
    """
    Run a single feature-importance model on one CV pair.
    Saves:
      - feature_importance/<path_name>/<path_name>_scores_cv_<k>.csv
      - feature_importance/<path_name>/selector_cv<k>.pickle ({id, params})
      - jobsCompleted flag
    Optionally writes model-specific selected CV copies if top_k/threshold supplied.
    """
    def __init__(
        self,
        cv_train_path: str,
        cv_test_path: str,
        experiment_path: str,
        *,
        model_id: str,
        model_params: Dict[str, Any] | None = None,
        top_k: "int | None" = None,
        threshold: "float | None" = None,
        keep_original_features: bool = False,
        overwrite_cv: bool = True,
        outcome_label: str = "Class",
        outcome_type: Optional[str] = None,      # for MI
        instance_label: Optional[str] = None,
        random_state: Optional[int] = None,
        instance_subset: int | None = None,
    ):
        self.cv_train_path = cv_train_path
        self.cv_test_path = cv_test_path
        self.experiment_path = experiment_path
        self.model_id = model_id
        self.model_params = model_params or {}
        self.top_k = top_k
        self.threshold = threshold
        self.keep_original_features = keep_original_features
        self.overwrite_cv = overwrite_cv
        self.outcome_label = outcome_label
        self.outcome_type = outcome_type
        self.instance_label = instance_label
        self.random_state = random_state
        self.instance_subset = int(instance_subset) if instance_subset not in (None, "") else None

        self.dataset_name: Optional[str] = None
        self.cv_count: Optional[str] = None
        self.job_start_time = time.time()

    def run(self):
        random.seed(self.random_state); np.random.seed(self.random_state)
        tr, te = self._load_data()
        logging.info("Prepared Train and Test for: %s_CV_%s", self.dataset_name, self.cv_count)

        y_tr = tr[self.outcome_label]; y_te = te[self.outcome_label]
        if self.instance_label and self.instance_label in tr.columns:
            inst_tr = tr[self.instance_label]; inst_te = te[self.instance_label]
        else:
            inst_tr = inst_te = None

        drop_cols = [self.outcome_label] + ([self.instance_label] if self.instance_label in tr.columns else [])
        Xtr = tr.drop(columns=drop_cols, errors="ignore")
        Xte = te.drop(columns=drop_cols, errors="ignore")

        params = dict(self.model_params)
        rebate_categorical_names: List[str] = []
        rebate_categorical_indices: List[int] = []
        if self.model_id == "mutualinformation" and self.outcome_type and "outcome_type" not in params:
            params["outcome_type"] = self.outcome_type
        if self.is_rebate_model(self.model_id):
            params, rebate_categorical_names, rebate_categorical_indices = self.configure_rebate_params(
                params,
                list(Xtr.columns),
            )

        logging.info("Running %s...", self.model_id)
        model = load_importance(self.model_id, **params)
        X_fit, y_fit = self.sample_instances_for_model(model, Xtr, y_tr)
        model.fit(X_fit, y_fit)

        # write scores CSV for this model
        logging.info("Sort and pickle feature importance scores...")
        self._write_scores_csv(model, Xtr.columns)

        # save params artifact
        base = self._model_dir(model)
        selector_payload = {
            "id": getattr(model, "id", self.model_id),
            "params": model.get_params(),
            "model_name": getattr(model, "model_name", self.model_id),
            "small_name": getattr(model, "small_name", self.model_id),
            "instance_subset": self.instance_subset,
            "instances_fit": int(len(X_fit)),
        }
        if self.is_rebate_model(self.model_id):
            selector_payload["categorical_features"] = rebate_categorical_names
            selector_payload["categorical_feature_indices"] = rebate_categorical_indices
        with open(os.path.join(base, f"selector_cv{self.cv_count}.pickle"), "wb") as f:
            pickle.dump(selector_payload, f)

        if self.top_k is not None or self.threshold is not None:
            Xtr_sel = model.transform(Xtr, top_k=self.top_k, threshold=self.threshold)
            Xte_sel = Xte.loc[:, Xtr_sel.columns]
            if self.keep_original_features:
                Xtr_out = pd.concat([Xtr.reset_index(drop=True), Xtr_sel.reset_index(drop=True)], axis=1)
                Xte_out = pd.concat([Xte.reset_index(drop=True), Xte_sel.reset_index(drop=True)], axis=1)
            else:
                Xtr_out, Xte_out = Xtr_sel, Xte_sel

            if inst_tr is None:
                train_out = pd.concat([y_tr.reset_index(drop=True), Xtr_out], axis=1)
                test_out  = pd.concat([y_te.reset_index(drop=True), Xte_out], axis=1)
            else:
                train_out = pd.concat([y_tr.reset_index(drop=True), inst_tr.reset_index(drop=True), Xtr_out], axis=1)
                test_out  = pd.concat([y_te.reset_index(drop=True), inst_te.reset_index(drop=True), Xte_out], axis=1)

            self._write_selected_cv_files(model, train_out, test_out)

        self._save_runtime(model)
        self._complete_flag(model)
        logging.info(
            "%s CV%s phase 4 %s evaluation complete",
            self.dataset_name,
            self.cv_count,
            self.model_id,
        )

    # ---- helpers ----
    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.dataset_name = self.cv_train_path.split('/')[-3]
        self.cv_count = self.cv_train_path.split('/')[-1].split("_")[-2]
        logging.info("-------------------------------------------------------")
        logging.info("Loading Dataset: %s_CV_%s_Train", self.dataset_name, self.cv_count)
        tr = pd.read_csv(self.cv_train_path, na_values='NA', sep=',')
        te = pd.read_csv(self.cv_test_path, na_values='NA', sep=',')
        return tr, te

    @staticmethod
    def is_rebate_model(model_id: str) -> bool:
        resolved = resolve_importance_id(model_id) or model_id
        return resolved in REBATE_MODEL_IDS or normalize_importance_key(model_id) in {
            "multisurf",
            "multisurfstar",
            "multiswrfdb",
            "multiswrfdbstar",
        }

    def load_categorical_feature_names(self) -> List[str]:
        path = os.path.join(
            self.experiment_path,
            self.dataset_name,
            "exploratory",
            "categorical_features.pickle",
        )
        if not os.path.exists(path):
            logging.warning(
                "STREAMLINE categorical feature list not found at %s; passing an empty "
                "categorical_features list to %s.",
                path,
                self.model_id,
            )
            return []
        try:
            with open(path, "rb") as f:
                values = pickle.load(f) or []
        except Exception as e:
            logging.warning(
                "Could not read STREAMLINE categorical feature list at %s (%s); passing "
                "an empty categorical_features list to %s.",
                path,
                e,
                self.model_id,
            )
            return []
        return [str(v) for v in values if str(v).strip()]

    def categorical_feature_indices(self, feature_columns: List[str]) -> Tuple[List[str], List[int]]:
        categorical_names = self.load_categorical_feature_names()
        categorical_set = set(categorical_names)
        indices = [idx for idx, col in enumerate(feature_columns) if col in categorical_set]
        kept_names = [feature_columns[idx] for idx in indices]
        logging.info(
            "Phase 4 %s passing %d STREAMLINE categorical feature index(es) to ReBATE.",
            self.model_id,
            len(indices),
        )
        return kept_names, indices

    def rebate_label_type(self) -> Optional[str]:
        if not self.outcome_type:
            return None
        return OUTCOME_TYPE_TO_REBATE_LABEL.get(str(self.outcome_type).strip().lower())

    def configure_rebate_params(self, params: Dict[str, Any], feature_columns: List[str]):
        params = dict(params)
        categorical_names, categorical_indices = self.categorical_feature_indices(feature_columns)
        if "categorical_features" in params and list(params.get("categorical_features") or []) != categorical_indices:
            logging.warning(
                "Ignoring user-supplied categorical_features for %s; using STREAMLINE's "
                "categorical_features.pickle list.",
                self.model_id,
            )
        params["categorical_features"] = categorical_indices
        label_type = self.rebate_label_type()
        if label_type and "label_type" not in params:
            params["label_type"] = label_type
        return params, categorical_names, categorical_indices

    @staticmethod
    def uses_instance_subset(model, model_id: str) -> bool:
        identifiers = {
            model_id,
            getattr(model, "id", ""),
            getattr(model, "path_name", ""),
            getattr(model, "small_name", ""),
            getattr(model, "model_name", ""),
        }
        normalized = {
            str(identifier).lower().replace(" ", "").replace("_", "").replace("-", "").replace("*", "star")
            for identifier in identifiers
            if str(identifier).strip()
        }
        subset_ids = {
            "multisurf", "ms",
            "multisurfstar", "mss",
            "multiswrfdb", "mswrfdb",
            "multiswrfdbstar", "mswrfdbstar",
        }
        return bool(getattr(model, "uses_instance_subset", False)) or bool(normalized.intersection(subset_ids))

    def sample_instances_for_model(self, model, Xtr: pd.DataFrame, y_tr: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        if not self.uses_instance_subset(model, self.model_id):
            return Xtr, y_tr
        if self.instance_subset is None or int(self.instance_subset) <= 0:
            return Xtr, y_tr
        n = min(len(Xtr), int(self.instance_subset))
        if n >= len(Xtr):
            return Xtr, y_tr
        idx = Xtr.sample(n, random_state=self.random_state).index
        logging.info(
            "Phase 4 %s using instance_subset=%s (%s of %s training instances).",
            self.model_id,
            self.instance_subset,
            n,
            len(Xtr),
        )
        return Xtr.loc[idx], y_tr.loc[idx]

    def _model_dir(self, model) -> str:
        path_name = getattr(model, "path_name", self.model_id)
        out_dir = os.path.join(self.experiment_path, self.dataset_name, "feature_importance", path_name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _write_scores_csv(self, model, in_cols):
        import pandas as pd
        base = self._model_dir(model)
        path_name = getattr(model, "path_name", self.model_id)
        out_csv = os.path.join(base, f"{path_name}_scores_cv_{self.cv_count}.csv")
        scores = model.get_scores()
        df = pd.DataFrame({
            "feature": list(in_cols),
            "score": [float(scores.get(c, 0.0)) for c in in_cols]
        }).sort_values("score", ascending=False)
        df.to_csv(out_csv, index=False)

    def _write_cv_files(self, train: pd.DataFrame, test: pd.DataFrame):
        if self.overwrite_cv:
            os.remove(self.cv_train_path); os.remove(self.cv_test_path)
        else:
            cvdir = os.path.join(self.experiment_path, self.dataset_name, "CVDatasets")
            os.rename(self.cv_train_path, os.path.join(cvdir, f"{self.dataset_name}_CVOnly_{self.cv_count}_Train.csv"))
            os.rename(self.cv_test_path,  os.path.join(cvdir, f"{self.dataset_name}_CVOnly_{self.cv_count}_Test.csv"))
        train.to_csv(self.cv_train_path, index=False)
        test.to_csv(self.cv_test_path, index=False)

    def _write_selected_cv_files(self, model, train: pd.DataFrame, test: pd.DataFrame):
        path_name = getattr(model, "path_name", self.model_id)
        out_dir = os.path.join(
            self.experiment_path,
            self.dataset_name,
            "feature_importance",
            path_name,
            "selected_cv",
        )
        os.makedirs(out_dir, exist_ok=True)
        train.to_csv(os.path.join(out_dir, f"{self.dataset_name}_CV_{self.cv_count}_Train.csv"), index=False)
        test.to_csv(os.path.join(out_dir, f"{self.dataset_name}_CV_{self.cv_count}_Test.csv"), index=False)

    def _save_runtime(self, model):
        rt = os.path.join(self.experiment_path, self.dataset_name, "runtime")
        os.makedirs(rt, exist_ok=True)
        elapsed = str(time.time() - self.job_start_time)
        path_name = getattr(model, "path_name", self.model_id)
        with open(os.path.join(rt, f"runtime_feature_importance_{path_name}_cv{self.cv_count}.txt"), "w+") as f:
            f.write(elapsed)
        with open(os.path.join(rt, f"runtime_feature_importance{self.cv_count}.txt"), "w+") as f:
            f.write(elapsed)

    def _complete_flag(self, model):
        done = os.path.join(self.experiment_path, "jobsCompleted")
        os.makedirs(done, exist_ok=True)
        path_name = getattr(model, "path_name", self.model_id)
        with open(os.path.join(done, f"job_feature_importance_{path_name}_{self.dataset_name}_{self.cv_count}.txt"), "w") as f:
            f.write("complete")
