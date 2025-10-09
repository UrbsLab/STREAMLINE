# streamline/phases/p4_feature_importance/job.py
from __future__ import annotations
import os, time, json, pickle, random, logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from streamline.p4_feature_importance.utils.fi_loader import load_importance

class FeatureImportance:
    """
    Phase 4: Feature Importance using MultiSURF / MultiSURF* / Mutual Information.
    """
    def __init__(
        self,
        cv_train_path: str,
        cv_test_path: str,
        experiment_path: str,
        *,
        importance_id: str = "mutualinformation",
        importance_params: "Dict[str, Any] | None" = None,
        top_k: "int | None" = None,
        threshold: "float | None" = None,
        keep_original_features: bool = False,
        overwrite_cv: bool = True,
        outcome_label: str = "Class",
        outcome_type: Optional[str] = None,      # "Binary" | "Multiclass" | "Continuous" (for MI)
        instance_label: Optional[str] = None,
        random_state: Optional[int] = None,
    ):
        self.cv_train_path = cv_train_path
        self.cv_test_path = cv_test_path
        self.experiment_path = experiment_path
        self.importance_id = importance_id
        self.importance_params = importance_params or {}
        self.top_k = top_k
        self.threshold = threshold
        self.keep_original_features = keep_original_features
        self.overwrite_cv = overwrite_cv
        self.outcome_label = outcome_label
        self.outcome_type = outcome_type
        self.instance_label = instance_label
        self.random_state = random_state

        self.dataset_name: Optional[str] = None
        self.cv_count: Optional[str] = None
        self.job_start_time = time.time()

    def run(self):
        random.seed(self.random_state); np.random.seed(self.random_state)

        tr, te = self._load_data()
        y_tr = tr[self.outcome_label]; y_te = te[self.outcome_label]

        inst_tr = inst_te = None
        if self.instance_label is not None and self.instance_label in tr.columns:
            inst_tr = tr[self.instance_label]; inst_te = te[self.instance_label]

        drop_cols = [self.outcome_label] + ([self.instance_label] if self.instance_label in tr.columns else [])
        Xtr = tr.drop(columns=drop_cols, errors="ignore")
        Xte = te.drop(columns=drop_cols, errors="ignore")

        # If MI and outcome_type is not given, try metadata default later (runner wires it)
        params = dict(self.importance_params)
        if self.importance_id == "mutualinformation" and self.outcome_type and "outcome_type" not in params:
            params["outcome_type"] = self.outcome_type

        sel = load_importance(self.importance_id, **params).fit(Xtr, y_tr)

        # selected subset
        Xtr_sel = sel.transform(Xtr, top_k=self.top_k, threshold=self.threshold)
        Xte_sel = Xte.loc[:, Xtr_sel.columns]  # align columns

        if self.keep_original_features:
            Xtr_out = pd.concat([Xtr.reset_index(drop=True), Xtr_sel.reset_index(drop=True)], axis=1)
            Xte_out = pd.concat([Xte.reset_index(drop=True), Xte_sel.reset_index(drop=True)], axis=1)
        else:
            Xtr_out, Xte_out = Xtr_sel, Xte_sel

        # reassemble
        if self.instance_label is None or self.instance_label not in tr.columns:
            train_out = pd.concat([y_tr.reset_index(drop=True), Xtr_out], axis=1)
            test_out  = pd.concat([y_te.reset_index(drop=True),  Xte_out], axis=1)
        else:
            train_out = pd.concat([y_tr.reset_index(drop=True), inst_tr.reset_index(drop=True), Xtr_out], axis=1)
            test_out  = pd.concat([y_te.reset_index(drop=True),  inst_te.reset_index(drop=True),  Xte_out], axis=1)

        self._write_cv_files(train_out, test_out)
        self._write_artifacts(sel, list(Xtr.columns), list(Xtr_out.columns))
        self._save_runtime()
        self._complete_flag()

    # ---- helpers ----
    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.dataset_name = self.cv_train_path.split('/')[-3]
        self.cv_count = self.cv_train_path.split('/')[-1].split("_")[-2]
        tr = pd.read_csv(self.cv_train_path, na_values='NA', sep=',')
        te = pd.read_csv(self.cv_test_path, na_values='NA', sep=',')
        return tr, te

    def _write_cv_files(self, train: pd.DataFrame, test: pd.DataFrame):
        if self.overwrite_cv:
            os.remove(self.cv_train_path); os.remove(self.cv_test_path)
        else:
            cvdir = os.path.join(self.experiment_path, self.dataset_name, "CVDatasets")
            os.rename(self.cv_train_path, os.path.join(cvdir, f"{self.dataset_name}_CVOnly_{self.cv_count}_Train.csv"))
            os.rename(self.cv_test_path,  os.path.join(cvdir, f"{self.dataset_name}_CVOnly_{self.cv_count}_Test.csv"))
        train.to_csv(self.cv_train_path, index=False)
        test.to_csv(self.cv_test_path, index=False)

    def _write_artifacts(self, sel, in_cols, out_cols):
        base = os.path.join(self.experiment_path, self.dataset_name, "feature_importance")
        os.makedirs(base, exist_ok=True)

        # rankings/scores
        scores = sel.get_scores()
        pd.DataFrame({"feature": list(scores.keys()), "score": list(scores.values())}) \
            .sort_values("score", ascending=False).to_csv(os.path.join(base, f"scores_cv{self.cv_count}.csv"), index=False)

        with open(os.path.join(base, f"importance_cv{self.cv_count}.pickle"), "wb") as f:
            pickle.dump({"id": self.importance_id, "params": sel.get_params()}, f)

        with open(os.path.join(base, f"selected_features_cv{self.cv_count}.txt"), "w") as f:
            f.write("\n".join(out_cols))

        manifest = {
            "dataset": self.dataset_name,
            "cv": int(self.cv_count),
            "importance": {"id": self.importance_id, "params": sel.get_params()},
            "top_k": self.top_k,
            "threshold": self.threshold,
            "keep_original_features": bool(self.keep_original_features),
            "input_feature_count": len(in_cols),
            "selected_feature_count": len(out_cols),
        }
        with open(os.path.join(base, f"feature_importance_manifest_cv{self.cv_count}.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    def _save_runtime(self):
        rt = os.path.join(self.experiment_path, self.dataset_name, "runtime")
        os.makedirs(rt, exist_ok=True)
        with open(os.path.join(rt, f"runtime_feature_importance{self.cv_count}.txt"), "w+") as f:
            f.write(str(time.time() - self.job_start_time))

    def _complete_flag(self):
        done = os.path.join(self.experiment_path, "jobsCompleted")
        os.makedirs(done, exist_ok=True)
        with open(os.path.join(done, f"job_feature_importance_{self.dataset_name}_{self.cv_count}.txt"), "w") as f:
            f.write("complete")
