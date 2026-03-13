# streamline/phases/p3_feature_learning/job.py
from __future__ import annotations
import os, time, json, pickle, random, logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from streamline.p3_feature_learning.utils.fl_loader import load_learner

class FeatureLearn:
    """
    Phase 3: Feature Learning (PCA only in this initial version).
    Fit on TRAIN, transform TRAIN/TEST, write updated CV CSVs and artifacts.
    """
    def __init__(
        self,
        cv_train_path: str,
        cv_test_path: str,
        experiment_path: str,
        *,
        learner_id: str = "pca",
        learner_params: Dict[str, Any] | None = None,
        feature_namespace: str = "FL_PCA",
        keep_original_features: bool = True,
        overwrite_cv: bool = True,
        outcome_label: str = "Class",
        instance_label: Optional[str] = None,
        random_state: Optional[int] = None,
    ):
        self.cv_train_path = cv_train_path
        self.cv_test_path = cv_test_path
        self.experiment_path = experiment_path
        self.learner_id = learner_id
        self.learner_params = learner_params or {}
        self.feature_namespace = feature_namespace
        self.keep_original_features = keep_original_features
        self.overwrite_cv = overwrite_cv
        self.outcome_label = outcome_label
        self.instance_label = instance_label
        self.random_state = random_state

        self.dataset_name: Optional[str] = None
        self.cv_count: Optional[str] = None
        self.job_start_time = time.time()

    # ------------ main ------------
    def run(self):
        random.seed(self.random_state); np.random.seed(self.random_state)

        data_train, data_test = self._load_data()
        logging.info("Prepared Train and Test for: %s_CV_%s", self.dataset_name, self.cv_count)
        y_train = data_train[self.outcome_label]
        y_test = data_test[self.outcome_label]

        i_train = i_test = None
        if self.instance_label is not None and self.instance_label in data_train.columns:
            i_train = data_train[self.instance_label]
            i_test = data_test[self.instance_label]

        # X = features-only
        drop_cols = [self.outcome_label] + ([self.instance_label] if self.instance_label in data_train.columns else [])
        X_train = data_train.drop(columns=drop_cols, errors="ignore")
        X_test  = data_test.drop(columns=drop_cols, errors="ignore")

        # Fit PCA on TRAIN (numeric-only inside learner)
        logging.info("Running Feature Learning (%s)...", self.learner_id)
        learner = load_learner(self.learner_id, random_state=self.random_state, **self.learner_params)
        learner.fit(X_train)
        Z_train = learner.transform(X_train)
        Z_test  = learner.transform(X_test)
        pc_added = int(Z_train.shape[1])
        pca_impl = getattr(learner, "_impl", None)
        evr = getattr(pca_impl, "explained_variance_ratio_", None)
        if evr is not None and len(evr) > 0:
            logging.info(
                "Principal components added: %d (cumulative explained variance: %.3f)",
                pc_added,
                float(np.sum(evr)),
            )
        else:
            logging.info("Principal components added: %d", pc_added)

        # Name engineered columns
        # if PCA decided n_components at fit-time, use that
        out_cols = learner.get_feature_names(X_train.columns.tolist(), self.feature_namespace)
        if len(out_cols) != Z_train.shape[1]:  # guard in case automatic n_components used
            out_cols = [f"{self.feature_namespace}_PC{i+1}"] * 0 + [f"{self.feature_namespace}_PC{i+1}" for i in range(Z_train.shape[1])]
        Z_train.columns = out_cols
        Z_test.columns = out_cols

        # Concatenate with original features or replace
        if self.keep_original_features:
            X_train_out = pd.concat([X_train.reset_index(drop=True), Z_train.reset_index(drop=True)], axis=1)
            X_test_out  = pd.concat([X_test.reset_index(drop=True),  Z_test.reset_index(drop=True)], axis=1)
        else:
            X_train_out, X_test_out = Z_train, Z_test

        # Reassemble
        if self.instance_label is None or self.instance_label not in data_train.columns:
            train_out = pd.concat([y_train.reset_index(drop=True), X_train_out], axis=1)
            test_out  = pd.concat([y_test.reset_index(drop=True),  X_test_out],  axis=1)
        else:
            train_out = pd.concat([y_train.reset_index(drop=True), i_train.reset_index(drop=True), X_train_out], axis=1)
            test_out  = pd.concat([y_test.reset_index(drop=True),  i_test.reset_index(drop=True),  X_test_out],  axis=1)

        # Write
        self._write_cv_files(train_out, test_out)
        self._write_artifacts(
            learner,
            out_cols,
            X_train.shape[1],
            Z_train.shape[1],
            train_out.shape,
            test_out.shape,
        )
        final_feature_count = int(train_out.shape[1] - 1 if self.instance_label is None else train_out.shape[1] - 2)
        logging.info(
            "Feature learning summary for %s_CV_%s: input=%d, PCs=%d, keep_original=%s, final_features=%d, train_shape=%s, test_shape=%s",
            self.dataset_name,
            self.cv_count,
            int(X_train.shape[1]),
            pc_added,
            bool(self.keep_original_features),
            final_feature_count,
            tuple(train_out.shape),
            tuple(test_out.shape),
        )
        self._save_runtime()
        self._complete_flag()
        logging.info(
            "%s CV%s phase 3 %s evaluation complete",
            self.dataset_name,
            self.cv_count,
            self.learner_id,
        )

    # ------------ helpers ------------
    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.dataset_name = self.cv_train_path.split('/')[-3]
        self.cv_count = self.cv_train_path.split('/')[-1].split("_")[-2]
        logging.info("-------------------------------------------------------")
        logging.info("Loading Dataset: %s_CV_%s_Train", self.dataset_name, self.cv_count)
        tr = pd.read_csv(self.cv_train_path, na_values='NA', sep=',')
        te = pd.read_csv(self.cv_test_path, na_values='NA', sep=',')
        return tr, te

    def _write_cv_files(self, data_train: pd.DataFrame, data_test: pd.DataFrame):
        if self.overwrite_cv:
            os.remove(self.cv_train_path); os.remove(self.cv_test_path)
        else:
            cvdir = os.path.join(self.experiment_path, self.dataset_name, "CVDatasets")
            os.rename(self.cv_train_path, os.path.join(cvdir, f"{self.dataset_name}_CVOnly_{self.cv_count}_Train.csv"))
            os.rename(self.cv_test_path,  os.path.join(cvdir, f"{self.dataset_name}_CVOnly_{self.cv_count}_Test.csv"))
        data_train.to_csv(self.cv_train_path, index=False)
        data_test.to_csv(self.cv_test_path, index=False)

    def _write_artifacts(self, learner, out_cols, in_feat_count, eng_feat_count, train_shape, test_shape):
        base = os.path.join(self.experiment_path, self.dataset_name, "feature_learning")
        os.makedirs(base, exist_ok=True)

        # Save learner as id+params (registry flavor)
        with open(os.path.join(base, f"learner_cv{self.cv_count}.pickle"), "wb") as f:
            pickle.dump({"id": self.learner_id, "params": learner.get_params()}, f)

        # Feature names
        with open(os.path.join(base, f"features_cv{self.cv_count}.txt"), "w") as f:
            f.write("\n".join(out_cols))

        manifest = {
            "dataset": self.dataset_name,
            "cv": int(self.cv_count),
            "namespace": self.feature_namespace,
            "keep_original_features": bool(self.keep_original_features),
            "learner": {"id": self.learner_id, "params": learner.get_params()},
            "input_feature_count": int(in_feat_count),
            "engineered_feature_count": int(eng_feat_count),
            "principal_components_added": int(eng_feat_count),
            "final_feature_count": int(train_shape[1] - 1 if self.instance_label is None else train_shape[1] - 2),
            "train_shape": list(train_shape),
            "test_shape": list(test_shape),
            "random_state": self.random_state,
        }
        pca_impl = getattr(learner, "_impl", None)
        evr = getattr(pca_impl, "explained_variance_ratio_", None)
        if evr is not None:
            manifest["pca_explained_variance_ratio_sum"] = float(np.sum(evr))
        with open(os.path.join(base, f"feature_manifest_cv{self.cv_count}.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    def _save_runtime(self):
        rt_dir = os.path.join(self.experiment_path, self.dataset_name, 'runtime')
        os.makedirs(rt_dir, exist_ok=True)
        with open(os.path.join(rt_dir, f"runtime_feature_learning{self.cv_count}.txt"), "w+") as f:
            f.write(str(time.time() - self.job_start_time))

    def _complete_flag(self):
        jobs_dir = os.path.join(self.experiment_path, "jobsCompleted")
        os.makedirs(jobs_dir, exist_ok=True)
        with open(os.path.join(jobs_dir, f"job_feature_learning_{self.dataset_name}_{self.cv_count}.txt"), "w") as f:
            f.write("complete")
