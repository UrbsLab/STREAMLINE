from __future__ import annotations
import os, re, pickle, logging, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    roc_auc_score, precision_recall_curve, roc_curve, auc, average_precision_score,
)

from streamline.p7_ensembles.utils.loader import get_ensemble_by_id
# We intentionally do NOT subclass prior Job to keep this phase standalone as requested before.

class EnsembleJob:
    """
    Phase 7: Ensemble Learning on top of Phase 6 models.
    Loads pickled base models per CV split, builds requested ensembles, fits on CV-Train, evaluates on CV-Test.
    Saves:
      - models/pickledModels/<ENSEMBLE_ID>_<cv>.pickle
      - model_evaluation/pickled_metrics/<ENSEMBLE_ID>_CV_<cv>_metrics.pickle
    """

    def __init__(
        self,
        dataset_dir: str,                 # <output>/<exp>/<dataset>
        n_splits: int,
        outcome_label: str = "Class",
        instance_label: Optional[str] = None,
        # ensembles: CSV of ids (vote_hard,vote_soft,stack_lr,stack_dt,stack_rf)
        ensembles: Optional[str] = "vote_hard,vote_soft,stack_lr",
        base_model_filter: Optional[str] = None,  # CSV of base model small_names to include; None = all found
        random_state: Optional[int] = 0,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.n_splits = int(n_splits)
        self.outcome_label = outcome_label
        self.instance_label = instance_label
        self.ensemble_ids = [e.strip() for e in (ensembles or "").split(",") if e.strip()]
        self.base_model_filter = None
        if base_model_filter:
            self.base_model_filter = {s.strip() for s in base_model_filter.split(",") if s.strip()}
        self.random_state = random_state

        # out dirs
        (self.dataset_dir / "models" / "pickledModels").mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "model_evaluation" / "pickled_metrics").mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "runtime").mkdir(parents=True, exist_ok=True)

        self.job_start_time = None

    # ----------------------------
    # Main
    # ----------------------------
    def run(self):
        self.job_start_time = time.time()
        for cv in range(self.n_splits):
            logging.info(f"[P7] CV {cv} — loading base models, building ensembles")
            base_estimators = self._load_base_estimators(cv)
            if not base_estimators:
                logging.warning(f"[P7] No base estimators found for CV {cv}, skipping.")
                continue

            X_train, y_train, X_test, y_test = self._load_cv_data(cv)

            for ens_id in self.ensemble_ids:
                cls = get_ensemble_by_id(ens_id)
                est = cls.build(base_estimators)

                # fit
                est.fit(X_train, y_train)

                # predict labels
                y_pred = est.predict(X_test)

                # basic metrics (label-based)
                metrics_dict = self._basic_class_metrics(y_test, y_pred)

                # probabilities for ROC/PRC:
                proba = None
                if hasattr(est, "predict_proba"):
                    try:
                        proba = est.predict_proba(X_test)[:, 1]
                    except Exception:
                        proba = None

                if proba is not None:
                    fpr, tpr, _ = roc_curve(y_test, proba)
                    prec, rec, _ = precision_recall_curve(y_test, proba)
                    metrics_dict["ROC AUC"] = float(auc(fpr, tpr))
                    metrics_dict["PRC AUC"] = float(auc(rec, prec))
                    metrics_dict["PRC APS"] = float(average_precision_score(y_test, proba))
                    metrics_dict["_curves"] = {"fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec}
                else:
                    # HARD VOTING special handling (threshold sweep over base probs → majority vote)
                    if ens_id.lower() == "vote_hard":
                        curves = self._hard_voting_curves(base_estimators, X_test, y_test)
                        metrics_dict["ROC AUC"] = float(curves["roc_auc"])
                        metrics_dict["PRC AUC"] = float(curves["prc_auc"])
                        metrics_dict["PRC APS"] = float(curves["prc_aps"])
                        metrics_dict["_curves"] = {
                            "fpr": curves["fpr"], "tpr": curves["tpr"],
                            "precision": curves["precision"], "recall": curves["recall"]
                        }

                # save model + metrics
                alg = cls.id
                with open(self.dataset_dir / "models" / "pickledModels" / f"{alg}_{cv}.pickle", "wb") as f:
                    pickle.dump(est, f)
                with open(self.dataset_dir / "model_evaluation" / "pickled_metrics" / f"{alg}_CV_{cv}_metrics.pickle", "wb") as f:
                    pickle.dump(metrics_dict, f)

                logging.info(f"[P7] Saved ensemble {alg} for CV {cv}")

        # runtime
        rt = self.dataset_dir / "runtime" / "runtime_ensemble.txt"
        rt.write_text(str(time.time() - self.job_start_time))

    # ----------------------------
    # Helpers
    # ----------------------------
    def _load_base_estimators(self, cv: int) -> List[Tuple[str, object]]:
        """
        Expect files: models/pickledModels/<SMALLNAME>_<cv>.pickle
        Optionally filtered by base_model_filter
        """
        pm_dir = self.dataset_dir / "models" / "pickledModels"
        if not pm_dir.exists():
            return []

        out: List[Tuple[str, object]] = []
        for fn in os.listdir(pm_dir):
            if not fn.endswith(".pickle"):
                continue
            m = re.match(r"(.+?)_([0-9]+)\.pickle$", fn)
            if not m:
                continue
            small, fold = m.group(1), int(m.group(2))
            if fold != cv:
                continue
            if self.base_model_filter and small not in self.base_model_filter:
                continue
            with open(pm_dir / fn, "rb") as f:
                model = pickle.load(f)
            out.append((small, model))
        return sorted(out, key=lambda t: t[0])

    def _load_cv_data(self, cv: int):
        ds = self.dataset_dir.name
        cv_dir = self.dataset_dir / "CVDatasets"
        train = pd.read_csv(cv_dir / f"{ds}_CV_{cv}_Train.csv")
        test  = pd.read_csv(cv_dir / f"{ds}_CV_{cv}_Test.csv")
        if self.instance_label and self.instance_label in train.columns:
            train = train.drop(columns=[self.instance_label])
            test = test.drop(columns=[self.instance_label])
        X_train = train.drop(columns=[self.outcome_label]).values
        y_train = train[self.outcome_label].values
        X_test  = test.drop(columns=[self.outcome_label]).values
        y_test  = test[self.outcome_label].values
        return X_train, y_train, X_test, y_test

    @staticmethod
    def _basic_class_metrics(y_true, y_pred) -> Dict[str, float]:
        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        lr_plus = tp / (tp + fp) if (tp + fp) else 0.0
        lr_minus = fn / (tn + fn) if (tn + fn) else 0.0
        return {
            "Accuracy": float(acc),
            "Balanced Accuracy": float(bacc),
            "F1": float(f1),
            "Precision": float(prec),
            "Recall": float(rec),
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "NPV": float(npv), "LR+": float(lr_plus), "LR-": float(lr_minus),
        }

    @staticmethod
    def _hard_voting_curves(base_estimators: List[Tuple[str, object]], X_test, y_test):
        """
        Compute ROC/PRC for hard voting by sweeping a threshold over base predict_proba
        and majority-voting binary decisions.
        """
        thresholds = np.linspace(0.0, 1.0, 101)
        fprs, tprs = [], []
        precisions, recalls = [], []

        # precompute probas for speed; skip models that lack predict_proba
        proba_list = []
        for _, m in base_estimators:
            if hasattr(m, "predict_proba"):
                try:
                    proba_list.append(m.predict_proba(X_test)[:, 1])
                except Exception:
                    pass
        proba_arr = np.column_stack(proba_list) if proba_list else None

        if proba_arr is None or proba_arr.shape[1] == 0:
            # fallback: no curves possible
            return {"fpr": [0,1], "tpr": [0,1], "precision": [1,0], "recall": [0,1], "roc_auc": 0.5, "prc_auc": 0.5, "prc_aps": 0.0}

        for th in thresholds:
            votes = (proba_arr >= th).astype(int)          # shape: (n, n_models)
            maj = (votes.mean(axis=1) >= 0.5).astype(int)  # majority
            tn, fp, fn, tp = confusion_matrix(y_test, maj).ravel()
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (tn + fp) if (tn + fp) else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 1.0
            recall = tpr
            fprs.append(fpr); tprs.append(tpr)
            precisions.append(precision); recalls.append(recall)

        # sort and integrate
        idx = np.argsort(fprs)
        fpr_s = np.array(fprs)[idx]; tpr_s = np.array(tprs)[idx]
        rocA = auc(fpr_s, tpr_s)

        idxp = np.argsort(recalls)
        rec_s = np.array(recalls)[idxp]; pre_s = np.array(precisions)[idxp]
        prcA = auc(rec_s, pre_s)
        aps = float(average_precision_score(y_test, (proba_arr.mean(axis=1))))  # a quick proxy

        return {"fpr": fpr_s, "tpr": tpr_s, "precision": pre_s, "recall": rec_s, "roc_auc": float(rocA), "prc_auc": float(prcA), "prc_aps": aps}
