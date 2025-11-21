from __future__ import annotations
import os, re, pickle, logging, json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve, average_precision_score, auc
)

from streamline.p7_ensembles.utils.loader import get_ensemble_by_id


class EnsemblePhaseJob:
    """
    Phase 7: Ensemble Learning on top of Phase 6 models.
    Loads pickled base models per CV split, builds requested ensembles, fits on CV-Train, evaluates on CV-Test.
    - Fits on CV Train and evaluates on CV Test
    - Saves per-CV metrics & ROC/PRC curves to dataset_dir/ensemble_evaluation
    """
    def __init__(
        self,
        dataset_dir: str,                 # <output>/<exp>/<dataset>
        n_splits: int,
        outcome_label: str = "Class",
        instance_label: Optional[str] = None,
        # ensembles: CSV of ids (hard_voting,soft_voting,stack_lr,stack_dt,stack_rf)
        ensembles: Optional[str] = "hard_voting,soft_voting,stack_lr",
        base_models: Optional[str] = None,    # comma list of small names filter (e.g., "LR,SVM,NB")
        meta_train_source: str = "train",  # "train" or "test"
        calibrate: int = 0,
        calibrate_method: str = "sigmoid",
        calibrate_cv: int = 5,
        random_state: Optional[int] = 0,
    ):
        self.ds_dir = Path(dataset_dir)
        self.dataset_name = self.ds_dir.name
        self.n_splits = int(n_splits)
        self.outcome_label = outcome_label
        self.instance_label = instance_label
        self.ensemble_ids = [e.strip() for e in (ensembles or "").split(",") if e.strip()]
        self.base_filter = [s.strip() for s in (base_models or "").split(",") if s.strip()] or None
        self.meta_train_source = meta_train_source
        self.calibrate = bool(calibrate)
        self.calibrate_method = calibrate_method
        self.calibrate_cv = int(calibrate_cv)
        self.random_state = random_state

        # output dirs
        self.out_root = self.ds_dir / "ensemble_evaluation"
        _ensure_dir(self.out_root)
        _ensure_dir(self.out_root / "pickled_ensembles")
        _ensure_dir(self.out_root / "metrics_by_cv")
        _ensure_dir(self.out_root / "curves_by_cv")

    def run(self):
        for cv in range(self.n_splits):
            logging.info(f"[P7] {self.dataset_name} CV={cv}: loading base estimators...")
            base_ests = _load_base_estimators(self.ds_dir, cv, self.base_filter)
            if not base_ests:
                logging.warning(f"[P7] No base estimators found for CV {cv}. Skipping.")
                continue

            train_df, test_df = _load_cv_df(self.ds_dir, self.dataset_name, cv)
            train_df = _drop_instance(train_df, self.instance_label)
            test_df  = _drop_instance(test_df,  self.instance_label)
            X_train, y_train = _prep_xy(train_df, self.outcome_label)
            X_test,  y_test  = _prep_xy(test_df,  self.outcome_label)

            for ens_id in self.ensemble_ids:
                Ens = get_ensemble_by_id(ens_id)
                ens_small = getattr(Ens, "small_name", ens_id)
                ens_name  = getattr(Ens, "model_name", ens_id)

                logging.info(f"[P7] Building ensemble: {ens_name} using {len(base_ests)} base models")
                model = Ens(base_estimators=base_ests, random_state=self.random_state)

                if ens_id in ("hard_voting", "soft_voting"):  
                    # ------- Voting ensembles -------
                    if self.calibrate:
                        model_cv = CalibratedClassifierCV(
                            base_estimator=model, method=self.calibrate_method, cv=self.calibrate_cv
                        )
                        model_cv.fit(X_train, y_train)  # calibration fits wrapper
                        model = model_cv  # use calibrated version
                    else:
                        model.fit(X_train, y_train)
                    # (metrics/curves saving stays as before)
                else:
                    # ------- Manual stacking -------
                    logging.info(f"[P7] Building ensemble (stacking): {ens_name} [meta on {self.meta_train_source}]")
                    meta = model._default_meta()

                    # choose meta training split
                    if self.meta_train_source == "train":
                        # Xm_train = _stack_meta_features(base_ests, X_train, prefer_proba=True)
                        Xm_train = X_train
                        ym_train = y_train
                    else:  # "test"
                        logging.warning("[P7] meta_train_source='test' will leak; using test for meta training by request.")
                        # Xm_train = _stack_meta_features(base_ests, X_test, prefer_proba=True)
                        Xm_train = X_test
                        ym_train = y_test

                    # optional calibration ON THE META space (no CV of bases)
                    if self.calibrate:
                        model_cv = CalibratedClassifierCV(
                            base_estimator=meta, method=self.calibrate_method, cv=self.calibrate_cv
                        )
                        model_cv.fit(Xm_train, ym_train)  # calibration fits wrapper
                        model = model_cv  # use calibrated version
                    else:
                        model.fit(Xm_train, ym_train)


                # Persist ensemble for this CV
                with open(self.out_root / "pickled_ensembles" / f"{ens_small}_{cv}.pickle", "wb") as f:
                    pickle.dump(model, f)

                # Predictions
                y_pred = model.predict(X_test)

                # Metrics (discrete)
                metrics = _calc_basic_metrics(y_test, y_pred)

                # Curves & probabilistic metrics
                roc_curve_dict = None
                prc_curve_dict = None
                roc_auc_val = None
                prc_auc_val = None
                aps_val = None

                if ens_id == "hard_voting":
                    roc_curve_dict, prc_curve_dict, roc_auc_val, prc_auc_val, aps_val = \
                        _hard_voting_threshold_sweep(base_ests, X_test, y_test)
                    # For reporting parity, also compute "naive" AUC using y_pred
                    try:
                        metrics["ROC AUC (hard from preds)"] = roc_auc_score(y_test, y_pred)
                    except Exception:
                        metrics["ROC AUC (hard from preds)"] = None
                else:
                    # Soft vote & Stacking expose proba
                    proba = None
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_test)[:, 1]
                    elif hasattr(model, "decision_function"):
                        s = model.decision_function(X_test)
                        proba = (s - s.min()) / (s.max() - s.min() + 1e-12)
                    if proba is not None:
                        roc_curve_dict, prc_curve_dict, roc_auc_val, prc_auc_val, aps_val = \
                            _calc_curves_scores_from_proba(y_test, proba)

                if roc_auc_val is not None:
                    metrics["ROC AUC"] = float(roc_auc_val)
                if prc_auc_val is not None:
                    metrics["PRC AUC"] = float(prc_auc_val)
                if aps_val is not None:
                    metrics["PRC APS"] = float(aps_val)

                # Save metrics (per CV per ensemble)
                mpath = self.out_root / "metrics_by_cv" / f"{ens_small}_CV_{cv}.json"
                with open(mpath, "w") as f:
                    json.dump(metrics, f, indent=2)

                # Save curves (if available)
                if roc_curve_dict:
                    with open(self.out_root / "curves_by_cv" / f"{ens_small}_CV_{cv}_roc.json", "w") as f:
                        json.dump(roc_curve_dict, f)
                if prc_curve_dict:
                    with open(self.out_root / "curves_by_cv" / f"{ens_small}_CV_{cv}_prc.json", "w") as f:
                        json.dump(prc_curve_dict, f)

                logging.info(f"[P7] Saved ensemble metrics/curves for {ens_small} CV={cv}")
                


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _load_cv_df(dataset_dir: Path, dataset_name: str, cv_idx: int):
    train = pd.read_csv(dataset_dir / "CVDatasets" / f"{dataset_name}_CV_{cv_idx}_Train.csv", na_values="NA")
    test  = pd.read_csv(dataset_dir / "CVDatasets" / f"{dataset_name}_CV_{cv_idx}_Test.csv",  na_values="NA")
    return train, test

def _drop_instance(df: pd.DataFrame, instance_label: Optional[str]):
    if instance_label and instance_label in df.columns:
        return df.drop(columns=[instance_label])
    return df

def _prep_xy(df: pd.DataFrame, outcome_label: str):
    X = df.drop(columns=[outcome_label]).values
    y = df[outcome_label].values
    return X, y

def _load_base_estimators(dataset_dir: Path, cv_idx: int, allow_list: Optional[List[str]]) -> List[Tuple[str, Any]]:
    pm = dataset_dir / "models" / "pickledModels"
    if not pm.exists():
        return []
    out: List[Tuple[str, Any]] = []
    for fn in os.listdir(pm):
        if not fn.endswith(".pickle"):
            continue
        m = re.match(r"(.+?)_([0-9]+)\.pickle$", fn)
        if not m:
            continue
        small, fold = m.group(1), int(m.group(2))
        if fold != cv_idx:
            continue
        if allow_list and small not in allow_list:
            continue
        with open(pm / fn, "rb") as f:
            try:
                est = pickle.load(f)
            except Exception as e:
                logging.warning(f"Skipping model {fn}: {e}")
                continue
        out.append((small, est))
    return sorted(out, key=lambda t: t[0])

def _calc_basic_metrics(y_true, y_pred) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    lr_plus  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    lr_minus = fn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "TP": float(tp), "TN": float(tn), "FP": float(fp), "FN": float(fn),
        "NPV": npv, "LR+": lr_plus, "LR-": lr_minus,
    }

def _calc_curves_scores_from_proba(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    # ensure ascending recall for AUC
    sidx = np.argsort(rec)
    prc_auc = auc(rec[sidx], prec[sidx])
    aps = average_precision_score(y_true, y_proba)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}, {"precision": prec.tolist(), "recall": rec.tolist()}, roc, prc_auc, aps

def _hard_voting_threshold_sweep(base_estimators, X_test, y_test, thresholds=None):
    """
    Build ROC/PRC for hard-vote by thresholding each base model's probabilities then majority vote.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    # collect proba for positive class from bases that support predict_proba
    base_probas = []
    for _, est in base_estimators:
        if hasattr(est, "predict_proba"):
            base_probas.append(est.predict_proba(X_test)[:, 1])
        else:
            # try decision_function → map via min-max to [0,1]
            if hasattr(est, "decision_function"):
                s = est.decision_function(X_test)
                s = (s - s.min()) / (s.max() - s.min() + 1e-12)
                base_probas.append(s)
    base_probas = np.column_stack(base_probas) if base_probas else None

    tpr_list, fpr_list, prec_list, rec_list = [], [], [], []
    if base_probas is None or base_probas.shape[1] == 0:
        # fallback: predict hard from first estimator only (degenerate)
        y_pred = base_estimators[0][1].predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        tpr_list = [tp / (tp + fn + 1e-12)]
        fpr_list = [fp / (tn + fp + 1e-12)]
        prec_list = [tp / (tp + fp + 1e-12)]
        rec_list = [tp / (tp + fn + 1e-12)]
    else:
        for th in thresholds:
            votes = (base_probas >= th).astype(int)
            y_pred = (votes.mean(axis=1) >= 0.5).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            tpr_list.append(tp / (tp + fn + 1e-12))
            fpr_list.append(fp / (tn + fp + 1e-12))
            prec_list.append(tp / (tp + fp + 1e-12))
            rec_list.append(tp / (tp + fn + 1e-12))

    # ROC AUC
    sidx = np.argsort(fpr_list)
    roc = auc(np.array(fpr_list)[sidx], np.array(tpr_list)[sidx])
    # PRC AUC
    sidx2 = np.argsort(rec_list)
    prc_auc = auc(np.array(rec_list)[sidx2], np.array(prec_list)[sidx2])
    # APS (average precision like)
    aps = float(np.mean(prec_list))
    roc_curve_dict = {"fpr": list(np.array(fpr_list)[sidx]), "tpr": list(np.array(tpr_list)[sidx])}
    prc_curve_dict = {"precision": list(np.array(prec_list)[sidx2]), "recall": list(np.array(rec_list)[sidx2])}
    return roc_curve_dict, prc_curve_dict, roc, prc_auc, aps

# Class to use if we remove stacking class from mlextend

def _stack_meta_features(base_estimators, X, prefer_proba=True):
    """
    Build meta features from frozen base estimators without refitting them:
      - if predict_proba: use positive-class probs
      - elif decision_function: min-max to [0,1]
      - else: use predicted labels (0/1)
    Returns: (n_samples, n_bases) numpy array
    """
    cols = []
    for _, est in base_estimators:
        if prefer_proba and hasattr(est, "predict_proba"):
            cols.append(est.predict_proba(X)[:, 1])
        elif hasattr(est, "decision_function"):
            s = est.decision_function(X)
            s = (s - s.min()) / (s.max() - s.min() + 1e-12)
            cols.append(s)
        else:
            cols.append(est.predict(X).astype(float))
    return np.column_stack(cols) if cols else np.empty((X.shape[0], 0))
