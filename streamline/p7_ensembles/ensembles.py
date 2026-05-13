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

from sklearn.metrics import brier_score_loss
from streamline.p6_modeling.utils.support import multiclass_brier_score
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
        outcome_type: Optional[str] = None,
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
        self.outcome_type = _normalize_outcome_type(outcome_type)
        if self.outcome_type == "Continuous":
            raise NotImplementedError(
                "Phase 7 ensembles currently support binary and multiclass classification only. "
                "Regression ensembles are not implemented."
            )
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
                
                try:
                    y_pred_proba = model.predict_proba(X_test)
                except Exception as e:
                    # logging.warning("predict_proba failed; setting y_pred_proba=None.")
                    # logging.warning(str(e))
                    y_pred_proba = None

                # Metrics (discrete)
                metrics = _calc_basic_metrics(y_test, y_pred)

                # Curves & probabilistic metrics
                roc_curve_dict = None
                prc_curve_dict = None
                roc_auc_val = None
                prc_auc_val = None
                aps_val = None
                brier_val = None

                if ens_id == "hard_voting":
                    roc_curve_dict, prc_curve_dict, roc_auc_val, prc_auc_val, aps_val = \
                        _hard_voting_threshold_sweep(base_ests, X_test, y_test)

                    # --- Brier score for hard voting ---
                    # Use mean predicted probability across base models as the ensemble probability.
                    # Binary: use P(class=1). Multiclass: use full P over classes.
                    try:
                        y_true = np.asarray(y_test)
                        classes = np.unique(y_true)
                        n_classes = classes.size

                        base_probas_list = []
                        for _, est in base_ests:
                            proba = None
                            if hasattr(est, "predict_proba"):
                                proba = est.predict_proba(X_test)
                            elif hasattr(est, "decision_function"):
                                s = np.asarray(est.decision_function(X_test))
                                if s.ndim == 1:
                                    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
                                    proba = np.column_stack([1 - s, s])
                                else:
                                    s_min = s.min(axis=0, keepdims=True)
                                    s_max = s.max(axis=0, keepdims=True)
                                    proba = (s - s_min) / (s_max - s_min + 1e-12)

                            if proba is None:
                                continue

                            proba = np.asarray(proba)

                            # Align estimator proba columns to global `classes` if possible
                            est_classes = getattr(est, "classes_", None)
                            if est_classes is not None:
                                est_classes = np.asarray(est_classes)
                                idx = []
                                ok = True
                                for c in classes:
                                    m = np.where(est_classes == c)[0]
                                    if m.size == 0:
                                        ok = False
                                        break
                                    idx.append(int(m[0]))
                                if not ok:
                                    continue
                                proba = proba[:, idx]
                            else:
                                # If no classes_, require same column count
                                if proba.ndim != 2 or proba.shape[1] != n_classes:
                                    continue

                            base_probas_list.append(proba)

                        if base_probas_list:
                            mean_proba = np.mean(np.stack(base_probas_list, axis=2), axis=2)  # (n, n_classes)

                            if n_classes == 2:
                                # Brier score (binary) on positive class probability
                                brier_val = brier_score_loss(y_true, mean_proba[:, 1])
                            else:
                                # Multiclass brier (use your Phase-6 helper)
                                brier_val = multiclass_brier_score(y_true, mean_proba)
                    except Exception as e:
                        logging.warning("Failed to compute Brier Score for hard voting; setting to None.")
                        logging.warning(str(e))
                        brier_val = None

                    # For reporting parity, also compute "naive" AUC using y_pred
                    try:
                        metrics["ROC AUC (hard from preds)"] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                    except Exception as e:
                        logging.warning("Failed to compute ROC AUC from hard predictions; setting to None.")
                        logging.warning(str(e))
                        metrics["ROC AUC (hard from preds)"] = None

                else:
                    # ------- Soft-voting & stacking: expose proba / scores -------
                    proba = None
                    if hasattr(model, "predict_proba"):
                        # For multiclass, this is (n_samples, n_classes)
                        try:
                            proba = model.predict_proba(X_test)
                        except Exception as e:
                            logging.warning("predict_proba failed; setting proba=None.")
                            logging.warning(str(e))
                            proba = None
                    elif hasattr(model, "decision_function"):
                        try:
                            s = model.decision_function(X_test)
                            s = np.asarray(s)
                            # Min-max scale per column to [0,1]
                            if s.ndim == 1:
                                proba = (s - s.min()) / (s.max() - s.min() + 1e-12)
                            else:
                                s_min = s.min(axis=0, keepdims=True)
                                s_max = s.max(axis=0, keepdims=True)
                                proba = (s - s_min) / (s_max - s_min + 1e-12)
                        except Exception as e:
                            logging.warning("decision_function failed; setting proba=None.")
                            logging.warning(str(e))
                            proba = None

                    if proba is not None:
                        classes = getattr(model, "classes_", None)
                        roc_curve_dict, prc_curve_dict, roc_auc_val, prc_auc_val, aps_val = \
                            _calc_curves_scores_from_proba(y_test, proba, classes=classes)

                        # --- Brier score for soft/stacking ---
                        try:
                            y_true = np.asarray(y_test)
                            proba_arr = np.asarray(proba)
                            n_classes = np.unique(y_true).size

                            if proba_arr.ndim == 1 and n_classes == 2:
                                # decision_function scaled to [0,1] as "pos prob"
                                brier_val = brier_score_loss(y_true, proba_arr)
                            elif proba_arr.ndim == 2 and n_classes == 2 and proba_arr.shape[1] >= 2:
                                brier_val = brier_score_loss(y_true, proba_arr[:, 1])
                            elif proba_arr.ndim == 2 and n_classes > 2:
                                brier_val = multiclass_brier_score(y_true, proba_arr)
                            else:
                                brier_val = None
                        except Exception as e:
                            logging.warning("Failed to compute Brier Score; setting to None.")
                            logging.warning(str(e))
                            brier_val = None

                # --- write metrics fields ---
                if brier_val is not None:
                    metrics["Brier Score"] = float(brier_val)

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

def _normalize_outcome_type(value: Optional[str]) -> Optional[str]:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().lower()
    if text in {"binary", "bin", "classification_binary"}:
        return "Binary"
    if text in {"multiclass", "multi", "classification_multiclass"}:
        return "Multiclass"
    if text in {"continuous", "regression", "numeric"}:
        return "Continuous"
    return str(value)

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

# def _calc_basic_metrics(y_true, y_pred) -> Dict[str, float]:
#     """
#     For binary classification:
#         - return Accuracy, Balanced Accuracy, F1, Precision, Recall
#         - plus confusion-matrix-derived TP, TN, FP, FN, NPV, LR+, LR-
#     For multiclass:
#         - return Accuracy, Balanced Accuracy
#         - macro-averaged F1, Precision, Recall
#         - omit TP/TN/FP/FN/NPV/LR+/- (not uniquely defined for multiclass)
#     """
#     y_true = np.asarray(y_true)
#     y_pred = np.asarray(y_pred)

#     labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
#     cm = confusion_matrix(y_true, y_pred, labels=labels)

#     # Binary case: keep your original behavior
#     if labels.size == 2:
#         tn, fp, fn, tp = cm.ravel()
#         npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
#         lr_plus  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#         lr_minus = fn / (tn + fn) if (tn + fn) > 0 else 0.0
#         return {
#             "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
#             "Accuracy": accuracy_score(y_true, y_pred),
#             "F1": f1_score(y_true, y_pred),
#             "Precision": precision_score(y_true, y_pred),
#             "Recall": recall_score(y_true, y_pred),
#             "TP": float(tp), "TN": float(tn), "FP": float(fp), "FN": float(fn),
#             "NPV": npv, "LR+": lr_plus, "LR-": lr_minus,
#         }

#     # Multiclass case: use macro-averaged metrics
#     return {
#         "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
#         "Accuracy": accuracy_score(y_true, y_pred),
#         "F1": f1_score(y_true, y_pred, average="macro"),
#         "Precision": precision_score(y_true, y_pred, average="macro"),
#         "Recall": recall_score(y_true, y_pred, average="macro"),
#     }

def _calc_basic_metrics(y_true, y_pred) -> Dict[str, Any]:
    """
    Phase-7 basic metrics with Phase-7 (legacy) naming.

    Multiclass results keys:
      'Balanced Accuracy', 'Accuracy', 'F1 Score', 'Sensitivity (Recall)',
      'Precision (PPV)'

    Binary results keys (adds):
      'Specificity', 'TP','TN','FP','FN','NPV','LR+','LR-'

    Notes:
      - Computes discrete metrics from y_true/y_pred.
      - Brier/AUC/PRC metrics require probability-like scores; set to None here
        (fill later if you have proba). If you pass proba elsewhere, compute there.
      - Any computation error returns None for that field.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    def _safe(fn, *args, **kwargs):
        try:
            v = fn(*args, **kwargs)
        except Exception:
            return None
        if v is None:
            return None
        try:
            v = float(v)
            if not np.isfinite(v):
                return None
            return v
        except Exception:
            return None

    # Base (shared) discrete stats
    s_bac = _safe(balanced_accuracy_score, y_true, y_pred)
    s_ac = _safe(accuracy_score, y_true, y_pred)
    # F1/Recall/Precision: binary uses default, multiclass uses macro
    labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    is_binary = labels.size == 2

    if is_binary:
        s_f1 = _safe(f1_score, y_true, y_pred)
        s_re = _safe(recall_score, y_true, y_pred)            # sensitivity
        s_pr = _safe(precision_score, y_true, y_pred)         # PPV
    else:
        s_f1 = _safe(f1_score, y_true, y_pred, average="macro")
        s_re = _safe(recall_score, y_true, y_pred, average="macro")
        s_pr = _safe(precision_score, y_true, y_pred, average="macro")

    # Probabilistic metrics placeholders here (compute later from proba if available)
    s_bs = None

    # -----------------------
    # Multiclass
    # -----------------------
    if not is_binary:
        return {
            "Balanced Accuracy": s_bac,
            "Accuracy": s_ac,
            "F1 Score": s_f1,
            "Sensitivity (Recall)": s_re,
            "Precision (PPV)": s_pr,
            "Brier Score": s_bs,
        }

    # -----------------------
    # Binary
    # -----------------------
    # Use confusion matrix in the same way your earlier code does (ravel)
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
    except Exception:
        cm = None
    if cm is None:
        # Can happen if something pathological; return best-effort with None extras
        return {
            "Balanced Accuracy": s_bac,
            "Accuracy": s_ac,
            "F1 Score": s_f1,
            "Sensitivity (Recall)": s_re,
            "Specificity": None,
            "Precision (PPV)": s_pr,
            "Brier Score": s_bs,
            "TP": None,
            "TN": None,
            "FP": None,
            "FN": None,
            "NPV": None,
            "LR+": None,
            "LR-": None,
        }

    try:
        tn, fp, fn, tp = cm.ravel()
        tn = float(tn); fp = float(fp); fn = float(fn); tp = float(tp)
    except Exception:
        tn = fp = fn = tp = None

    # Specificity and NPV
    if tn is not None and fp is not None and (tn + fp) > 0:
        s_sp = float(tn / (tn + fp))
    else:
        s_sp = None

    if tn is not None and fn is not None and (tn + fn) > 0:
        s_npv = float(tn / (tn + fn))
    else:
        s_npv = None

    # LR+/LR- using your Phase-6-style definitions (from your BinaryClassificationModel)
    # LR+ = (TPR) / (FPR)
    if tp is not None and fn is not None and fp is not None and tn is not None:
        if (tp + fn) > 0 and (fp + tn) > 0 and fp > 0:
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            s_lrp = float(tpr / fpr) if fpr > 0 else None
        else:
            s_lrp = 0.0
        # LR- = (FNR) / (TNR)
        if (tp + fn) > 0 and (fp + tn) > 0 and tn > 0:
            fnr = fn / (tp + fn)
            tnr = tn / (fp + tn)
            s_lrm = float(fnr / tnr) if tnr > 0 else None
        else:
            s_lrm = 0.0
    else:
        s_lrp = None
        s_lrm = None

    return {
        "Balanced Accuracy": s_bac,
        "Accuracy": s_ac,
        "F1 Score": s_f1,
        "Sensitivity (Recall)": s_re,
        "Specificity": s_sp,
        "Precision (PPV)": s_pr,
        "Brier Score": s_bs,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "NPV": s_npv,
        "LR+": s_lrp,
        "LR-": s_lrm,
    }

def _calc_curves_scores_from_proba(y_true, y_proba, classes=None):
    """
    Binary:
        - y_proba: 1D or 2D (n_samples, 1 or 2); returns single ROC/PRC and AUC/AP.
    Multiclass:
        - y_proba: 2D (n_samples, n_classes)
        - classes: sequence of class labels aligned with columns of y_proba
        - returns per-class ROC/PRC curves in dictionaries, plus macro-averaged AUC/AP.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    n_unique = np.unique(y_true).size

    # ------- Binary case -------
    if y_proba.ndim == 1 or (y_proba.ndim == 2 and y_proba.shape[1] in (1, 2) and n_unique == 2):
        if y_proba.ndim == 2:
            # If 2 cols, use column 1 as "positive" class score
            if y_proba.shape[1] == 2:
                y_score = y_proba[:, 1]
            else:
                y_score = y_proba[:, 0]
        else:
            y_score = y_proba

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        sidx = np.argsort(rec)
        prc_auc = auc(rec[sidx], prec[sidx])
        aps = average_precision_score(y_true, y_score)
        return (
            {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            {"precision": prec.tolist(), "recall": rec.tolist()},
            float(roc),
            float(prc_auc),
            float(aps),
        )

    # ------- Multiclass case: one-vs-rest curves -------
    if classes is None:
        classes = np.unique(y_true)
    classes = np.array(classes)
    n_classes = classes.size

    if y_proba.ndim != 2:
        raise ValueError("Multiclass probabilities must be 2D (n_samples, n_classes).")

    if y_proba.shape[1] != n_classes:
        logging.warning(
            "y_proba has %d columns but there are %d classes; "
            "using min(n_cols, n_classes) for alignment.",
            y_proba.shape[1],
            n_classes,
        )
        n = min(y_proba.shape[1], n_classes)
        y_proba = y_proba[:, :n]
        classes = classes[:n]
        n_classes = n

    roc_curve_dict: Dict[str, Dict[str, List[float]]] = {}
    prc_curve_dict: Dict[str, Dict[str, List[float]]] = {}
    roc_aucs: List[float] = []
    prc_aucs: List[float] = []
    aps_list: List[float] = []

    for idx, cls in enumerate(classes):
        y_bin = (y_true == cls).astype(int)
        y_score = y_proba[:, idx]

        # ROC
        fpr, tpr, _ = roc_curve(y_bin, y_score)
        roc_val = auc(fpr, tpr)
        # PRC
        prec, rec, _ = precision_recall_curve(y_bin, y_score)
        sidx = np.argsort(rec)
        prc_auc_val = auc(rec[sidx], prec[sidx])
        aps_val = average_precision_score(y_bin, y_score)

        key = str(cls)
        roc_curve_dict[key] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        prc_curve_dict[key] = {"precision": prec.tolist(), "recall": rec.tolist()}

        roc_aucs.append(float(roc_val))
        prc_aucs.append(float(prc_auc_val))
        aps_list.append(float(aps_val))

    roc_macro = float(np.mean(roc_aucs)) if roc_aucs else None
    prc_macro = float(np.mean(prc_aucs)) if prc_aucs else None
    aps_macro = float(np.mean(aps_list)) if aps_list else None

    return roc_curve_dict, prc_curve_dict, roc_macro, prc_macro, aps_macro

def _hard_voting_threshold_sweep(base_estimators, X_test, y_test, thresholds=None):
    """
    Build ROC/PRC for hard-vote by thresholding each base model's probabilities then majority vote.

    - Binary:
        returns single ROC and PRC curves (same as original behavior).

    - Multiclass:
        performs one-vs-rest in a micro-averaged fashion:
          * for each class c, and each sample i, treat (i, c) as a binary problem:
              y_true_bin[i, c] = 1 if y_test[i] == c else 0
              y_pred_bin[i, c] = 1 if ensemble votes for c at given threshold else 0
          * aggregate TP/FP/FN/TN across all classes at each threshold
          * compute one global TPR/FPR/precision/recall per threshold
        returns a single ROC/PR curve, just like in the binary case.
    """
    y_test = np.asarray(y_test)
    classes = np.unique(y_test)
    n_classes = classes.size

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    # Collect per-base probabilities aligned to `classes`
    base_probas_list = []
    for _, est in base_estimators:
        proba = None

        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(X_test)
        elif hasattr(est, "decision_function"):
            s = est.decision_function(X_test)
            s = np.asarray(s)
            if s.ndim == 1:
                # Binary decision_function -> map to 2-col "probabilities"
                s = (s - s.min()) / (s.max() - s.min() + 1e-12)
                proba = np.column_stack([1 - s, s])
            else:
                # Multiclass decision_function -> min-max per column
                s_min = s.min(axis=0, keepdims=True)
                s_max = s.max(axis=0, keepdims=True)
                proba = (s - s_min) / (s_max - s_min + 1e-12)

        if proba is None:
            continue

        # Align columns of proba to global `classes`
        est_classes = getattr(est, "classes_", None)
        if est_classes is not None:
            est_classes = np.asarray(est_classes)
            idx = []
            ok = True
            for c in classes:
                matches = np.where(est_classes == c)[0]
                if matches.size == 0:
                    ok = False
                    break
                idx.append(matches[0])
            if not ok:
                logging.warning(
                    "Skipping estimator in _hard_voting_threshold_sweep due to class mismatch."
                )
                continue
            proba = proba[:, idx]
        else:
            # Fallback: require same number of classes and same order
            if proba.shape[1] != n_classes:
                logging.warning(
                    "Estimator proba columns (%d) != n_classes (%d) and no classes_; skipping.",
                    proba.shape[1], n_classes
                )
                continue

        base_probas_list.append(proba)

    if not base_probas_list:
        logging.warning(
            "No probability-capable base estimators for hard voting threshold sweep."
        )
        return None, None, None, None, None

    # base_probas: (n_samples, n_bases, n_classes)
    base_probas = np.stack(base_probas_list, axis=1)
    n_samples = base_probas.shape[0]
    n_bases = base_probas.shape[1]

    # -------------------------------------------------------------------------
    # Binary case: keep original behavior
    # -------------------------------------------------------------------------
    if n_classes == 2:
        pos_cls = classes[1]
        base_pos = base_probas[..., 1]  # (n_samples, n_bases)

        tpr_list, fpr_list, prec_list, rec_list = [], [], [], []
        y_true_bin = (y_test == pos_cls).astype(int)

        for th in thresholds:
            votes = (base_pos >= th).astype(int)        # (n_samples, n_bases)
            y_pred_bin = (votes.mean(axis=1) >= 0.5).astype(int)

            cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn + 1e-12)
            fpr = fp / (tn + fp + 1e-12)
            prec = tp / (tp + fp + 1e-12)
            rec = tpr

            tpr_list.append(tpr)
            fpr_list.append(fpr)
            prec_list.append(prec)
            rec_list.append(rec)

        fpr_arr = np.array(fpr_list)
        tpr_arr = np.array(tpr_list)
        prec_arr = np.array(prec_list)
        rec_arr = np.array(rec_list)

        # ROC AUC
        sidx = np.argsort(fpr_arr)
        roc = auc(fpr_arr[sidx], tpr_arr[sidx])
        # PRC AUC
        sidx2 = np.argsort(rec_arr)
        prc_auc = auc(rec_arr[sidx2], prec_arr[sidx2])
        # APS-like
        aps = float(np.mean(prec_arr))

        roc_curve_dict = {"fpr": fpr_arr[sidx].tolist(), "tpr": tpr_arr[sidx].tolist()}
        prc_curve_dict = {
            "precision": prec_arr[sidx2].tolist(),
            "recall": rec_arr[sidx2].tolist(),
        }
        return roc_curve_dict, prc_curve_dict, float(roc), float(prc_auc), aps

    # -------------------------------------------------------------------------
    # Multiclass case: micro-averaged one-vs-rest
    # -------------------------------------------------------------------------
    tpr_list, fpr_list, prec_list, rec_list = [], [], [], []

    # Precompute one-vs-rest true labels for all (sample, class)
    # y_true_bin: (n_samples, n_classes) in {0,1}
    y_true_bin = (y_test[:, None] == classes[None, :]).astype(int)

    for th in thresholds:
        # votes: (n_samples, n_bases, n_classes) -> 0/1 per base, class, sample
        votes = (base_probas >= th).astype(int)
        # majority vote per class (one-vs-rest): (n_samples, n_classes)
        y_pred_bin = (votes.mean(axis=1) >= 0.5).astype(int)

        # Flatten across classes to do micro-averaging
        y_true_flat = y_true_bin.ravel()
        y_pred_flat = y_pred_bin.ravel()

        cm = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        tpr = tp / (tp + fn + 1e-12) if (tp + fn) > 0 else 0.0  # micro TPR (a.k.a. recall)
        fpr = fp / (tn + fp + 1e-12) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp + 1e-12) if (tp + fp) > 0 else 0.0
        rec = tpr

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        prec_list.append(prec)
        rec_list.append(rec)

    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    prec_arr = np.array(prec_list)
    rec_arr = np.array(rec_list)

    # ROC AUC (micro)
    sidx = np.argsort(fpr_arr)
    roc = auc(fpr_arr[sidx], tpr_arr[sidx])
    # PRC AUC (micro over thresholds)
    sidx2 = np.argsort(rec_arr)
    prc_auc = auc(rec_arr[sidx2], prec_arr[sidx2])
    aps = float(np.mean(prec_arr))

    roc_curve_dict = {"fpr": fpr_arr[sidx].tolist(), "tpr": tpr_arr[sidx].tolist()}
    prc_curve_dict = {
        "precision": prec_arr[sidx2].tolist(),
        "recall": rec_arr[sidx2].tolist(),
    }
    return roc_curve_dict, prc_curve_dict, float(roc), float(prc_auc), aps

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
