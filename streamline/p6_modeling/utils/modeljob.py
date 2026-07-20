import os
import logging
import pickle
import random
import time
import json
import numpy as np
import optuna
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.api.types import is_object_dtype, is_string_dtype

from streamline.p6_modeling.utils.categorical import (
    FeatureTypeModelWrapper,
    NATIVE_CATEGORICAL_MODEL_IDS_DEFAULT,
    cast_native_categoricals,
    normalize_model_id,
    one_hot_align,
    parse_model_id_csv,
    to_numeric_matrix,
)


class ModelJob:
    def __init__(self, full_path, output_path, experiment_name, cv_count, outcome_label="Class",
                 instance_label=None, scoring_metric='balanced_accuracy', metric_direction='maximize', n_trials=200,
                 timeout=900, training_subsample=0, uniform_fi=False, save_plot=False, random_state=None,
                 bypass_one_hot_for_native_models=True, native_categorical_models=None,
                 # NEW: calibration controls (classification only)
                 calibrate=False, calibrate_method="sigmoid", calibrate_cv=5):
        """
        Phase-local ModelJob with optional probability calibration.
        """
        super().__init__()
        self.algorithm = ""
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.outcome_label = outcome_label
        self.instance_label = instance_label
        self.scoring_metric = scoring_metric
        self.metric_direction = metric_direction
        self.full_path = full_path
        self.cv_count = cv_count
        self.data_name = self.full_path.split('/')[-1]
        self.train_file_path = self.full_path + '/CVDatasets/' + self.data_name \
                               + '_CV_' + str(self.cv_count) + '_Train.csv'
        self.test_file_path = self.full_path + '/CVDatasets/' + self.data_name \
                              + '_CV_' + str(self.cv_count) + '_Test.csv'

        feature_names = pd.read_csv(self.train_file_path).columns.values.tolist()
        if self.instance_label is not None:
            try:
                feature_names.remove(self.instance_label)
            except ValueError:
                pass
        if self.outcome_label in feature_names:
            feature_names.remove(self.outcome_label)
        self.feature_names = feature_names

        if not os.path.exists(self.output_path):
            raise Exception("Output path must exist (from phase 1) before phase 6 can begin")
        if not os.path.exists(self.output_path + '/' + self.experiment_name):
            raise Exception("Experiment must exist (from phase 1) before phase 6 can begin")

        self.n_trials = n_trials
        self.timeout = timeout
        self.training_subsample = training_subsample
        self.random_state = random_state
        self.uniform_fi = uniform_fi
        self.feature_importance = None
        self.save_plot = save_plot
        self.param_grid = None
        self.bypass_one_hot_for_native_models = bool(bypass_one_hot_for_native_models)
        self.native_categorical_models = parse_model_id_csv(
            native_categorical_models,
            default=NATIVE_CATEGORICAL_MODEL_IDS_DEFAULT,
        )
        self.raw_feature_names = list(self.feature_names)
        self.raw_categorical_feature_names = []
        self.categorical_feature_names = []
        self.categorical_encoding_mode = "none"
        self.encoded_feature_names = list(self.feature_names)

        # calibration
        self.calibrate = bool(calibrate)
        self.calibrate_method = calibrate_method
        self.calibrate_cv = calibrate_cv

    def run(self, model):
        self.job_start_time = time.time()
        self.algorithm = model.small_name
        logging.info('Running ' + str(self.algorithm) + ' on ' + str(self.train_file_path))

        metrics_dir = os.path.join(self.full_path, 'model_evaluation', 'metrics_by_cv')
        curves_dir = os.path.join(self.full_path, 'model_evaluation', 'curves_by_cv')
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(curves_dir, exist_ok=True)

        if model.model_type != "Regression":
            metrics_payload, curves_payload = self.run_model(model)

            # ---- JSON metrics ----
            mpath = os.path.join(
                metrics_dir,
                f"{self.algorithm}_CV_{self.cv_count}.json",
            )
            with open(mpath, "w") as f:
                json.dump(metrics_payload, f, indent=2)

            # ---- JSON curves (ROC + PRC) ----
            if curves_payload is not None:
                roc = curves_payload.get("roc", {})
                prc = curves_payload.get("prc", {})

                if roc:
                    rpath = os.path.join(
                        curves_dir,
                        f"{self.algorithm}_CV_{self.cv_count}_roc.json",
                    )
                    with open(rpath, "w") as f:
                        json.dump(roc, f, indent=2)

                if prc:
                    ppath = os.path.join(
                        curves_dir,
                        f"{self.algorithm}_CV_{self.cv_count}_prc.json",
                    )
                    with open(ppath, "w") as f:
                        json.dump(prc, f, indent=2)

        else:
            metrics_payload, residuals = self.run_model(model)

            # ---- JSON metrics (regression) ----
            mpath = os.path.join(
                metrics_dir,
                f"{self.algorithm}_CV_{self.cv_count}.json",
            )
            with open(mpath, "w") as f:
                json.dump(metrics_payload, f, indent=2)

            # ---- residuals still pickled (not metrics) ----
            rpath = os.path.join(
                self.full_path,
                'model_evaluation',
                'pickled_metrics',
                f"{self.algorithm}_CV_{self.cv_count}_residuals.pickle",
            )
            with open(rpath, "wb") as f:
                pickle.dump(residuals, f)

        self.save_runtime()
        logging.info(self.full_path.split('/')[-1] + " [CV_" + str(self.cv_count) + "] (" + self.algorithm
                     + ") training complete. ------------------------------------")
        experiment_path = '/'.join(self.full_path.split('/')[:-1])
        os.makedirs(experiment_path + '/jobsCompleted/', exist_ok=True)
        job_file = open(experiment_path + '/jobsCompleted/job_model_' + self.full_path.split('/')[-1]
                        + '_' + str(self.cv_count) + '_' + self.algorithm + '.txt', 'w')
        job_file.write('complete')
        job_file.close()


    def run_model(self, model):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        x_train, y_train, x_test, y_test = self.data_prep(model)
        self._configure_native_categorical_model(model)

        # optional training subsample for models that explicitly allow it
        if 0 < self.training_subsample < x_train.shape[0] and getattr(model, "subsampling_allowed", False):
            y_array = np.asarray(y_train)
            model_type = getattr(model, "model_type", None)
            strategy = str(getattr(model, "subsampling_strategy", "balanced") or "balanced").strip().lower()

            if model_type in {"Binary", "Multiclass"} and strategy in {"balanced",}:
                try:
                    from imblearn.under_sampling import RandomUnderSampler
                except (ModuleNotFoundError, ImportError) as exc:
                    raise ImportError(
                        "Balanced training_subsample requires imbalanced-learn. "
                        "Install it with `pip install imbalanced-learn` or use the project requirements."
                    ) from exc

                class_values = np.unique(y_array)
                if len(class_values) <= 1:
                    rng = np.random.default_rng(self.random_state)
                    train_index = rng.choice(
                        np.arange(len(y_array)),
                        size=min(int(self.training_subsample), len(y_array)),
                        replace=False,
                    )
                else:
                    index_column = np.arange(len(y_array)).reshape(-1, 1)
                    sampler = RandomUnderSampler(
                        sampling_strategy=getattr(model, "undersampling_strategy", "auto"),
                        random_state=self.random_state,
                    )
                    sampled_indices, _ = sampler.fit_resample(index_column, y_array)
                    train_index = np.asarray(sampled_indices.ravel(), dtype=int)

            elif model_type in {"Binary", "Multiclass"} and strategy in {"stratified", "proportional"}:
                class_values, counts = np.unique(y_array, return_counts=True)
                if len(class_values) <= 1 or counts.min() < 2 or int(self.training_subsample) < len(class_values):
                    logging.warning(
                        "training_subsample stratified sampling is not possible for this class distribution; "
                        "using random sampling instead."
                    )
                    rng = np.random.default_rng(self.random_state)
                    train_index = rng.choice(
                        np.arange(len(y_array)),
                        size=min(int(self.training_subsample), len(y_array)),
                        replace=False,
                    )
                else:
                    sss = StratifiedShuffleSplit(
                        n_splits=1,
                        train_size=min(int(self.training_subsample), len(y_array)),
                        random_state=self.random_state,
                    )
                    for train_index, _ in sss.split(np.zeros(len(y_array)), y_array):
                        train_index = np.asarray(train_index, dtype=int)

            elif strategy in {"random", "uniform"} or model_type == "Regression":
                rng = np.random.default_rng(self.random_state)
                train_index = rng.choice(
                    np.arange(len(y_array)),
                    size=min(int(self.training_subsample), len(y_array)),
                    replace=False,
                )

            else:
                raise ValueError(
                    f"Unknown subsampling_strategy '{strategy}' for {getattr(model, 'small_name', 'model')}. "
                    "Use 'balanced', 'stratified', or 'random'."
                )

            x_train = self.take_rows(x_train, train_index)
            y_train = y_array[train_index]
            if model_type in {"Binary", "Multiclass"}:
                values, counts = np.unique(y_train, return_counts=True)
                class_counts = {str(value): int(count) for value, count in zip(values, counts)}
                logging.warning(
                    "For %s, training sample reduced to %s instances with %s class counts",
                    model.small_name,
                    x_train.shape[0],
                    class_counts,
                )
            else:
                logging.warning(
                    "For %s, training sample reduced to %s instances",
                    model.small_name,
                    x_train.shape[0],
                )

        try:
            model.fit(x_train, y_train, self.n_trials, self.timeout, self.feature_names)
        except Exception:
            if (
                self.categorical_encoding_mode == "native"
                and self.bypass_one_hot_for_native_models
                and not self._p1_one_hot_disabled()
            ):
                logging.warning(
                    "Native categorical fit failed for %s; retrying with one-hot encoded categoricals.",
                    model.small_name,
                    exc_info=True,
                )
                x_train, y_train, x_test, y_test = self.data_prep(model, force_one_hot=True)
                self._configure_native_categorical_model(model)
                model.fit(x_train, y_train, self.n_trials, self.timeout, self.feature_names)
            else:
                raise

        os.makedirs(self.full_path + '/models/', exist_ok=True)
        os.makedirs(self.full_path + '/models/pickledModels/', exist_ok=True)
        # keep pickled_metrics only for residuals (regression)
        if not os.path.exists(self.full_path + '/model_evaluation/pickled_metrics/'):
            os.makedirs(self.full_path + '/model_evaluation/pickled_metrics/', exist_ok=True)
        # NEW: JSON outputs
        if not os.path.exists(self.full_path + '/model_evaluation/metrics_by_cv/'):
            os.makedirs(self.full_path + '/model_evaluation/metrics_by_cv/', exist_ok=True)
        if not os.path.exists(self.full_path + '/model_evaluation/curves_by_cv/'):
            os.makedirs(self.full_path + '/model_evaluation/curves_by_cv/', exist_ok=True)

        self.export_optuna_report(model)

        # Export tuned / used params
        if not model.is_single:
            if self.save_plot:
                try:
                    fig = optuna.visualization.plot_parallel_coordinate(model.study)
                    fig.write_image(self.full_path + '/models/' + self.algorithm +
                                    '_ParamOptimization_' + str(self.cv_count) + '.png')
                except Exception as e:
                    logging.warning(str(e))
                    logging.warning('Warning: Optuna plot failed. Consider optuna==2.0.0.')
            self.export_best_params(self.full_path + '/models/' + self.algorithm +
                                    '_bestparams' + str(self.cv_count) + '.csv',
                                    model.params)
        else:
            self.export_best_params(self.full_path + '/models/' + self.algorithm +
                                    '_usedparams' + str(self.cv_count) + '.csv',
                                    model.params)

        # ---------- NEW: probability calibration (classification only) ----------
        if self.calibrate and model.model_type in ["Binary", "Multiclass"]:
            try:
                cal = CalibratedClassifierCV(estimator=model.model,
                                             method=self.calibrate_method,
                                             cv=self.calibrate_cv)
                cal.fit(x_train, y_train)
                model.model = cal
                logging.info(f"Calibrated {self.algorithm} with {self.calibrate_method} (cv={self.calibrate_cv})")
            except Exception as e:
                logging.warning(f"Calibration failed for {self.algorithm}: {e}")

        # Feature importance
        perm_scoring_metric = self.scoring_metric
        if self.scoring_metric.startswith('mean_'):
            perm_scoring_metric = 'neg_'+self.scoring_metric
        
        if self.uniform_fi:
            results = permutation_importance(model.model, x_train, y_train, n_repeats=10,
                                             random_state=self.random_state, scoring=perm_scoring_metric)
            self.feature_importance = results.importances_mean
        else:
            try:
                self.feature_importance = model.model.feature_importances_
            except AttributeError:
                results = permutation_importance(model.model, x_train, y_train, n_repeats=10,
                                                 random_state=self.random_state, scoring=perm_scoring_metric)
                self.feature_importance = results.importances_mean
        self.feature_importance = self._feature_importance_for_report(self.feature_importance)

        # Persist model
        persisted_model = self._wrap_model_if_needed(model.model)
        with open(self.full_path + '/models/pickledModels/' + self.algorithm +
                  '_' + str(self.cv_count) + '.pickle', 'wb') as file:
            pickle.dump(persisted_model, file)

        fi = self.feature_importance
        # convert FI to plain list for JSON
        if hasattr(fi, "tolist"):
            fi_list = fi.tolist()
        else:
            fi_list = [float(x) for x in fi]

        # Evaluate (uses each model’s own model_evaluation)
        if model.model_type == "Regression":
            metric_dict = model.model_evaluation(x_test, y_test)

            y_train_pred = model.predict(x_train)
            y_pred = model.predict(x_test)
            residual_train = y_train - y_train_pred
            residual_test = y_test - y_pred

            metrics_payload = {
                "metrics": metric_dict,
                "feature_importance": fi_list,
                "optuna": self._json_safe(getattr(model, "optuna_report", {})),
                "categorical_feature_handling": self._categorical_report(),
            }

            # curves are not defined for regression
            curves_payload = None

            return metrics_payload, [residual_train, residual_test, y_train_pred, y_pred, y_train, y_test]

        elif model.model_type in ["Binary", "Multiclass"]:
            metric_dict, curves_dict = model.model_evaluation(x_test, y_test)

            metrics_payload = {
                "metrics": metric_dict,
                "feature_importance": fi_list,
                "optuna": self._json_safe(getattr(model, "optuna_report", {})),
                "categorical_feature_handling": self._categorical_report(),
            }
            curves_payload = curves_dict

            return metrics_payload, curves_payload


    def data_prep(self, model=None, force_one_hot=False):
        train = pd.read_csv(self.train_file_path)
        test = pd.read_csv(self.test_file_path)
        if self.instance_label is not None:
            train = train.drop(self.instance_label, axis=1)
            test = test.drop(self.instance_label, axis=1)
        x_train = train.drop(self.outcome_label, axis=1)
        y_train = train[self.outcome_label].values
        x_test = test.drop(self.outcome_label, axis=1)
        y_test = test[self.outcome_label].values
        self.raw_feature_names = list(x_train.columns)

        categorical_cols = self._categorical_columns(x_train, x_test)
        self.raw_categorical_feature_names = list(categorical_cols)

        use_native = (
            bool(categorical_cols)
            and not force_one_hot
            and self.bypass_one_hot_for_native_models
            and model is not None
            and self._model_allows_native_categorical(model)
        )

        if use_native:
            x_train = cast_native_categoricals(x_train, categorical_cols)
            x_test = cast_native_categoricals(x_test, categorical_cols)
            self.feature_names = list(x_train.columns)
            self.encoded_feature_names = list(x_train.columns)
            self.categorical_feature_names = list(categorical_cols)
            self.categorical_encoding_mode = "native"
            del train; del test
            return x_train, y_train, x_test, y_test

        if categorical_cols and self._p1_one_hot_disabled():
            model_name = self._model_label(model)
            raise ValueError(
                "P1 feature metadata shows one_hot_encoding=False, so Phase 6 cannot "
                f"one-hot encode raw categorical columns for {model_name}. "
                "Only models listed in --native_categorical_models may run in this mode. "
                "Native categorical model ids: "
                + ", ".join(sorted(self.native_categorical_models))
            )

        if categorical_cols:
            x_train = one_hot_align(x_train, categorical_cols)
            x_test = one_hot_align(x_test, categorical_cols, list(x_train.columns))
            self.feature_names = list(x_train.columns)
            self.encoded_feature_names = list(x_train.columns)
            self.categorical_feature_names = []
            self.categorical_encoding_mode = "one_hot"
            del train; del test
            return to_numeric_matrix(x_train), y_train, to_numeric_matrix(x_test), y_test

        self.feature_names = list(x_train.columns)
        self.encoded_feature_names = list(x_train.columns)
        self.categorical_feature_names = []
        self.categorical_encoding_mode = "none"
        del train; del test
        return to_numeric_matrix(x_train), y_train, to_numeric_matrix(x_test), y_test

    @staticmethod
    def take_rows(x, indices):
        if hasattr(x, "iloc"):
            return x.iloc[indices].reset_index(drop=True)
        return x[indices]

    def _load_feature_meta(self):
        meta_pickle = os.path.join(self.full_path, "exploratory", "feature_meta.pickle")
        meta_json = os.path.join(self.full_path, "exploratory", "feature_meta.json")
        try:
            if os.path.exists(meta_pickle):
                with open(meta_pickle, "rb") as f:
                    return pickle.load(f)
            if os.path.exists(meta_json):
                with open(meta_json, "r") as f:
                    return json.load(f)
        except Exception as exc:
            logging.warning("Could not load feature metadata for categorical handling: %s", exc)
        return {}

    def _p1_one_hot_disabled(self):
        meta = self._load_feature_meta()
        if "one_hot" not in meta:
            return False
        value = meta.get("one_hot")
        if isinstance(value, bool):
            return not value
        return str(value).strip().lower() in {"0", "false", "f", "no", "n"}

    @staticmethod
    def _model_label(model):
        if model is None:
            return "this model"
        small = getattr(model, "small_name", "")
        name = getattr(model, "model_name", "")
        if small and name:
            return f"{small} ({name})"
        return small or name or "this model"

    def _metadata_categorical_columns(self, feature_columns):
        meta = self._load_feature_meta()
        names = meta.get("feature_names", [])
        mask = meta.get("categorical_mask", [])
        one_hot_features = set(meta.get("one_hot_features", []))
        declared_features = meta.get("categorical_features", [])
        if isinstance(declared_features, str):
            declared_features = [declared_features]
        categorical = {
            name
            for name, is_categorical in zip(names, mask)
            if is_categorical and name not in one_hot_features
        }
        categorical.update(declared_features)
        return [col for col in feature_columns if col in categorical]

    @staticmethod
    def _dtype_categorical_columns(df):
        cols = []
        for col in df.columns:
            dtype = df[col].dtype
            if is_object_dtype(dtype) or is_string_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
                cols.append(col)
        return cols

    def _categorical_columns(self, x_train, x_test):
        feature_columns = list(x_train.columns)
        categorical = set(self._metadata_categorical_columns(feature_columns))
        categorical.update(self._dtype_categorical_columns(x_train))
        categorical.update(self._dtype_categorical_columns(x_test))
        return [col for col in feature_columns if col in categorical]

    def _model_allows_native_categorical(self, model):
        ids = {
            normalize_model_id(getattr(model, "small_name", "")),
            normalize_model_id(getattr(model, "model_name", "")),
        }
        return bool(ids.intersection(self.native_categorical_models))

    def _configure_native_categorical_model(self, model):
        estimator = getattr(model, "model", None)
        if estimator is None or not hasattr(estimator, "set_params"):
            return

        try:
            params = estimator.get_params()
        except Exception:
            params = {}

        model_ids = {
            normalize_model_id(getattr(model, "small_name", "")),
            normalize_model_id(getattr(model, "model_name", "")),
        }
        module = estimator.__class__.__module__.lower()
        name = estimator.__class__.__name__.lower()
        supports_cat_features = "cat_features" in params or "catboost" in module or "catboost" in name
        supports_exstracs_discrete_attributes = (
            self.categorical_encoding_mode == "native"
            and (
                "exstracs" in model_ids
                or "exstracs" in module
                or "exstracs" in name
                or (
                    "discrete_attribute_limit" in params
                    and "specified_attributes" in params
                )
            )
        )
        if not supports_cat_features and not supports_exstracs_discrete_attributes:
            return

        if supports_cat_features:
            cat_features = list(self.categorical_feature_names)
            try:
                estimator.set_params(cat_features=cat_features)
            except Exception as exc:
                logging.warning("Could not set cat_features on %s: %s", model.small_name, exc)

        if supports_exstracs_discrete_attributes:
            categorical_indices = [
                self.feature_names.index(col)
                for col in self.categorical_feature_names
                if col in self.feature_names
            ]
            try:
                estimator.set_params(
                    discrete_attribute_limit="d",
                    specified_attributes=np.asarray(categorical_indices, dtype=int),
                )
            except Exception as exc:
                logging.warning("Could not set ExSTraCS categorical attributes on %s: %s", model.small_name, exc)

    def _wrap_model_if_needed(self, estimator):
        mode = self.categorical_encoding_mode
        if mode not in {"one_hot", "native"}:
            mode = "numeric"
        return FeatureTypeModelWrapper(
            estimator,
            mode=mode,
            raw_feature_names=self.raw_feature_names,
            categorical_columns=self.raw_categorical_feature_names,
            encoded_feature_names=self.encoded_feature_names,
        )

    def _categorical_report(self):
        return {
            "mode": self.categorical_encoding_mode,
            "raw_feature_count": len(self.raw_feature_names),
            "encoded_feature_count": len(self.encoded_feature_names),
            "categorical_features": list(self.raw_categorical_feature_names),
            "native_categorical_features": list(self.categorical_feature_names),
            "native_categorical_indices": [
                self.feature_names.index(col)
                for col in self.categorical_feature_names
                if col in self.feature_names
            ],
            "native_categorical_models": sorted(self.native_categorical_models),
            "bypass_one_hot_for_native_models": bool(self.bypass_one_hot_for_native_models),
        }

    def _feature_importance_for_report(self, fi):
        if self.categorical_encoding_mode != "one_hot":
            return fi
        if len(fi) != len(self.encoded_feature_names):
            return fi

        fi_by_encoded = pd.Series(fi, index=self.encoded_feature_names, dtype="float64")
        raw_values = []
        categorical = set(self.raw_categorical_feature_names)
        for raw_name in self.raw_feature_names:
            if raw_name in categorical:
                prefix = raw_name + "_"
                encoded_cols = [c for c in self.encoded_feature_names if c.startswith(prefix)]
                raw_values.append(float(fi_by_encoded.loc[encoded_cols].sum()) if encoded_cols else 0.0)
            elif raw_name in fi_by_encoded.index:
                raw_values.append(float(fi_by_encoded.loc[raw_name]))
            else:
                raw_values.append(0.0)
        return np.asarray(raw_values)

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(k): ModelJob._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [ModelJob._json_safe(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def export_optuna_report(self, model):
        report = self._json_safe(getattr(model, "optuna_report", {}))
        if not report:
            return
        out_dir = os.path.join(self.full_path, "models", "optuna_trials")
        os.makedirs(out_dir, exist_ok=True)
        row = {
            "algorithm": self.algorithm,
            "model_name": getattr(model, "model_name", self.algorithm),
            "cv": self.cv_count,
            **report,
        }
        for key, value in list(row.items()):
            if isinstance(value, (dict, list, tuple, set)):
                row[key] = json.dumps(self._json_safe(value), sort_keys=True)
        out_path = os.path.join(out_dir, f"{self.algorithm}_optuna_trials{self.cv_count}.csv")
        pd.DataFrame([row]).to_csv(out_path, index=False)
        if report.get("optuna_used"):
            logging.info(
                "%s CV_%s Optuna trials: %s run, %s complete, requested=%s, timeout=%s",
                self.algorithm,
                self.cv_count,
                report.get("trials_run"),
                report.get("trials_complete"),
                report.get("requested_trials"),
                report.get("timeout_seconds"),
            )

    def save_runtime(self):
        os.makedirs(self.full_path + '/runtime/models/' , exist_ok=True)
        runtime_file = open(self.full_path + '/runtime/models/runtime_' + self.algorithm + '_CV' + str(self.cv_count) + '.txt','w')
        runtime_file.write(str(time.time() - self.job_start_time))
        runtime_file.close()

    @staticmethod
    def export_best_params(file_name, param_grid):
        best_params_copy = param_grid
        for best in best_params_copy:
            best_params_copy[best] = [best_params_copy[best]]
        df = pd.DataFrame.from_dict(best_params_copy)
        df.to_csv(file_name, index=False)
