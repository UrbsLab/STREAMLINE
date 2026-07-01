# streamline/phases/p2_impute_scale/job.py
import os
import time
import pickle
import random
import logging
import importlib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from streamline.p2_impute_scale.utils.impute_loader import load_imputer
from streamline.p2_impute_scale.utils.scale_loader import load_scaler


class ImputeAndScale:
    """
    Phase 2: Scaling and Imputation of CV Datasets
    - Backwards compatible with the original behavior
    - Optional registry-driven imputer selection
    """

    def __init__(
        self,
        cv_train_path,
        cv_test_path,
        experiment_path,
        scale_data: bool = True,
        impute_data: bool = True,
        multi_impute: bool = False,             # original flag: if True uses IterativeImputer
        overwrite_cv: bool = True,
        outcome_label: str = "Class",
        instance_label=None,
        random_state: "int | None" = None,
        # NEW (optional)
        imputer_id: "str | None" = "iterative",         # e.g., "simple", "knn", "iterative", "median_map"
        imputer_params: "dict | None" = None,    # params forwarded to registry imputer
        # NEW (optional)
        scaler_id: "str | None" = None,
        scaler_params: "dict | None" = None,
        outcome_type: "str | None" = None,
        smote: bool = False,
        smote_method: str = "auto",
        smote_sampling_strategy: "str | dict | float" = "auto",
        smote_k_neighbors: int = 5,
    ):
        self.cv_train_path = cv_train_path
        self.cv_test_path = cv_test_path
        self.experiment_path = experiment_path
        self.scale_data = scale_data
        self.impute_data = impute_data
        self.multi_impute = multi_impute
        self.overwrite_cv = overwrite_cv
        self.outcome_label = outcome_label
        self.instance_label = instance_label
        self.categorical_variables = None
        self.dataset_name = None
        self.cv_count = None
        self.random_state = random_state
        self.imputer_id = imputer_id
        self.imputer_params = imputer_params or {}
        self.scaler_id = scaler_id
        self.scaler_params = scaler_params or {}
        self.outcome_type = outcome_type
        self.smote = bool(smote)
        self.smote_method = str(smote_method or "auto").strip().lower()
        self.smote_sampling_strategy = smote_sampling_strategy
        self.smote_k_neighbors = int(smote_k_neighbors)
        

    def run(self):
        self.job_start_time = time.time()
        # Seeds for repeatability
        random.seed(self.random_state)
        np.random.seed(self.random_state)


        # Load CV fold train/test
        data_train, data_test = self.load_data()

        # Feature header (exclude outcome/instance)
        header = data_train.columns.tolist()
        header.remove(self.outcome_label)
        if self.instance_label is not None and self.instance_label in header:
            header.remove(self.instance_label)

        logging.info('Preparing Train and Test for: %s_CV_%s', self.dataset_name, self.cv_count)

        y_train = data_train[self.outcome_label]
        y_test = data_test[self.outcome_label]

        i_train = i_test = None
        if self.instance_label is not None:
            i_train = data_train[self.instance_label]
            i_test = data_test[self.instance_label]

        if self.instance_label is None:
            x_train = data_train.drop([self.outcome_label], axis=1)
            x_test = data_test.drop([self.outcome_label], axis=1)
        else:
            x_train = data_train.drop([self.outcome_label, self.instance_label], axis=1)
            x_test = data_test.drop([self.outcome_label, self.instance_label], axis=1)
        del data_train, data_test

        # Load categorical features list
        with open(os.path.join(self.experiment_path, self.dataset_name, 'exploratory', 'categorical_features.pickle'), 'rb') as f:
            self.categorical_variables = pickle.load(f)

        # Imputation
        if self.impute_data:
            logging.info('Imputing Missing Values...')
            data_counts = pd.read_csv(
                os.path.join(self.experiment_path, self.dataset_name, 'exploratory', 'DataCounts.csv'),
                na_values='NA', sep=','
            )
            missing_values = int(data_counts['Count'].values[4])
            if missing_values != 0:
                x_train, x_test = self.impute_cv_data(x_train, x_test)
                x_train = pd.DataFrame(x_train, columns=header)
                x_test = pd.DataFrame(x_test, columns=header)
            else:
                logging.info('Notice: No missing values found. Imputation skipped.')

        # Scaling
        if self.scale_data:
            logging.info('Scaling Data Values...')
            x_train, x_test = self.data_scaling(x_train, x_test)

        # SMOTE is intentionally applied after imputation/scaling and only to train.
        if self.smote:
            logging.info('Applying SMOTE to training fold...')
            x_train, y_train, i_train = self.apply_smote(x_train, y_train, i_train)

        # Reassemble
        if self.instance_label is None:
            data_train = pd.concat([pd.DataFrame(y_train, columns=[self.outcome_label]),
                                    pd.DataFrame(x_train, columns=header)], axis=1, sort=False)
            data_test = pd.concat([pd.DataFrame(y_test, columns=[self.outcome_label]),
                                   pd.DataFrame(x_test, columns=header)], axis=1, sort=False)
        else:
            data_train = pd.concat([pd.DataFrame(y_train, columns=[self.outcome_label]),
                                    pd.DataFrame(i_train, columns=[self.instance_label]),
                                    pd.DataFrame(x_train, columns=header)], axis=1, sort=False)
            data_test = pd.concat([pd.DataFrame(y_test, columns=[self.outcome_label]),
                                   pd.DataFrame(i_test, columns=[self.instance_label]),
                                   pd.DataFrame(x_test, columns=header)], axis=1, sort=False)
        del x_train, x_test

        # Export & finalize
        logging.info('Saving Processed Train and Test Data...')
        if self.impute_data or self.scale_data or self.smote:
            self.write_cv_files(data_train, data_test)

        self.save_runtime()
        logging.info('%s Phase 2 complete', self.dataset_name)
        with open(os.path.join(self.experiment_path, 'jobsCompleted',
                               f'job_preprocessing_{self.dataset_name}_{self.cv_count}.txt'), 'w') as jf:
            jf.write('complete')

    def load_data(self):
        self.dataset_name = self.cv_train_path.split('/')[-3]
        self.cv_count = self.cv_train_path.split('/')[-1].split("_")[-2]
        data_train = pd.read_csv(self.cv_train_path, na_values='NA', sep=',')
        data_test = pd.read_csv(self.cv_test_path, na_values='NA', sep=',')
        return data_train, data_test

    def impute_cv_data(self, x_train: pd.DataFrame, x_test: pd.DataFrame):
        """
        Categorical: mode imputation (train mode applied to test).
        Quantitative: either IterativeImputer (multi_impute) or registry-selected imputer;
                      fallback to 'median_map' if requested.
        """
        # 1) Categorical mode map (saved)
        mode_dict = {}
        for c in x_train.columns:
            if c in self.categorical_variables:
                train_mode = x_train[c].mode(dropna=True).iloc[0]
                x_train[c] = x_train[c].fillna(train_mode)
                mode_dict[c] = train_mode
        for c in x_test.columns:
            if c in self.categorical_variables:
                x_test[c] = x_test[c].fillna(mode_dict[c])

        out_cat = os.path.join(self.experiment_path, self.dataset_name, "impute_scale",
                               f"categorical_imputer_cv{self.cv_count}.pickle")
        os.makedirs(os.path.dirname(out_cat), exist_ok=True)
        with open(out_cat, "wb") as f:
            pickle.dump(mode_dict, f)

        # 2) Quantitative
        # If an explicit registry imputer is requested, use it:
        if self.imputer_id:
            if self.imputer_params:
                imp = load_imputer(self.imputer_id, **self.imputer_params)
            else:
                imp = load_imputer(self.imputer_id)
            imp = imp.fit(x_train.select_dtypes(include=['number']))
            Xtr_num = imp.transform(x_train.select_dtypes(include=['number']))
            Xte_num = imp.transform(x_test.select_dtypes(include=['number']))
            # put back numeric columns
            x_train.loc[:, Xtr_num.columns] = Xtr_num
            x_test.loc[:, Xte_num.columns] = Xte_num

            out_num = os.path.join(self.experiment_path, self.dataset_name, "impute_scale",
                                   f"ordinal_imputer_cv{self.cv_count}.pickle")
            with open(out_num, "wb") as f:
                pickle.dump({"id": self.imputer_id, "params": imp.get_params()}, f)
            return x_train, x_test

        # Otherwise, preserve original behavior
        else:
            if self.multi_impute:
                imputer = IterativeImputer(random_state=self.random_state, max_iter=30)
                imputer = imputer.fit(x_train.select_dtypes(include=['number']))
                Xtr_num = imputer.transform(x_train.select_dtypes(include=['number']))
                Xte_num = imputer.transform(x_test.select_dtypes(include=['number']))
                x_train.loc[:, x_train.select_dtypes(include=['number']).columns] = Xtr_num
                x_test.loc[:, x_test.select_dtypes(include=['number']).columns] = Xte_num
                out_num = os.path.join(self.experiment_path, self.dataset_name, 'impute_scale',
                                    f'ordinal_imputer_cv{self.cv_count}.pickle')
                with open(out_num, 'wb') as f:
                    pickle.dump(imputer, f)
            else:
                # median_map fallback for numeric
                median_dict = {}
                num_cols = x_train.select_dtypes(include=['number']).columns
                for c in num_cols:
                    m = x_train[c].median()
                    x_train[c] = x_train[c].fillna(m)
                    median_dict[c] = m
                for c in num_cols:
                    x_test[c] = x_test[c].fillna(median_dict[c])

                out_num = os.path.join(self.experiment_path, self.dataset_name, 'impute_scale',
                                    f'ordinal_imputer_cv{self.cv_count}.pickle')
                with open(out_num, 'wb') as f:
                    pickle.dump(median_dict, f)

        return x_train, x_test

    def data_scaling(self, x_train, x_test):
        decimal_places = 7

        if self.scaler_id:
            # use registry scaler (fits on numeric columns internally)
            scaler = load_scaler(self.scaler_id, **(self.scaler_params or {})).fit(x_train)
            x_train = scaler.transform(x_train).round(decimal_places)
            x_test  = scaler.transform(x_test).round(decimal_places)
            out_scl = os.path.join(self.experiment_path, self.dataset_name,
                                   'impute_scale', f'scaler_cv{self.cv_count}.pickle')
            with open(out_scl, 'wb') as f:
                pickle.dump({"id": self.scaler_id, "params": scaler.get_params()}, f)
            return x_train, x_test

        # original default (StandardScaler on numeric only)
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(x_train.select_dtypes(include=['number']))

            num_cols = x_train.select_dtypes(include=['number']).columns
            x_train_num = pd.DataFrame(
                scaler.transform(x_train[num_cols]).round(decimal_places),
                columns=num_cols, index=x_train.index
            )
            x_test_num = pd.DataFrame(
                scaler.transform(x_test[num_cols]).round(decimal_places),
                columns=num_cols, index=x_test.index
            )
            x_train = x_train.astype(x_train_num.dtypes.to_dict())
            x_test = x_test.astype(x_test_num.dtypes.to_dict())
            x_train.loc[:, num_cols] = x_train_num
            x_test.loc[:, num_cols] = x_test_num

        out_scl = os.path.join(self.experiment_path, self.dataset_name,
                               'impute_scale', f'scaler_cv{self.cv_count}.pickle')
        with open(out_scl, 'wb') as f:
            pickle.dump(scaler, f)
        return x_train, x_test

    def apply_smote(self, x_train: pd.DataFrame, y_train: pd.Series, i_train: "pd.Series | None"):
        if self.outcome_type == "Continuous":
            raise ValueError("SMOTE is only supported for Binary and Multiclass outcomes, not Continuous outcomes.")

        class_counts = y_train.value_counts(dropna=False).to_dict()
        if len(class_counts) < 2:
            logging.warning("SMOTE skipped for %s CV%s because only one class is present.", self.dataset_name, self.cv_count)
            self.save_smote_metadata("none", False, class_counts, class_counts, [], "only one class present")
            return x_train, y_train, i_train

        smallest_class_count = min(class_counts.values())
        largest_class_count = max(class_counts.values())
        if smallest_class_count == largest_class_count:
            logging.info("SMOTE skipped for %s CV%s because classes are already balanced.", self.dataset_name, self.cv_count)
            self.save_smote_metadata("none", False, class_counts, class_counts, [], "classes already balanced")
            return x_train, y_train, i_train
        if smallest_class_count < 2:
            logging.warning("SMOTE skipped for %s CV%s because at least one class has fewer than 2 rows.", self.dataset_name, self.cv_count)
            self.save_smote_metadata("none", False, class_counts, class_counts, [], "class count below 2")
            return x_train, y_train, i_train

        try:
            from imblearn.over_sampling import SMOTE, SMOTENC
        except ImportError as exc:
            raise ImportError(
                "SMOTE requires the optional dependency imbalanced-learn. "
                "Install it with `pip install imbalanced-learn` or use the project requirements."
            ) from exc

        categorical_columns = [
            col for col in (self.categorical_variables or [])
            if col in x_train.columns
        ]
        categorical_indices = [x_train.columns.get_loc(col) for col in categorical_columns]
        resolved_method = self.resolve_smote_method(categorical_indices, x_train)
        neighbors = min(max(int(self.smote_k_neighbors), 1), smallest_class_count - 1)

        if resolved_method == "smotenc":
            sampler = SMOTENC(
                categorical_features=categorical_indices,
                sampling_strategy=self.smote_sampling_strategy,
                k_neighbors=neighbors,
                random_state=self.random_state,
            )
        else:
            sampler = SMOTE(
                sampling_strategy=self.smote_sampling_strategy,
                k_neighbors=neighbors,
                random_state=self.random_state,
            )

        original_rows = len(x_train)
        x_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
        x_resampled = pd.DataFrame(x_resampled, columns=x_train.columns)
        y_resampled = pd.Series(y_resampled, name=self.outcome_label)
        self.restore_resampled_dtypes(x_resampled, x_train, categorical_columns)

        if i_train is not None:
            original_ids = list(pd.Series(i_train).astype(str))
            synthetic_count = len(x_resampled) - original_rows
            synthetic_ids = [
                f"{self.dataset_name}_CV_{self.cv_count}_SMOTE_{idx + 1:06d}"
                for idx in range(synthetic_count)
            ]
            i_resampled = pd.Series(original_ids + synthetic_ids, name=self.instance_label)
        else:
            i_resampled = None

        after_counts = y_resampled.value_counts(dropna=False).to_dict()
        self.save_smote_metadata(
            resolved_method,
            True,
            class_counts,
            after_counts,
            categorical_columns,
            None,
            neighbors,
            len(x_resampled) - original_rows,
        )
        logging.info(
            "SMOTE complete for %s CV%s: %s rows -> %s rows",
            self.dataset_name,
            self.cv_count,
            original_rows,
            len(x_resampled),
        )

        return (
            x_resampled.reset_index(drop=True),
            y_resampled.reset_index(drop=True),
            None if i_resampled is None else i_resampled.reset_index(drop=True),
        )

    def resolve_smote_method(self, categorical_indices, x_train):
        if self.smote_method not in {"auto", "smote", "smotenc"}:
            raise ValueError("smote_method must be one of: auto, smote, smotenc")

        if self.smote_method == "auto":
            if categorical_indices:
                if len(categorical_indices) == len(x_train.columns):
                    raise ValueError("SMOTENC requires at least one non-categorical feature.")
                return "smotenc"
            return "smote"

        if self.smote_method == "smotenc" and not categorical_indices:
            raise ValueError("smote_method='smotenc' requires at least one categorical feature.")

        if self.smote_method == "smotenc" and len(categorical_indices) == len(x_train.columns):
            raise ValueError("SMOTENC requires at least one non-categorical feature.")

        return self.smote_method

    def restore_resampled_dtypes(self, x_resampled, x_original, categorical_columns):
        categorical_set = set(categorical_columns)
        for col in x_resampled.columns:
            if col in categorical_set:
                try:
                    x_resampled[col] = x_resampled[col].astype(x_original[col].dtype)
                except Exception:
                    pass
            else:
                x_resampled[col] = pd.to_numeric(x_resampled[col], errors="coerce")

    def save_smote_metadata(
        self,
        method,
        applied,
        class_counts_before,
        class_counts_after,
        categorical_columns,
        reason=None,
        k_neighbors=None,
        synthetic_rows=0,
    ):
        out_path = os.path.join(
            self.experiment_path,
            self.dataset_name,
            "impute_scale",
            f"smote_cv{self.cv_count}.pickle",
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "applied": bool(applied),
                    "method": method,
                    "requested_method": self.smote_method,
                    "sampling_strategy": self.smote_sampling_strategy,
                    "k_neighbors": k_neighbors,
                    "synthetic_rows": int(synthetic_rows),
                    "categorical_features": list(categorical_columns),
                    "class_counts_before": dict(class_counts_before),
                    "class_counts_after": dict(class_counts_after),
                    "reason": reason,
                },
                f,
            )


    def write_cv_files(self, data_train, data_test):
        if self.overwrite_cv:
            os.remove(self.cv_train_path)
            os.remove(self.cv_test_path)
        else:
            os.rename(self.cv_train_path,
                      os.path.join(self.experiment_path, self.dataset_name,
                                   'CVDatasets', f'{self.dataset_name}_CVOnly_{self.cv_count}_Train.csv'))
            os.rename(self.cv_test_path,
                      os.path.join(self.experiment_path, self.dataset_name,
                                   'CVDatasets', f'{self.dataset_name}_CVOnly_{self.cv_count}_Test.csv'))
        data_train.to_csv(self.cv_train_path, index=False)
        data_test.to_csv(self.cv_test_path, index=False)

    def save_runtime(self):
        rt_dir = os.path.join(self.experiment_path, self.dataset_name, 'runtime')
        os.makedirs(rt_dir, exist_ok=True)
        with open(os.path.join(rt_dir, f'runtime_preprocessing{self.cv_count}.txt'), 'w+') as f:
            f.write(str(time.time() - self.job_start_time))
