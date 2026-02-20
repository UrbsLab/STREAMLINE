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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.calibration import CalibratedClassifierCV


class ModelJob:
    def __init__(self, full_path, output_path, experiment_name, cv_count, outcome_label="Class",
                 instance_label=None, scoring_metric='balanced_accuracy', metric_direction='maximize', n_trials=200,
                 timeout=900, training_subsample=0, uniform_fi=False, save_plot=False, random_state=None,
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
        if os.path.exists(experiment_path + '/jobsCompleted/') is False:
            os.makedirs(experiment_path + '/jobsCompleted/')
        job_file = open(experiment_path + '/jobsCompleted/job_model_' + self.full_path.split('/')[-1]
                        + '_' + str(self.cv_count) + '_' + self.algorithm + '.txt', 'w')
        job_file.write('complete')
        job_file.close()


    def run_model(self, model):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        x_train, y_train, x_test, y_test = self.data_prep()
        model.fit(x_train, y_train, self.n_trials, self.timeout, self.feature_names)

        # optional training subsample for certain models
        if 0 < self.training_subsample < x_train.shape[0] and model.small_name in ['XGB', 'SVM', 'ANN', 'KNN']:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=self.training_subsample, random_state=self.random_state)
            for train_index, _ in sss.split(x_train, y_train):
                x_train = x_train[train_index]
                y_train = y_train[train_index]
            logging.warning('For ' + model.small_name
                            + ', training sample reduced to ' + str(x_train.shape[0]) + ' instances')

        if not os.path.exists(self.full_path + '/models/'):
            os.makedirs(self.full_path + '/models/')
        if not os.path.exists(self.full_path + '/models/pickledModels/'):
            os.makedirs(self.full_path + '/models/pickledModels/')
        # keep pickled_metrics only for residuals (regression)
        if not os.path.exists(self.full_path + '/model_evaluation/pickled_metrics/'):
            os.makedirs(self.full_path + '/model_evaluation/pickled_metrics/', exist_ok=True)
        # NEW: JSON outputs
        if not os.path.exists(self.full_path + '/model_evaluation/metrics_by_cv/'):
            os.makedirs(self.full_path + '/model_evaluation/metrics_by_cv/', exist_ok=True)
        if not os.path.exists(self.full_path + '/model_evaluation/curves_by_cv/'):
            os.makedirs(self.full_path + '/model_evaluation/curves_by_cv/', exist_ok=True)


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
        if self.uniform_fi:
            results = permutation_importance(model.model, x_train, y_train, n_repeats=10,
                                             random_state=self.random_state, scoring=self.scoring_metric)
            self.feature_importance = results.importances_mean
        else:
            try:
                self.feature_importance = model.model.feature_importances_
            except AttributeError:
                results = permutation_importance(model.model, x_train, y_train, n_repeats=10,
                                                 random_state=self.random_state, scoring=self.scoring_metric)
                self.feature_importance = results.importances_mean

        # Persist model
        # Persist model
        with open(self.full_path + '/models/pickledModels/' + self.algorithm +
                  '_' + str(self.cv_count) + '.pickle', 'wb') as file:
            pickle.dump(model.model, file)

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
            }

            # curves are not defined for regression
            curves_payload = None

            return metrics_payload, [residual_train, residual_test, y_train_pred, y_pred, y_train, y_test]

        elif model.model_type in ["Binary", "Multiclass"]:
            metric_dict, curves_dict = model.model_evaluation(x_test, y_test)

            metrics_payload = {
                "metrics": metric_dict,
                "feature_importance": fi_list,
            }
            curves_payload = curves_dict

            return metrics_payload, curves_payload


    def data_prep(self):
        train = pd.read_csv(self.train_file_path)
        test = pd.read_csv(self.test_file_path)
        if self.instance_label is not None:
            train = train.drop(self.instance_label, axis=1)
            test = test.drop(self.instance_label, axis=1)
        x_train = train.drop(self.outcome_label, axis=1).values
        y_train = train[self.outcome_label].values
        x_test = test.drop(self.outcome_label, axis=1).values
        y_test = test[self.outcome_label].values
        del train; del test
        return x_train, y_train, x_test, y_test

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
