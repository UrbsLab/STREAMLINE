import copy
import os
import logging
import random
import time
import numpy as np
import optuna
import pandas as pd
from streamline.utils.job import Job


class ModelJob(Job):
    def __init__(self, full_path, output_path, experiment_name, cv_count, class_label="Class",
                 instance_label=None, scoring_metric='primary_metric', n_trails=None, timeout=None,
                 save_plot=False, random_state=None):
        super().__init__()
        self.algorithm = ""
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.class_label = class_label
        self.instance_label = instance_label
        self.scoring_metric = scoring_metric
        self.full_path = full_path
        self.cv_count = cv_count
        self.data_name = self.full_path.split('/')[-1]
        self.train_file_path = self.full_path + '/CVDatasets/' + self.data_name \
                               + '_CV_' + str(self.cv_count) + '_Train.csv'
        self.test_file_path = self.full_path + '/CVDatasets/' + self.data_name \
                              + '_CV_' + str(self.cv_count) + '_Test.csv'

        feature_names = pd.read_csv(self.test_file_path).columns.values.tolist()
        if self.instance_label is not None:
            feature_names.remove(self.instance_label)
        feature_names.remove(self.class_label)
        self.feature_names = feature_names

        # Argument checks
        if not os.path.exists(self.output_path):
            raise Exception("Output path must exist (from phase 1) before phase 5 can begin")
        if not os.path.exists(self.output_path + '/' + self.experiment_name):
            raise Exception("Experiment must exist (from phase 1) before phase 5 can begin")

        self.n_trails = n_trails
        self.timeout = timeout
        self.random_state = random_state
        self.save_plot = save_plot
        self.param_grid = None

    def run(self, model):
        self.job_start_time = time.time()  # for tracking phase runtime
        logging.info('Running ' + str(self.algorithm) + ' on ' + str(self.train_file_path))
        self.run_model(model)

        # Save runtime of ml algorithm training and evaluation
        self.save_runtime()

        # Print phase completion
        logging.info(self.full_path.split('/')[-1] + " [CV_" + str(self.cv_count) + "] (" + self.algorithm
                     + ") training complete. ------------------------------------")
        experiment_path = '/'.join(self.full_path.split('/')[:-1])
        job_file = open(experiment_path + '/jobsCompleted/job_model_' + self.full_path.split('/')[-1]
                        + '_' + str(self.cv_count) + '_' + self.algorithm + '.txt', 'w')
        job_file.write('complete')
        job_file.close()

    def run_model(self, model):
        # Set random seeds for reproducibility
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        # Load training and testing datasets separating features from outcome for scikit-learn-based modeling
        x_train, y_train, x_test, y_test = self.data_prep()
        model.optmize(x_train, y_train, self.n_trails, self.timeout)

        if not model.is_single:
            if self.save_plot:
                try:
                    fig = optuna.visualization.plot_parallel_coordinate(model.study)
                    fig.write_image(self.full_path + '/models/LR_ParamOptimization_' + str() + '.png')
                except Exception:
                    logging.warning('Warning: Optuna Optimization Visualization Generation Failed for '
                                    'Due to Known Release Issue.  '
                                    'Please install Optuna 2.0.0 to avoid this issue.')
            # Print results and hyperparamter values for best hyperparameter sweep trial
            logging.info('Best trial:')
            best_trial = model.study.best_trial
            logging.info('  Value: ', best_trial.value)
            logging.info('  Params: ')
            for key, value in best_trial.params.items():
                logging.info('    {}: {}'.format(key, value))
            # Specify model with optimized hyperparameters
            # Export final model hyperparamters to csv file
            self.export_best_params(self.full_path + '/models/_bestparams' + str(self.cv_count) + '.csv',
                                    model.study.best_trial.params)
        else:  # Specify hyperparameter values (no sweep)
            params = copy.deepcopy(model.param_grid)
            for key, value in model.param_grid.items():
                params[key] = value[0]
            self.export_best_params(self.full_path + '/models/LR_usedparams' + str(self.cv_count) + '.csv',
                                    params)  # Export final model hyperparamters to csv file

    def data_prep(self):
        """
        Loads target cv training dataset, separates class from features and removes instance labels.
        """
        train = pd.read_csv(self.train_file_path)
        if self.instance_label is not None:
            train = train.drop(self.instance_label, axis=1)
        x_train = train.drop(self.class_label, axis=1).values
        y_train = train[self.class_label].values
        del train  # memory cleanup
        test = pd.read_csv(self.test_file_path)
        if self.instance_label != 'None':
            test = test.drop(self.instance_label, axis=1)
        x_test = test.drop(self.class_label, axis=1).values
        y_test = test[self.class_label].values
        del test  # memory cleanup
        return x_train, y_train, x_test, y_test

    def save_runtime(self):
        """
        Save ML algorithm training and evaluation runtime for this phase.
        """
        runtime_file = open(self.full_path + '/runtime/runtime_' + self.algorithm + '_CV' + str(self.cv_count) + '.txt',
                            'w')
        runtime_file.write(str(time.time() - self.job_start_time))
        runtime_file.close()

    @staticmethod
    def export_best_params(file_name, param_grid):
        """
        Exports the best hyperparameter scores to output file.
        """
        best_params_copy = param_grid
        for best in best_params_copy:
            best_params_copy[best] = [best_params_copy[best]]
        df = pd.DataFrame.from_dict(best_params_copy)
        df.to_csv(file_name, index=False)
